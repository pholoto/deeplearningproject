from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_loader import Vectorizers, create_vectorizations, make_dataset
from utils.vectorizer import save_vectorizer

from .baseline_utils import TextPair, compute_prediction_metrics

START_TOKEN = "[start]"
END_TOKEN = "[end]"


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        pad_id: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if bidirectional and hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even when using bidirectional encoder")
        self.encoder_hidden_dim = hidden_dim // self.num_directions
        self.decoder_hidden_dim = hidden_dim
        self.src_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(
            emb_dim,
            self.encoder_hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_embed(src)
        _, (hidden, cell) = self.encoder(src_emb)
        if self.bidirectional:
            hidden = self._merge_directions(hidden)
            cell = self._merge_directions(cell)
        tgt_emb = self.tgt_embed(tgt_in)
        decoder_out, _ = self.decoder(tgt_emb, (hidden, cell))
        return self.output_proj(decoder_out)

    def _merge_directions(self, state: torch.Tensor) -> torch.Tensor:
        num_layers = self.encoder.num_layers
        batch = state.size(1)
        state = state.view(num_layers, self.num_directions, batch, self.encoder_hidden_dim)
        state = state.transpose(1, 2).reshape(num_layers, batch, self.decoder_hidden_dim)
        return state


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vectorizers: Vectorizers,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 3,
) -> Seq2SeqLSTM:
    vocab_size = max(vectorizers.source.vocab_size(), vectorizers.target.vocab_size())
    model = Seq2SeqLSTM(
        vocab_size=vocab_size,
        emb_dim=256,
        hidden_dim=512,
        pad_id=vectorizers.target.pad_id(),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vectorizers.target.pad_id())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in train_loader:
            src = src.to(device)
            teacher_inputs = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            logits = model(src, teacher_inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        val_loss = evaluate_lstm_loss(model, val_loader, loss_fn, device)
        print(f"LSTM epoch {epoch}/{epochs}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered after {epoch} epochs (best val_loss={best_val_loss:.4f})"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


@torch.no_grad()
def evaluate_lstm_loss(
    model: Seq2SeqLSTM,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        teacher_inputs = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        logits = model(src, teacher_inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
        total += loss.item()
    return total / len(dataloader)


@torch.no_grad()
def lstm_predictions(
    model: Seq2SeqLSTM,
    dataloader: DataLoader,
    vectorizers: Vectorizers,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    model.eval()
    pad_id = vectorizers.target.pad_id()
    preds: List[str] = []
    refs: List[str] = []
    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        teacher_inputs = tgt_in.to(device)
        logits = model(src, teacher_inputs)
        pred_ids = logits.argmax(dim=-1).cpu()
        for pred_seq, tgt_seq in zip(pred_ids, tgt_out):
            ref_text = decode_sequence(tgt_seq.tolist(), vectorizers.target, pad_id)
            limit = len(ref_text.split()) if ref_text else None
            preds.append(
                decode_sequence(pred_seq.tolist(), vectorizers.target, pad_id, max_tokens=limit)
            )
            refs.append(ref_text)
    return preds, refs


def decode_sequence(
    ids: List[int],
    vectorizer,
    pad_id: int,
    max_tokens: Optional[int] = None,
) -> str:
    tokens: List[str] = []
    for idx in ids:
        if idx == pad_id:
            continue
        token = vectorizer.id_to_token(idx)
        if token == START_TOKEN:
            continue
        if token == END_TOKEN:
            break
        tokens.append(token)
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return " ".join(tokens).strip()


@torch.no_grad()
def lstm_greedy_decode(
    model: Seq2SeqLSTM,
    sentence: str,
    vectorizers: Vectorizers,
    device: torch.device,
) -> str:
    model.eval()
    src_tensor = vectorizers.source.encode(sentence).unsqueeze(0).to(device)
    src_emb = model.src_embed(src_tensor)
    _, (hidden, cell) = model.encoder(src_emb)
    if model.bidirectional:
        hidden = model._merge_directions(hidden)
        cell = model._merge_directions(cell)

    pad_id = vectorizers.target.pad_id()
    start_id = vectorizers.target.token_to_id(START_TOKEN)
    input_id = torch.tensor([[start_id]], device=device)
    generated: List[str] = []
    source_tokens = sentence.split()
    target_len = len(source_tokens)
    max_steps = min(
        vectorizers.target.sequence_length - 1,
        target_len if target_len else vectorizers.target.sequence_length - 1,
    )
    repetition_count = 0
    last_token: Optional[str] = None

    for _ in range(max_steps):
        tgt_emb = model.tgt_embed(input_id)
        output, (hidden, cell) = model.decoder(tgt_emb, (hidden, cell))
        logits = model.output_proj(output[:, -1, :])
        next_token = logits.argmax(dim=-1)
        token_id = int(next_token.item())
        if token_id == pad_id:
            break
        token = vectorizers.target.id_to_token(token_id)
        if token == START_TOKEN:
            input_id = next_token.unsqueeze(0)
            continue
        if token == END_TOKEN:
            break
        if token == last_token:
            repetition_count += 1
        else:
            repetition_count = 0
        last_token = token
        if repetition_count >= 3 or token == "<unk>":
            break
        generated.append(token)
        if target_len and len(generated) >= target_len:
            break
        input_id = next_token.unsqueeze(0)

    if target_len and len(generated) < target_len:
        generated.extend(source_tokens[len(generated):target_len])

    return " ".join(generated).strip()


def _build_dataloaders(
    vectorizers: Vectorizers,
    pairs: Tuple[List[TextPair], List[TextPair], List[TextPair]],
    batch_size: int,
) -> Dict[str, DataLoader]:
    train_pairs, val_pairs, test_pairs = pairs
    return {
        "train": make_dataset(train_pairs, vectorizers, batch_size=batch_size, shuffle=True),
        "val": make_dataset(val_pairs, vectorizers, batch_size=batch_size, shuffle=False),
        "test": make_dataset(test_pairs, vectorizers, batch_size=batch_size, shuffle=False),
    }


def run_lstm_baseline(
    train_pairs: List[TextPair],
    val_pairs: List[TextPair],
    test_pairs: List[TextPair],
    sequence_length: int,
    vocab_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_dir: str,
    patience: int,
    eval_splits: Optional[Mapping[str, List[TextPair]]] = None,
) -> Dict[str, object]:
    vectorizers = create_vectorizations(train_pairs, sequence_length=sequence_length, vocab_size=vocab_size)
    dataloaders = _build_dataloaders(vectorizers, (train_pairs, val_pairs, test_pairs), batch_size)

    model = train_lstm(
        dataloaders["train"],
        dataloaders["val"],
        vectorizers,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    eval_pairs: Dict[str, List[TextPair]] = {"test": test_pairs}
    if eval_splits:
        eval_pairs.update(eval_splits)

    metrics: Dict[str, Dict[str, float]] = {}
    for split_name, pairs in eval_pairs.items():
        if not pairs:
            continue
        if split_name == "test":
            loader = dataloaders["test"]
        else:
            loader = make_dataset(pairs, vectorizers, batch_size=batch_size, shuffle=False)
        preds, refs = lstm_predictions(model, loader, vectorizers, device)
        metrics[split_name] = compute_prediction_metrics(preds, refs)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    lstm_model_path = checkpoint_path / "lstm.pt"
    torch.save(model.state_dict(), lstm_model_path)
    save_vectorizer(vectorizers.source, str(checkpoint_path / "lstm_source_vectorizer.pkl"))
    save_vectorizer(vectorizers.target, str(checkpoint_path / "lstm_target_vectorizer.pkl"))

    return {
        "metrics": metrics,
        "checkpoint": str(lstm_model_path),
    }


__all__ = [
    "Seq2SeqLSTM",
    "run_lstm_baseline",
    "lstm_greedy_decode",
]
