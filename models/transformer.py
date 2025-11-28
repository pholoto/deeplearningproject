from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_loader import Vectorizers, create_vectorizations, load_data, make_dataset
from utils.positional_embedding import PositionalEmbedding
from utils.sam import SAM
from utils.vectorizer import save_vectorizer

from .baseline_utils import compute_prediction_metrics, strip_special_tokens

TextPair = Tuple[str, str]


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(embed_dim, dense_dim)
        self.linear2 = nn.Linear(dense_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(inputs, inputs, inputs, key_padding_mask=padding_mask)
        x = self.norm1(inputs + self.dropout(attn_output))
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(ff))


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(embed_dim, dense_dim)
        self.linear2 = nn.Linear(dense_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    @staticmethod
    def _generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = tgt.size(1)
        causal_mask = self._generate_causal_mask(seq_len, tgt.device)
        attn_output, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=causal_mask,
            key_padding_mask=tgt_padding_mask,
        )
        x = self.norm1(tgt + self.dropout(attn_output))
        cross_output, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
        )
        x = self.norm2(x + self.dropout(cross_output))
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        return self.norm3(x + self.dropout(ff))


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        embed_dim: int = 256,
        dense_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_embedding = PositionalEmbedding(src_seq_len, vocab_size, embed_dim, dropout=dropout)
        self.tgt_embedding = PositionalEmbedding(tgt_seq_len, vocab_size, embed_dim, dropout=dropout)
        self.encoder = TransformerEncoder(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout=dropout)
        self.decoder = TransformerDecoder(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout=dropout)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt_in)
        memory = self.encoder(src_emb, padding_mask=src_padding_mask)
        decoded = self.decoder(
            tgt_emb,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask,
        )
        return self.output_proj(decoded)


@dataclass
class TrainConfig:
    dataset_path: str = "data/wiki_pairs.txt"
    sequence_length: int = 50
    vocab_size: int = 15000
    data_seed: Optional[int] = 42
    embed_dim: int = 256
    dense_dim: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 5
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = False


def _build_dataloaders(
    vectorizers: Vectorizers,
    config: TrainConfig,
    splits: Tuple[List[TextPair], List[TextPair], List[TextPair]],
) -> Dict[str, DataLoader]:
    train_pairs, val_pairs, test_pairs = splits
    return {
        "train": make_dataset(train_pairs, vectorizers, batch_size=config.batch_size, shuffle=True),
        "val": make_dataset(val_pairs, vectorizers, batch_size=config.batch_size, shuffle=False),
        "test": make_dataset(test_pairs, vectorizers, batch_size=config.batch_size, shuffle=False),
    }


def _compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    loss_fn: nn.Module,
) -> Tuple[torch.Tensor, float]:
    vocab = logits.size(-1)
    logits_flat = logits.view(-1, vocab)
    targets_flat = targets.view(-1)
    loss = loss_fn(logits_flat, targets_flat)

    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        mask = targets_flat.ne(pad_id)
        correct = preds.eq(targets_flat) & mask
        total = mask.sum().item()
        accuracy = correct.sum().item() / total if total else 0.0
    return loss, accuracy


def train_epoch(
    model: TransformerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    pad_id: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        teacher_inputs = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        src_mask = src.eq(pad_id)
        tgt_mask = teacher_inputs.eq(pad_id)

        optimizer.zero_grad()
        logits = model(src, teacher_inputs, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
        loss, acc = _compute_loss_and_metrics(logits, tgt_out, pad_id, loss_fn)

        if isinstance(optimizer, SAM):
            loss.backward()

            def closure():
                optimizer.zero_grad()
                logits2 = model(src, teacher_inputs, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
                loss2, _ = _compute_loss_and_metrics(logits2, tgt_out, pad_id, loss_fn)
                loss2.backward()
                return loss2

            optimizer.step(closure)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        batches += 1
    return total_loss / batches, total_acc / batches if batches else 0.0


@torch.no_grad()
def transformer_greedy_decode(
    model: TransformerModel,
    sentence: str,
    vectorizers: Vectorizers,
    device: torch.device,
) -> str:
    model.eval()
    src_ids = vectorizers.source.encode(sentence).unsqueeze(0).to(device)
    pad_id = vectorizers.target.pad_id()
    start_id = vectorizers.target.token_to_id("[start]")

    tgt_seq_len = vectorizers.target.sequence_length
    tgt_ids = torch.full((1, tgt_seq_len), pad_id, dtype=torch.long, device=device)
    tgt_ids[0, 0] = start_id

    generated: List[str] = []
    target_len = len(sentence.split())
    repetition = 0
    last_tok: Optional[str] = None

    for step in range(1, min(tgt_seq_len, target_len + 1)):
        src_mask = src_ids.eq(vectorizers.source.pad_id())
        tgt_mask = tgt_ids.eq(pad_id)
        logits = model(src_ids, tgt_ids, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
        pred_id = int(logits[:, step - 1, :].argmax(dim=-1).item())
        if pred_id == pad_id:
            break
        tok = vectorizers.target.id_to_token(pred_id)
        if tok == "[start]":
            tgt_ids[0, step] = pred_id
            continue
        if tok == "[end]":
            break
        if tok == last_tok:
            repetition += 1
        else:
            repetition = 0
        last_tok = tok
        if repetition >= 2 or tok == "<unk>":
            break
        generated.append(tok)
        tgt_ids[0, step] = pred_id
        if tok.strip() and tok.strip()[-1] in ".?!":
            break
        if target_len and len(generated) >= target_len:
            break

    return " ".join(generated).strip()


@torch.no_grad()
def evaluate(
    model: TransformerModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    pad_id: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    for src, tgt_in, tgt_out in dataloader:
        src = src.to(device)
        teacher_inputs = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        src_mask = src.eq(pad_id)
        tgt_mask = teacher_inputs.eq(pad_id)

        logits = model(src, teacher_inputs, src_padding_mask=src_mask, tgt_padding_mask=tgt_mask)
        loss, acc = _compute_loss_and_metrics(logits, tgt_out, pad_id, loss_fn)
        total_loss += loss.item()
        total_acc += acc
        batches += 1
    return total_loss / batches if batches else 0.0, total_acc / batches if batches else 0.0


def _decode_ids(ids: Sequence[int], vectorizer) -> str:
    tokens: List[str] = []
    pad_id = vectorizer.pad_id()
    for idx in ids:
        if idx == pad_id:
            continue
        tok = vectorizer.id_to_token(int(idx))
        if tok in ("[start]", "[end]"):
            continue
        tokens.append(tok)
    return " ".join(tokens).strip()


def _teacher_forced_metrics(
    model: TransformerModel,
    vectorizers: Vectorizers,
    device: torch.device,
    pairs: Sequence[TextPair],
    batch_size: int,
) -> Dict[str, float]:
    dataloader = make_dataset(pairs, vectorizers, batch_size=batch_size, shuffle=False)
    src_pad = vectorizers.source.pad_id()
    tgt_pad = vectorizers.target.pad_id()
    predictions: List[str] = []
    references: List[str] = []
    model.eval()
    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_mask = src.eq(src_pad)
            tgt_mask = tgt_in.eq(tgt_pad)
            logits = model(
                src,
                tgt_in,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask,
            )
            pred_ids = logits.argmax(dim=-1).cpu()
            tgt_cpu = tgt_out.cpu()
            for pred_seq, ref_seq in zip(pred_ids, tgt_cpu):
                predictions.append(_decode_ids(pred_seq.tolist(), vectorizers.target))
                references.append(_decode_ids(ref_seq.tolist(), vectorizers.target))
    return compute_prediction_metrics(predictions, references)


def train_model(
    config: TrainConfig,
    data_splits: Optional[Tuple[List[TextPair], List[TextPair], List[TextPair]]] = None,
    eval_splits: Optional[Mapping[str, List[TextPair]]] = None,
) -> Dict[str, object]:
    device = torch.device(config.device)
    if data_splits is None:
        train_pairs, val_pairs, test_pairs = load_data(
            config.dataset_path,
            shuffle=True,
            seed=config.data_seed,
        )
    else:
        train_pairs, val_pairs, test_pairs = data_splits

    vectorizers = create_vectorizations(
        train_pairs,
        sequence_length=config.sequence_length,
        vocab_size=config.vocab_size,
    )

    dataloaders = _build_dataloaders(vectorizers, config, (train_pairs, val_pairs, test_pairs))

    vocab_size = max(vectorizers.source.vocab_size(), vectorizers.target.vocab_size())
    model = TransformerModel(
        vocab_size=vocab_size,
        src_seq_len=vectorizers.source.sequence_length,
        tgt_seq_len=vectorizers.target.sequence_length,
        embed_dim=config.embed_dim,
        dense_dim=config.dense_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vectorizers.target.pad_id())
    if config.use_sam:
        optimizer = SAM(
            model.parameters(),
            torch.optim.Adam,
            rho=config.sam_rho,
            adaptive=config.sam_adaptive,
            lr=config.learning_rate,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            dataloaders["train"],
            optimizer,
            loss_fn,
            vectorizers.target.pad_id(),
            device,
        )
        val_loss, val_acc = evaluate(
            model,
            dataloaders["val"],
            loss_fn,
            vectorizers.target.pad_id(),
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(model, vectorizers, config)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(
                    f"Early stopping: no improvement for {config.patience} epochs "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_acc = evaluate(
        model,
        dataloaders["test"],
        loss_fn,
        vectorizers.target.pad_id(),
        device,
    )
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.4f}")

    eval_pairs: Dict[str, List[TextPair]] = {"test": test_pairs}
    if eval_splits:
        eval_pairs.update(eval_splits)

    metrics: Dict[str, Dict[str, float]] = {}
    for split_name, pairs in eval_pairs.items():
        if not pairs:
            continue
        metrics[split_name] = _teacher_forced_metrics(
            model,
            vectorizers,
            device,
            pairs,
            batch_size=config.batch_size,
        )

    return {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "train_acc": history["train_acc"][-1] if history["train_acc"] else float("nan"),
        "val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "val_acc": history["val_acc"][-1] if history["val_acc"] else float("nan"),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "metrics": metrics,
        "checkpoint": str(Path(config.checkpoint_dir) / _checkpoint_filename(config)),
    }


def _checkpoint_filename(config: TrainConfig) -> str:
    return "transformer_sam.pt" if config.use_sam else "transformer.pt"


def _save_checkpoint(model: TransformerModel, vectorizers: Vectorizers, config: TrainConfig) -> None:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / _checkpoint_filename(config)
    torch.save(model.state_dict(), model_path)

    save_vectorizer(vectorizers.source, str(checkpoint_dir / "source_vectorizer.pkl"))
    save_vectorizer(vectorizers.target, str(checkpoint_dir / "target_vectorizer.pkl"))


__all__ = [
    "TransformerModel",
    "TransformerEncoder",
    "TransformerDecoder",
    "TrainConfig",
    "train_model",
    "transformer_greedy_decode",
]
