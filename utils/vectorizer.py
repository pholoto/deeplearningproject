import pickle
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    def __init__(self, max_tokens: int, reserved: Optional[List[str]] = None) -> None:
        reserved = reserved or []
        ordered = [PAD_TOKEN, UNK_TOKEN] + [t for t in reserved if t not in {PAD_TOKEN, UNK_TOKEN}]
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(ordered)}
        self.id_to_token: List[str] = ordered.copy()
        self.max_tokens = max_tokens

    def build(self, texts: Iterable[Iterable[str]]) -> None:
        counter: Counter[str] = Counter()
        for tokens in texts:
            counter.update(tokens)
        available = self.max_tokens - len(self.id_to_token)
        if available <= 0:
            return
        for token, _ in counter.most_common():
            if token in self.token_to_id:
                continue
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
            if len(self.id_to_token) >= self.max_tokens:
                break

    def lookup(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Vocabulary":
        id_to_token = list(state["id_to_token"])
        max_tokens = int(state.get("max_tokens", 0)) or len(id_to_token)
        vocab = cls(max_tokens=max_tokens)
        vocab.token_to_id = {k: int(v) for k, v in state["token_to_id"].items()}
        vocab.id_to_token = id_to_token
        return vocab


class TextVectorizer:
    def __init__(
        self,
        sequence_length: int,
        vocab_size: int,
        reserved_tokens: Optional[List[str]] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.vocab = Vocabulary(vocab_size, reserved_tokens)
        self.is_adapted = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.strip().split()

    def adapt(self, texts: Iterable[str]) -> None:
        tokenised = (self.tokenize(t) for t in texts)
        self.vocab.build(tokenised)
        self.is_adapted = True

    def encode(self, text: str) -> torch.Tensor:
        if not self.is_adapted:
            raise RuntimeError("Vectorizer has not been adapted yet")
        tokens = self.tokenize(text)
        ids = [self.vocab.lookup(tok) for tok in tokens]
        if len(ids) < self.sequence_length:
            ids.extend([self.vocab.lookup(PAD_TOKEN)] * (self.sequence_length - len(ids)))
        else:
            ids = ids[: self.sequence_length]
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: Iterable[str]) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if not encoded:
            raise ValueError("No texts provided to batch_encode")
        return torch.stack(encoded, dim=0)

    def vocab_size(self) -> int:
        return len(self.vocab.id_to_token)

    def pad_id(self) -> int:
        return self.vocab.lookup(PAD_TOKEN)

    def token_to_id(self, token: str) -> int:
        return self.vocab.lookup(token)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.vocab.id_to_token):
            return self.vocab.id_to_token[idx]
        return UNK_TOKEN

    def state_dict(self) -> Dict[str, Any]:
        if not self.is_adapted:
            raise RuntimeError("Cannot serialise an unadapted vectorizer")
        return {
            "sequence_length": self.sequence_length,
            "vocab": self.vocab.state_dict(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TextVectorizer":
        seq_len = int(state["sequence_length"])
        vocab_state = state["vocab"]
        vocab_obj = Vocabulary.from_state(vocab_state)
        vectorizer = cls(sequence_length=seq_len, vocab_size=vocab_obj.max_tokens)
        vectorizer.vocab = vocab_obj
        vectorizer.is_adapted = True
        return vectorizer


def save_vectorizer(vectorizer: TextVectorizer, file_path: str) -> None:
    state = vectorizer.state_dict()
    with open(file_path, "wb") as f:
        pickle.dump(state, f)


def load_vectorizer(file_path: str) -> TextVectorizer:
    with open(file_path, "rb") as f:
        state = pickle.load(f)
    return TextVectorizer.from_state(state)


def shift_target(target_batch: torch.Tensor) -> torch.Tensor:
    pad_id = 0
    shifted = torch.roll(target_batch, shifts=-1, dims=1)
    shifted[:, -1] = pad_id
    return shifted
