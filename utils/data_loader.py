import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .vectorizer import TextVectorizer


TextPair = Tuple[str, str]


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def get_text_pairs(file_path: str, limit: Optional[int] = None) -> List[TextPair]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    pairs: List[TextPair] = []
    for line in lines:
        if "\t" not in line:
            continue
        stripped, original = line.split("\t", 1)
        src_tokens = stripped.split()
        tgt_tokens = original.split()
        if len(src_tokens) != len(tgt_tokens):
            continue
        pairs.append((stripped, "[start] " + original + " [end]"))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def split_pairs(
    pairs: Sequence[TextPair],
    ratio: float = 0.15,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    if not 0.0 < ratio < 0.5:
        raise ValueError("ratio must be between 0 and 0.5 (exclusive)")
    pairs_list = list(pairs)
    if shuffle:
        if seed is None:
            random.shuffle(pairs_list)
        else:
            random.Random(seed).shuffle(pairs_list)
    num_val = int(ratio * len(pairs_list))
    num_train = len(pairs_list) - 2 * num_val
    if num_train <= 0:
        raise ValueError("Not enough pairs to split into train/val/test")
    train_pairs = pairs_list[:num_train]
    val_pairs = pairs_list[num_train:num_train + num_val]
    test_pairs = pairs_list[num_train + num_val:]
    return train_pairs, val_pairs, test_pairs


def load_data(
    file_path: str,
    limit: Optional[int] = None,
    ratio: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[TextPair], List[TextPair], List[TextPair]]:
    pairs = get_text_pairs(file_path, limit=limit)
    return split_pairs(pairs, ratio=ratio, shuffle=shuffle, seed=seed)


def load_eval_pairs(file_path: str) -> List[TextPair]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation text file not found: {file_path}")
    pairs: List[TextPair] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            line_lower = line.lower()
            stripped_line = strip_diacritics(line_lower)
            if not stripped_line:
                continue
            if len(stripped_line.split()) != len(line_lower.split()):
                continue
            pairs.append((stripped_line, "[start] " + line_lower + " [end]"))
    return pairs


@dataclass
class Vectorizers:
    source: TextVectorizer
    target: TextVectorizer


def create_vectorizations(
    train_pairs: Sequence[TextPair],
    sequence_length: int = 50,
    vocab_size: int = 15000,
) -> Vectorizers:
    source_vectorizer = TextVectorizer(sequence_length=sequence_length, vocab_size=vocab_size)
    target_vectorizer = TextVectorizer(
        sequence_length=sequence_length + 1,
        vocab_size=vocab_size,
        reserved_tokens=["[start]", "[end]"],
    )

    train_stripped = [pair[0] for pair in train_pairs]
    train_original = [pair[1] for pair in train_pairs]

    source_vectorizer.adapt(train_stripped)
    target_vectorizer.adapt(train_original)

    return Vectorizers(source=source_vectorizer, target=target_vectorizer)


class DiacriticDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: Sequence[TextPair], vectorizers: Vectorizers) -> None:
        self.pairs = list(pairs)
        self.source_vec = vectorizers.source
        self.target_vec = vectorizers.target

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stripped, original = self.pairs[index]
        src_ids = self.source_vec.encode(stripped)
        tgt_ids = self.target_vec.encode(original)
        tgt_input = tgt_ids[:-1].clone()
        tgt_output = tgt_ids[1:].clone()
        return src_ids, tgt_input, tgt_output


def make_dataset(
    pairs: Sequence[TextPair],
    vectorizers: Vectorizers,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = DiacriticDataset(pairs, vectorizers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
