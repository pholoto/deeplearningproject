from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

TextPair = Tuple[str, str]
START_TOKEN = "[start]"
END_TOKEN = "[end]"


def strip_special_tokens(text: str) -> str:
    text = text.strip()
    if text.startswith(f"{START_TOKEN} "):
        text = text[len(START_TOKEN) + 1 :]
    if text.endswith(f" {END_TOKEN}"):
        text = text[: -(len(END_TOKEN) + 1)]
    return text


def compute_sentence_accuracy(predictions: List[str], references: List[str]) -> float:
    if not references:
        return 0.0
    matches = sum(int(p == r) for p, r in zip(predictions, references))
    return matches / len(references)


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    total_ref_tokens = 0
    correct_tokens = 0
    for pred, ref in zip(predictions, references):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        total_ref_tokens += len(ref_tokens)
        for idx, ref_tok in enumerate(ref_tokens):
            if idx < len(pred_tokens) and pred_tokens[idx] == ref_tok:
                correct_tokens += 1
    return (correct_tokens / total_ref_tokens) if total_ref_tokens else 0.0


def compute_prediction_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    return {
        "sentence_accuracy": compute_sentence_accuracy(predictions, references),
        "token_accuracy": compute_token_accuracy(predictions, references),
    }


def build_token_statistics(train_pairs: Iterable[TextPair]) -> Tuple[Dict[str, List[str]], Dict[str, Counter[str]]]:
    variants: Dict[str, List[str]] = defaultdict(list)
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for stripped, label in train_pairs:
        label_body = strip_special_tokens(label)
        stripped_tokens = stripped.split()
        label_tokens = label_body.split()
        if len(stripped_tokens) != len(label_tokens):
            continue
        for src_tok, lbl_tok in zip(stripped_tokens, label_tokens):
            key = src_tok.lower()
            variants[key].append(lbl_tok)
            counts[key][lbl_tok] += 1
    return variants, counts
