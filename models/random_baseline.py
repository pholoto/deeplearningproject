import random
from typing import Dict, List, Mapping

from .baseline_utils import (
    TextPair,
    build_token_statistics,
    compute_prediction_metrics,
    strip_special_tokens,
)

RNG = random.Random(42)


def _random_variant_sentence(sentence: str, variants: Dict[str, List[str]]) -> str:
    tokens: List[str] = []
    for token in sentence.split():
        pool = variants.get(token.lower())
        if pool:
            tokens.append(RNG.choice(pool))
        else:
            tokens.append(token)
    return " ".join(tokens)


def run_random_baseline(
    train_pairs: List[TextPair],
    eval_splits: Mapping[str, List[TextPair]],
) -> Dict[str, object]:
    variants, _ = build_token_statistics(train_pairs)
    metrics: Dict[str, Dict[str, float]] = {}

    for split_name, pairs in eval_splits.items():
        inputs = [pair[0] for pair in pairs]
        references = [strip_special_tokens(pair[1]) for pair in pairs]
        predictions = [_random_variant_sentence(sentence, variants) for sentence in inputs]
        metrics[split_name] = compute_prediction_metrics(predictions, references)

    return {"metrics": metrics}


__all__ = ["run_random_baseline"]
