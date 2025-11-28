from collections import Counter
from typing import Dict, List, Mapping, Optional

from .baseline_utils import (
    TextPair,
    build_token_statistics,
    compute_prediction_metrics,
    strip_special_tokens,
)


def _most_common_variant(token: str, counts: Dict[str, "Counter[str]"]) -> Optional[str]:
    counter = counts.get(token.lower())
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def run_most_frequent_baseline(
    train_pairs: List[TextPair],
    eval_splits: Mapping[str, List[TextPair]],
) -> Dict[str, object]:
    _, counts = build_token_statistics(train_pairs)
    metrics: Dict[str, Dict[str, float]] = {}

    for split_name, pairs in eval_splits.items():
        predictions: List[str] = []
        references: List[str] = []
        for stripped, labeled in pairs:
            restored_tokens: List[str] = []
            for tok in stripped.split():
                replacement = _most_common_variant(tok, counts)
                restored_tokens.append(replacement if replacement else tok)
            predictions.append(" ".join(restored_tokens))
            references.append(strip_special_tokens(labeled))
        metrics[split_name] = compute_prediction_metrics(predictions, references)

    return {"metrics": metrics}


__all__ = ["run_most_frequent_baseline"]
