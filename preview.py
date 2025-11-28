import os
import random
import unicodedata
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from models.baseline_utils import strip_special_tokens
from models.lstm_baseline import Seq2SeqLSTM, lstm_greedy_decode
from models.transformer import TransformerModel, transformer_greedy_decode
from utils.data_loader import Vectorizers, strip_diacritics

TextPair = Tuple[str, str]
ModelInfo = Tuple[Union[Seq2SeqLSTM, TransformerModel], Vectorizers]


def _is_symbol_char(ch: str) -> bool:
    """Return True if the character should be stripped before prediction."""
    if ch.isspace():
        return False
    category = unicodedata.category(ch)
    return category.startswith("P") or category.startswith("S")


def _strip_symbols_for_preview(text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Remove symbol characters and remember their positions for later restore."""
    if not text:
        return text, []
    clean_chars: List[str] = []
    placements: List[Tuple[int, str]] = []
    clean_index = 0
    for ch in text:
        if _is_symbol_char(ch):
            placements.append((clean_index, ch))
            continue
        clean_chars.append(ch)
        clean_index += 1
    cleaned = "".join(clean_chars)
    if cleaned.strip():
        return cleaned, placements
    # fallback to original text when everything was stripped
    return text, []


def _restore_symbols(text: str, placements: Sequence[Tuple[int, str]]) -> str:
    if not placements:
        return text
    chars = list(text)
    inserted = 0
    for pos, symbol in placements:
        insert_at = pos + inserted
        if insert_at < 0:
            insert_at = 0
        if insert_at > len(chars):
            insert_at = len(chars)
        chars.insert(insert_at, symbol)
        inserted += 1
    return "".join(chars)


def _best_variant_for_token(
    base_token: str,
    variants: Optional[Dict[str, List[str]]],
    counts: Optional[Dict[str, Counter[str]]],
) -> str:
    key = base_token.lower()
    if counts:
        token_counts = counts.get(key)
        if token_counts:
            candidate, _ = max(token_counts.items(), key=lambda kv: kv[1])
            return candidate
    if variants:
        token_variants = variants.get(key)
        if token_variants:
            return token_variants[0]
    return base_token


def _constrain_prediction_to_source(
    prediction: str,
    stripped_sentence: str,
    variants: Optional[Dict[str, List[str]]],
    counts: Optional[Dict[str, Counter[str]]],
) -> str:
    if not stripped_sentence:
        return prediction
    source_tokens = stripped_sentence.split()
    if not source_tokens:
        return prediction
    pred_tokens = prediction.split()
    adjusted: List[str] = []
    for idx, source_token in enumerate(source_tokens):
        candidate = pred_tokens[idx] if idx < len(pred_tokens) else ""
        if candidate:
            normalized = strip_diacritics(candidate.lower())
            if normalized == source_token.lower():
                adjusted.append(candidate)
                continue
        replacement = _best_variant_for_token(source_token, variants, counts)
        adjusted.append(replacement)
    return " ".join(adjusted)


def _safe_label_for_filename(label: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return sanitized or "dataset"


def _random_prediction(
    stripped: str,
    variants: Optional[Dict[str, List[str]]],
    rng: random.Random,
) -> str:
    if not variants:
        return "(no train data)"
    tokens: List[str] = []
    for tok in stripped.split():
        pool = variants.get(tok.lower())
        tokens.append(rng.choice(pool) if pool else tok)
    return " ".join(tokens)


def _most_frequent_prediction(
    stripped: str,
    counts: Optional[Dict[str, Counter[str]]],
) -> str:
    if not counts:
        return "(no train data)"
    tokens: List[str] = []
    for tok in stripped.split():
        counter = counts.get(tok.lower())
        if counter:
            tokens.append(max(counter.items(), key=lambda x: x[1])[0])
        else:
            tokens.append(tok)
    return " ".join(tokens)


def write_preview_predictions(
    *,
    ckpt_dir: str,
    preview_labels: Sequence[str],
    preview_sources: Mapping[str, Sequence[TextPair]],
    model_display: Mapping[str, str],
    prediction_model_keys: Sequence[str],
    seq_model_info: Dict[str, ModelInfo],
    transformer_load_errors: Mapping[str, str],
    lstm_load_error: Optional[str],
    variants: Optional[Dict[str, List[str]]],
    counts: Optional[Dict[str, Counter[str]]],
    rng: Optional[random.Random],
    preview_count: int,
    device: torch.device,
) -> None:
    if not preview_labels or not prediction_model_keys:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    rng = rng or random.Random(42)

    for label in preview_labels:
        dataset_pairs = preview_sources.get(label)
        if not dataset_pairs:
            continue
        head = list(dataset_pairs[:preview_count])
        if not head:
            continue

        preds_by_model = {key: [] for key in prediction_model_keys}
        for stripped, orig in head:
            clean_input, placements = _strip_symbols_for_preview(stripped)
            effective_input = clean_input if clean_input else stripped
            if "random" in preds_by_model:
                random_pred = _random_prediction(effective_input, variants, rng)
                preds_by_model["random"].append(_restore_symbols(random_pred, placements))

            if "most_frequent" in preds_by_model:
                freq_pred = _most_frequent_prediction(effective_input, counts)
                preds_by_model["most_frequent"].append(_restore_symbols(freq_pred, placements))

            if "lstm" in preds_by_model:
                info = seq_model_info.get("lstm")
                if info:
                    model_obj, vecs = info
                    lstm_model_for_pred = model_obj
                    lstm_pred = lstm_greedy_decode(lstm_model_for_pred, effective_input, vecs, device)
                    lstm_pred = _constrain_prediction_to_source(lstm_pred, effective_input, variants, counts)
                    lstm_pred = _restore_symbols(lstm_pred, placements)
                    preds_by_model["lstm"].append(lstm_pred)
                else:
                    msg = f"(lstm load failed: {lstm_load_error})" if lstm_load_error else "(lstm unavailable)"
                    preds_by_model["lstm"].append(msg)

            for variant in ("transformer", "transformer_sam"):
                if variant not in preds_by_model:
                    continue
                info = seq_model_info.get(variant)
                if info:
                    model_obj, vecs = info
                    transformer_pred = transformer_greedy_decode(
                        model_obj,
                        effective_input,
                        vecs,
                        device,
                    )
                    transformer_pred = _constrain_prediction_to_source(
                        transformer_pred,
                        effective_input,
                        variants,
                        counts,
                    )
                    transformer_pred = _restore_symbols(transformer_pred, placements)
                    preds_by_model[variant].append(transformer_pred)
                else:
                    err = transformer_load_errors.get(variant)
                    message = f"({variant} load failed: {err})" if err else f"({variant} unavailable)"
                    preds_by_model[variant].append(message)

        file_label = _safe_label_for_filename(label)
        out_file = os.path.join(ckpt_dir, f"{file_label}_predictions_first{preview_count}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{label.title()} predictions — first {preview_count} lines\n")
            trailing = " / ".join(model_display.get(key, key.title()) for key in prediction_model_keys)
            if trailing:
                f.write("Each block: Input (stripped) / Expected (original) / " + trailing + "\n\n")
            else:
                f.write("Each block: Input (stripped) / Expected (original)\n\n")
            for idx, (stripped, orig) in enumerate(head, 1):
                f.write(f"Line {idx}:\n")
                f.write(f"  Input   : {stripped}\n")
                expected_text = strip_special_tokens(orig)
                f.write(f"  Expected: {expected_text}\n")
                for key in prediction_model_keys:
                    model_label = model_display.get(key, key.title())
                    pred_list = preds_by_model.get(key, [])
                    pred_value = pred_list[idx - 1] if idx - 1 < len(pred_list) else "(unavailable)"
                    f.write(f"  {model_label:<18}: {pred_value}\n")
                f.write("\n")

        print(f"Wrote predictions for '{label}' to {out_file}")
