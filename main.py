import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import random

from models.baseline_utils import build_token_statistics
from models.lstm_baseline import Seq2SeqLSTM, run_lstm_baseline
from models.most_frequent_baseline import run_most_frequent_baseline
from models.random_baseline import run_random_baseline
from models.transformer import TrainConfig, TransformerModel, train_model
from preview import write_preview_predictions
from utils.data_loader import Vectorizers, load_data, load_eval_pairs
from utils.vectorizer import load_vectorizer

VALID_MODELS = ["random", "most_frequent", "lstm", "transformer"]
MODEL_DISPLAY = {
    "random": "Random",
    "most_frequent": "Most-frequent",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "transformer_sam": "Transformer (SAM)",
}

CHECKPOINT_CANDIDATES = {
    "lstm": ("lstm/lstm.pt", "lstm.pt"),
    "transformer": ("transformer_base/transformer.pt", "transformer.pt"),
    "transformer_sam": (
        "transformer_sam/transformer_sam.pt",
        "transformer_sam.pt",
        "transformer_sam/transformer.pt"
    ),
}

PREDICTION_MODEL_KEYS = ["random", "most_frequent", "lstm", "transformer", "transformer_sam"]

SPLIT_META_FILE = "data_split_meta.json"

DEFAULT_EVAL_FILES = [
    "data/truyen_kieu.txt",
    "data/magazine.txt",
]

TextPair = Tuple[str, str]


def _find_checkpoint_path(model_key: str, checkpoint_dir: str) -> Optional[str]:
    candidates = CHECKPOINT_CANDIDATES.get(model_key, ())
    for rel_path in candidates:
        candidate = os.path.join(checkpoint_dir, rel_path)
        if os.path.exists(candidate):
            return candidate
    return None


def _split_meta_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, SPLIT_META_FILE)


def _load_split_seed(checkpoint_dir: str, dataset_path: str) -> Optional[int]:
    meta_path = _split_meta_path(checkpoint_dir)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    stored_path = payload.get("dataset_path")
    if stored_path and os.path.abspath(stored_path) != os.path.abspath(dataset_path):
        return None
    seed = payload.get("seed")
    return int(seed) if isinstance(seed, int) else None


def _save_split_seed(checkpoint_dir: str, dataset_path: str, seed: Optional[int]) -> None:
    meta_path = _split_meta_path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "dataset_path": os.path.abspath(dataset_path),
        "seed": seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _derive_label_from_path(path: str) -> str:
    base = os.path.basename(path.strip())
    if not base:
        return "dataset"
    name, _ = os.path.splitext(base)
    return name or "dataset"


def _parse_eval_specs(raw_specs: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for raw in raw_specs:
        if not raw:
            continue
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            label, path = entry.split(":", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = entry
            label = _derive_label_from_path(path)
        if not label or not path:
            print(f"Skipping extra eval spec '{raw}' (empty label or path)")
            continue
        if any(existing_label == label for existing_label, _ in parsed):
            print(f"Skipping duplicate extra eval label '{label}'")
            continue
        parsed.append((label, path))
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Vietnamese diacritic restoration baselines and transformer model")
    parser.add_argument("--dataset", default="data/wiki_pairs.txt", type=str)
    parser.add_argument("--sequence-length", default=50, type=int)
    parser.add_argument("--vocab-size", default=35000, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--embed-dim", default=256, type=int)
    parser.add_argument("--dense-dim", default=2048, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--checkpoint-dir", default="checkpoints", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use (cpu or cuda)")
    parser.add_argument(
        "--eval-file",
        action="append",
        default=[],
        metavar="[LABEL:]PATH",
        help=(
            "Text file(s) to evaluate (accepts label:path or just path). "
            "If omitted, defaults to data/truyen_kieu.txt and data/magazine.txt. "
            "Example: --eval-file magazine:data/magazine.txt or --eval-file data/custom.txt"
        ),
    )
    parser.add_argument("--data-seed", default=42, type=int, help="Seed used for dataset splitting (set <0 to disable caching)")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Whether to train models or only load checkpoints and evaluate",
    )
    parser.add_argument("--sam-only", action="store_true", help="Train only the SAM-enabled transformer variant")
    parser.add_argument(
        "--skip-sam",
        dest="run_sam_variant",
        action="store_false",
        help="Skip the SAM-enabled transformer run",
    )
    parser.add_argument(
        "--use-sam",
        dest="run_sam_variant",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sam-rho", default=0.05, type=float, help="SAM rho (neighborhood size)")
    parser.add_argument("--sam-adaptive", action="store_true", help="Use adaptive SAM (ASAM)")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience for transformer training")
    parser.add_argument("--lstm-patience", default=3, type=int, help="Early stopping patience for LSTM baseline")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=VALID_MODELS + ["all"],
        help="Which models to run (random, most_frequent, lstm, transformer, or all)",
    )
    parser.set_defaults(run_sam_variant=True)
    return parser


def _models_to_run(selected: Sequence[str]) -> List[str]:
    if "all" in selected:
        return VALID_MODELS.copy()
    ordered: List[str] = []
    seen = set()
    for name in selected:
        if name not in VALID_MODELS or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    selected_models = _models_to_run(args.models)
    if not selected_models:
        print("No valid models selected. Exiting.")
        return

    configured_seed = args.data_seed if args.data_seed is not None and args.data_seed >= 0 else None
    cached_seed = _load_split_seed(args.checkpoint_dir, args.dataset)
    seed_to_use: Optional[int]
    if args.mode == "train":
        seed_to_use = configured_seed
    else:
        seed_to_use = cached_seed if cached_seed is not None else configured_seed
        if cached_seed is not None and cached_seed != configured_seed:
            print(
                f"Using cached dataset split seed {cached_seed} (stored in {SPLIT_META_FILE})"
            )
        elif cached_seed is None and configured_seed is None:
            print(
                "Warning: No dataset split seed provided or cached; evaluation results may not match training splits."
            )

    data_splits = load_data(args.dataset, shuffle=True, seed=seed_to_use)

    if args.mode == "train":
        _save_split_seed(args.checkpoint_dir, args.dataset, seed_to_use)
    default_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(default_device)

    train_pairs, val_pairs, test_pairs = data_splits

    eval_text_pairs: Dict[str, List[TextPair]] = {}
    preview_labels: List[str] = []
    eval_inputs: Sequence[str] = args.eval_file if args.eval_file else DEFAULT_EVAL_FILES
    parsed_eval_specs = _parse_eval_specs(eval_inputs)
    for label, path in parsed_eval_specs:
        try:
            pairs = load_eval_pairs(path)
        except FileNotFoundError as exc:
            print(f"Skipping eval file '{path}': {exc}")
            continue
        if not pairs:
            print(f"Skipping eval file '{path}': no usable lines")
            continue
        eval_text_pairs[label] = pairs
        preview_labels.append(label)

    results: Dict[str, Dict[str, Any]] = {}
    summary_order: List[str] = []

    eval_splits_full: Dict[str, List[TextPair]] = {"test": test_pairs}
    eval_splits_full.update(eval_text_pairs)

    if "random" in selected_models:
        random_result = run_random_baseline(train_pairs, eval_splits_full)
        results["random"] = random_result
        summary_order.append("random")

    if "most_frequent" in selected_models:
        freq_result = run_most_frequent_baseline(train_pairs, eval_splits_full)
        results["most_frequent"] = freq_result
        summary_order.append("most_frequent")

    extra_eval = eval_text_pairs if eval_text_pairs else {}

    if "lstm" in selected_models:
        if args.mode == "train":
            lstm_result = run_lstm_baseline(
                train_pairs,
                val_pairs,
                test_pairs,
                sequence_length=args.sequence_length,
                vocab_size=args.vocab_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.learning_rate,
                device=torch_device,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "lstm"),
                patience=args.lstm_patience,
                eval_splits=extra_eval,
            )
            results["lstm"] = lstm_result
        else:
            # mark placeholder so we will attempt to load checkpoint later
            results["lstm"] = {}
        summary_order.append("lstm")

    run_transformer = "transformer" in selected_models
    run_standard_transformer = run_transformer and not args.sam_only
    run_sam_transformer = run_transformer and (args.run_sam_variant or args.sam_only)

    if run_standard_transformer:
        if args.mode == "train":
            base_config = TrainConfig(
                dataset_path=args.dataset,
                sequence_length=args.sequence_length,
                vocab_size=args.vocab_size,
                data_seed=seed_to_use,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                embed_dim=args.embed_dim,
                dense_dim=args.dense_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "transformer_base"),
                device=default_device,
                use_sam=False,
                sam_rho=args.sam_rho,
                sam_adaptive=args.sam_adaptive,
                patience=args.patience,
            )
            transformer_result = train_model(
                base_config,
                data_splits=data_splits,
                eval_splits=extra_eval,
            )
            results["transformer"] = transformer_result
        else:
            results["transformer"] = {}
        summary_order.append("transformer")

    if run_sam_transformer:
        if args.mode == "train":
            sam_config = TrainConfig(
                dataset_path=args.dataset,
                sequence_length=args.sequence_length,
                vocab_size=args.vocab_size,
                data_seed=seed_to_use,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                embed_dim=args.embed_dim,
                dense_dim=args.dense_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                checkpoint_dir=os.path.join(args.checkpoint_dir, "transformer_sam"),
                device=default_device,
                use_sam=True,
                sam_rho=args.sam_rho,
                sam_adaptive=args.sam_adaptive,
                patience=args.patience,
            )
            transformer_sam_result = train_model(
                sam_config,
                data_splits=data_splits,
                eval_splits=extra_eval,
            )
            results["transformer_sam"] = transformer_sam_result
        else:
            results["transformer_sam"] = {}
        summary_order.append("transformer_sam")

    if not summary_order:
        print("No model runs were executed.")
        return

    # dataset order used for final reporting
    dataset_order: List[str] = ["test"]
    dataset_order.extend(eval_text_pairs.keys())

    ckpt_dir = args.checkpoint_dir
    checkpoint_paths: Dict[str, str] = {}
    for key in summary_order:
        existing = results.get(key, {}).get("checkpoint")
        candidate = existing or _find_checkpoint_path(key, ckpt_dir)
        if candidate:
            checkpoint_paths[key] = candidate

    summary_accs: Dict[str, Any] = {}

    if checkpoint_paths:
        print("\nCheckpoints:")
        for key in summary_order:
            path = checkpoint_paths.get(key)
            if path:
                label = MODEL_DISPLAY.get(key, key.title())
                print(f"  {label:<18}: {path}")

        preview_count = 8
        os.makedirs(ckpt_dir, exist_ok=True)

        def try_load_vectorizer_names(base_dir: str, names):
            for name in names:
                p = os.path.join(base_dir, name)
                if os.path.exists(p):
                    try:
                        return load_vectorizer(p)
                    except Exception:
                        continue
            return None

        try:
            variants, counts = build_token_statistics(train_pairs)
        except Exception:
            variants, counts = {}, {}

        rng = random.Random(42)

        def load_vectorizer_from_dirs(directories: List[str], filenames: List[str]):
            seen = set()
            for directory in directories:
                if not directory or directory in seen:
                    continue
                seen.add(directory)
                vec = try_load_vectorizer_names(directory, filenames)
                if vec is not None:
                    return vec
            return None

        lstm_vec_dirs = [
            os.path.join(ckpt_dir, "lstm"),
            ckpt_dir,
        ]
        lstm_src_vec = load_vectorizer_from_dirs(lstm_vec_dirs, ["lstm_source_vectorizer.pkl"])
        lstm_tgt_vec = load_vectorizer_from_dirs(lstm_vec_dirs, ["lstm_target_vectorizer.pkl"])
        lstm_vectorizers = Vectorizers(source=lstm_src_vec, target=lstm_tgt_vec) if lstm_src_vec and lstm_tgt_vec else None

        transformer_vec_dirs: List[str] = []
        for variant in ("transformer", "transformer_sam"):
            ckpt_path = checkpoint_paths.get(variant)
            if ckpt_path:
                transformer_vec_dirs.append(os.path.dirname(ckpt_path))
        transformer_vec_dirs.extend([
            os.path.join(ckpt_dir, "transformer_base"),
            os.path.join(ckpt_dir, "transformer_sam"),
            ckpt_dir,
        ])
        trans_src_vec = load_vectorizer_from_dirs(transformer_vec_dirs, ["source_vectorizer.pkl"])
        trans_tgt_vec = load_vectorizer_from_dirs(transformer_vec_dirs, ["target_vectorizer.pkl"])
        transformer_vectorizers = Vectorizers(source=trans_src_vec, target=trans_tgt_vec) if trans_src_vec and trans_tgt_vec else None

        lstm_model = None
        lstm_load_error: Optional[str] = None
        if "lstm" in checkpoint_paths:
            if lstm_vectorizers:
                try:
                    vocab_size = max(
                        lstm_vectorizers.source.vocab_size(),
                        lstm_vectorizers.target.vocab_size(),
                    )
                    lstm_model = Seq2SeqLSTM(
                        vocab_size=vocab_size,
                        emb_dim=256,
                        hidden_dim=512,
                        pad_id=lstm_vectorizers.target.pad_id(),
                    )
                    lstm_state = torch.load(checkpoint_paths["lstm"], map_location="cpu")
                    lstm_model.load_state_dict(lstm_state)
                    lstm_model.to(torch_device)
                    lstm_model.eval()
                except Exception as exc:
                    lstm_load_error = str(exc)
                    lstm_model = None
            else:
                lstm_load_error = "vectorizer unavailable"

        transformer_models: Dict[str, TransformerModel] = {}
        transformer_load_errors: Dict[str, str] = {}
        for variant in ("transformer", "transformer_sam"):
            ckpt_path = checkpoint_paths.get(variant)
            if not ckpt_path:
                continue
            if not transformer_vectorizers:
                transformer_load_errors[variant] = "vectorizer unavailable"
                continue
            try:
                vocab_size = max(
                    transformer_vectorizers.source.vocab_size(),
                    transformer_vectorizers.target.vocab_size(),
                )
                model = TransformerModel(
                    vocab_size=vocab_size,
                    src_seq_len=transformer_vectorizers.source.sequence_length,
                    tgt_seq_len=transformer_vectorizers.target.sequence_length,
                    embed_dim=args.embed_dim,
                    dense_dim=args.dense_dim,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                )
                model_state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(model_state)
                model.to(torch_device)
                model.eval()
                transformer_models[variant] = model
            except Exception as exc:
                transformer_load_errors[variant] = str(exc)

        seq_model_info: Dict[str, Tuple[Union[Seq2SeqLSTM, TransformerModel], Vectorizers]] = {}
        if lstm_model and lstm_vectorizers:
            seq_model_info["lstm"] = (lstm_model, lstm_vectorizers)
        if transformer_vectorizers:
            for variant, model in transformer_models.items():
                seq_model_info[variant] = (model, transformer_vectorizers)

        preview_model_keys = [key for key in summary_order if key in PREDICTION_MODEL_KEYS]
        preview_sources = {label: eval_text_pairs[label] for label in preview_labels if label in eval_text_pairs}
        write_preview_predictions(
            ckpt_dir=ckpt_dir,
            preview_labels=preview_labels,
            preview_sources=preview_sources,
            model_display=MODEL_DISPLAY,
            prediction_model_keys=preview_model_keys,
            seq_model_info=seq_model_info,
            transformer_load_errors=transformer_load_errors,
            lstm_load_error=lstm_load_error,
            variants=variants,
            counts=counts,
            rng=rng,
            preview_count=preview_count,
            device=torch_device,
        )

        from utils.data_loader import make_dataset

        def _token_accuracy_from_dataloader(model, dataloader, vectorizers, device=torch.device("cpu")):
            model.to(device)
            model.eval()
            total_correct = 0
            total_tokens = 0
            src_pad = vectorizers.source.pad_id()
            tgt_pad = vectorizers.target.pad_id()
            with torch.no_grad():
                for src, tgt_in, tgt_out in dataloader:
                    src = src.to(device)
                    teacher_inputs = tgt_in.to(device)
                    tgt_out = tgt_out.to(device)
                    if isinstance(model, TransformerModel):
                        src_mask = src.eq(src_pad)
                        tgt_mask = teacher_inputs.eq(tgt_pad)
                        logits = model(
                            src,
                            teacher_inputs,
                            src_padding_mask=src_mask,
                            tgt_padding_mask=tgt_mask,
                        )
                    else:
                        logits = model(src, teacher_inputs)
                    preds = logits.argmax(dim=-1)
                    mask = tgt_out.ne(tgt_pad)
                    correct = preds.eq(tgt_out) & mask
                    total_correct += int(correct.sum().item())
                    total_tokens += int(mask.sum().item())
            return (total_correct / total_tokens) if total_tokens else 0.0

        eval_datasets_for_metrics: List[Tuple[str, List[TextPair]]] = [("test", test_pairs)]
        eval_datasets_for_metrics.extend(eval_text_pairs.items())

        for mname, (model_obj, vecs) in seq_model_info.items():
            try:
                metrics_entry: Dict[str, float] = {}
                for split_label, split_pairs in eval_datasets_for_metrics:
                    if not split_pairs:
                        continue
                    loader = make_dataset(
                        split_pairs,
                        vecs,
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                    metrics_entry[split_label] = _token_accuracy_from_dataloader(
                        model_obj,
                        loader,
                        vecs,
                        torch_device,
                    )
                summary_accs[mname] = metrics_entry
            except Exception as exc:
                summary_accs[mname] = {"error": str(exc)}

        if summary_accs:
            print("\nToken-accuracy computed via teacher-forced evaluation (matches training):")
            for k, v in summary_accs.items():
                print(f"  {k}: {v}")

    print("\nSentence / token accuracy summary:")
    for split_name in dataset_order:
        print(f"\n{split_name.title()} dataset:")
        for key in summary_order:
            label = MODEL_DISPLAY.get(key, key.title())
            payload = results.get(key, {})

            accs_for_model = summary_accs.get(key) if isinstance(summary_accs, dict) else None
            if accs_for_model and isinstance(accs_for_model, dict) and not accs_for_model.get("error"):
                tok = accs_for_model.get(split_name)
                if tok is None:
                    print(f"  {label:<18}: token=(unavailable)")
                else:
                    print(f"  {label:<18}: token={tok:.4f}")
                continue

            metrics = payload.get("metrics", {}).get(split_name) if payload.get("metrics") else None
            if metrics:
                sent_acc = metrics.get("sentence_accuracy")
                tok_acc = metrics.get("token_accuracy")
                sent_display = f"{sent_acc:.4f}" if isinstance(sent_acc, (int, float)) else "n/a"
                tok_display = f"{tok_acc:.4f}" if isinstance(tok_acc, (int, float)) else "n/a"
                print(f"  {label:<18}: sent={sent_display}  token={tok_display}")
            else:
                test_acc = payload.get("test_acc")
                if test_acc is not None:
                    print(f"  {label:<18}: token={float(test_acc):.4f}")
                else:
                    print(f"  {label:<18}: (no data)")

if __name__ == "__main__":
    main()
