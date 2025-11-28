# Vietnamese Diacritic Restoration

This repository contains baseline and transformer models for restoring Vietnamese diacritics from stripped text. The pipeline loads paired training data, trains multiple sequence-to-sequence models, and can preview or evaluate checkpointed models on arbitrary text files.

## Environment & Setup
- Python 3.10+ (the code relies on standard typing features and PyTorch).
- [PyTorch](https://pytorch.org) with CPU or CUDA support depending on your hardware.
- Optional: create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
```

> If you already have PyTorch installed system-wide you can skip the virtual environment steps.

## Data Layout
- `data/wiki_pairs.txt` is the default training corpus. Each line must contain a stripped sentence and its diacritic-rich counterpart separated by a tab, with identical token counts.
- `data/truyen_kieu.txt` and `data/magazine.txt` are sample plain-text evaluation files that already include diacritics. When evaluated, the script automatically strips diacritics internally for comparison.
- Splits (train/val/test) are cached via `checkpoints/data_split_meta.json`, so subsequent `--mode eval` runs reuse the original seed unless you override it.

## Models
- **Random baseline** (`models/random_baseline.py`): swaps each token for a random variant observed in training statistics.
- **Most-frequent baseline** (`models/most_frequent_baseline.py`): deterministically replaces tokens with their most common labeled form.
- **Seq2Seq LSTM** (`models/lstm_baseline.py`): bidirectional encoder + decoder with teacher forcing, early stopping, and vectorizers saved alongside checkpoints.
- **Transformer (base)** (`models/transformer.py`): encoder-decoder transformer with positional embeddings, multi-head attention, and Adam optimization.
- **Transformer + SAM**: identical architecture trained with Sharpness-Aware Minimization (`--run-sam-variant/--sam-only`) for improved generalization.

## Running `main.py`
`main.py` is the entry point for training and evaluation. The most relevant arguments:
- `--models random most_frequent lstm transformer`: choose any subset or use `all` (default).
- `--mode train|eval`: `train` fits the requested models and saves checkpoints under `checkpoints/`; `eval` skips training and only loads existing checkpoints.
- `--dataset PATH`: defaults to `data/wiki_pairs.txt`.
- `--eval-file [LABEL:]PATH`: add one or more external evaluation/preview files (see below).
- `--checkpoint-dir checkpoints`: directory that stores checkpoints, vectorizers, previews, and split metadata.

### Train example
```powershell
python main.py --mode train --models transformer lstm --dataset data/wiki_pairs.txt --checkpoint-dir checkpoints/train_run --epochs 50 --sequence-length 60 --vocab-size 35000
```
This command:
- Splits the dataset (seed = `--data-seed`, default 42) and caches it.
- Trains the LSTM and both transformer variants (unless `--sam-only` or `--skip-sam` are set).
- Stores weights under `checkpoints/train_run/<model_name>/` plus the necessary vectorizers.

### Evaluate existing checkpoints
Once checkpoints exist, run:
```powershell
python main.py --mode eval --models transformer lstm --checkpoint-dir checkpoints/train_run --eval-file truyen_kieu:data/truyen_kieu.txt --eval-file magazine:data/magazine.txt
```
This loads the saved vectorizers/models, evaluates them on the cached test split and the provided evaluation files, then prints sentence/token accuracy.

## Preview Files (`preview.py`)
`preview.py` is invoked automatically inside `main.py` whenever:
1. At least one checkpointed sequence model (LSTM or Transformer) is available, and
2. You pass one or more `--eval-file` entries (or rely on the defaults).

For each evaluation label, the script writes files such as `checkpoints/train_run/magazine_predictions_first8.txt`. Every block shows:
- Stripped input
- Expected diacritic-rich text
- Greedy predictions from each requested model, plus random/most-frequent baselines

To refresh previews, simply rerun `main.py` in `--mode eval` with the desired `--eval-file` arguments; the preview files will be regenerated.

## Evaluating a New Text File
To test checkpoints on your own material:
1. Create a UTF-8 text file under `data/`, e.g. `data/my_story.txt`. Each line should be natural Vietnamese text **with** diacritics; the script handles stripping internally.
2. Run `main.py` in eval mode and pass the new file via `--eval-file`. You may optionally prepend a label before `:` to control how results are printed and named.

```powershell
python main.py --mode eval --models transformer --checkpoint-dir checkpoints/train_run --eval-file my_story:data/my_story.txt
```
3. Inspect the console accuracy summary and the preview file `checkpoints/train_run/my_story_predictions_first8.txt` to manually review predictions.

> Tip: you can chain multiple `--eval-file` flags. When only evaluating new text, you can skip retraining by omitting `--mode train` entirely.

## Troubleshooting & Tips
- Make sure the checkpoint directory you pass to `--checkpoint-dir` matches the folder that already contains `transformer.pt`, `transformer_sam.pt`, or `lstm.pt`; otherwise `--mode eval` cannot load weights.
- If you change `--sequence-length`, `--vocab-size`, or the dataset path, retrain so that the saved vectorizers and model shapes stay consistent.
- Preview generation requires vectorizer `.pkl` files; do not delete them if you plan to evaluate later.
