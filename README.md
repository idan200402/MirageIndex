# MirageIndex

Course project for detecting hallucinations in ChatGPT responses. It trains a
family of baseline classifiers on a labeled dataset and exports comparable
metrics for each one.

## Overview

Every model reads `data/general_data.json` (4,507 records of `ID`, `user_query`,
`chatgpt_response`, `hallucination` = `yes`/`no`, and `hallucination_spans`),
performs a deterministic train/test split, trains, and prints test metrics
(accuracy, precision, recall, PR-AUC, ROC-AUC). Passing `--export-metrics True`
writes those metrics to `artifacts/<model_name>/metrics.json`.

The **classical models are pure standard-library** implementations (the TF-IDF
vectorizer, Naive Bayes, logistic regression, random forest, and XGBoost-style
booster are all written from scratch in `source/`). Only the three neural models
depend on PyTorch.

## Requirements

- Python 3.12
- Classical models + `plot_stats.py`: **no third-party packages**.
- Neural models (`encoder_head`, `LLM_train_head`, `LLM_LoRA`): `torch`,
  `transformers` (models are pulled from the Hugging Face Hub on first run).
- `plot_artifacts.py`: `matplotlib`.

There is no dependency manifest; install the extras only if you need the neural
models or the matplotlib plots:

```powershell
pip install torch transformers matplotlib
```

## Project Layout

| Path | Purpose |
| --- | --- |
| `data/general_data.json` | The labeled dataset. |
| `source/shallow_stats.py` | Prints dataset statistics and sample records. |
| `source/model_training/` | One script per model (see below). |
| `source/utils/data.py` | Dataset loading and train/test splitting. |
| `source/utils/text.py` | Tokenizer, TF-IDF vectorizer, and span-chunk labeling. |
| `source/utils/general.py` | Shared CLI args, span aggregation, threshold tuning. |
| `source/utils/training_metrics.py` | Metric computation and JSON export. |
| `source/utils/LLM_train.py` | Shared PyTorch training helpers for the neural models. |
| `source/utils/plot_stats.py` | Single SVG bar chart (PR-AUC + accuracy). |
| `source/utils/plot_artifacts.py` | Four matplotlib views over `artifacts/`. |
| `artifacts/` | Exported per-model `metrics.json` and plots. |

## Models

| Script | Type | Deps |
| --- | --- | --- |
| `majority_voting.py` | Predicts the train-split majority label. | stdlib |
| `naive_bayes.py` | Bag-of-words Naive Bayes. | stdlib |
| `tfidf_logistic_regression.py` | TF-IDF + logistic regression. | stdlib |
| `tfidf_random_forest.py` | TF-IDF + random forest. | stdlib |
| `tfidf_xgboost.py` | TF-IDF + gradient-boosted trees. | stdlib |
| `encoder_head.py` | Linear head on a frozen ModernBERT encoder. | torch |
| `LLM_train_head.py` | Linear head on a frozen Qwen backbone. | torch |
| `LLM_LoRA.py` | LoRA adapters + head on Qwen (adapters hand-rolled). | torch |

## Quickstart

Run from the project root (each script adds the root to `sys.path` itself).

```powershell
# dataset statistics
python source\shallow_stats.py

# train a model and export its metrics
python source\model_training\naive_bayes.py --export-metrics True

# all classical models
python source\model_training\majority_voting.py --export-metrics True
python source\model_training\tfidf_logistic_regression.py --export-metrics True
python source\model_training\tfidf_random_forest.py --export-metrics True
python source\model_training\tfidf_xgboost.py --export-metrics True

# neural models (require torch + transformers)
python source\model_training\encoder_head.py --export-metrics True
python source\model_training\LLM_train_head.py --export-metrics True
python source\model_training\LLM_LoRA.py --export-metrics True
```

Common flags (all models): `--data`, `--seed`, `--test-size`, `--export-metrics`,
`--artifacts-dir`. Neural models add `--base-model`, `--max-length`,
`--batch-size`, `--epochs`, `--learning-rate`, `--weight-decay`, `--dropout`,
`--patience`; `LLM_LoRA` adds `--lora-r`, `--lora-alpha`, `--lora-dropout`.

## Visualizing Results

```powershell
# quick SVG comparison (no dependencies)
python source\utils\plot_stats.py

# richer matplotlib views (needs matplotlib); run as a module
python -m source.utils.plot_artifacts --plot metrics
python -m source.utils.plot_artifacts --plot spans-compare --metric pr_auc
python -m source.utils.plot_artifacts --plot training-curves
python -m source.utils.plot_artifacts --plot span-coverage
```

`plot_stats.py` writes `artifacts/model_stats_comparison.svg`. `plot_artifacts.py`
writes a PNG per view (`--output` / `--show` to override); its four modes cover
all-model metric comparison, regular-vs-spans deltas, neural training curves, and
span-coverage audits.

## Span Mode (`--use-spans`)

By default every model predicts hallucination at the whole
`(user_query, chatgpt_response)` document level. `--use-spans True` instead trains
at the `(user_query, chunk)` level, deriving chunk labels from
`hallucination_spans`. With the flag omitted, output is byte-identical to the
document-level baseline, so span and baseline runs sit side by side for A/B
comparison.

At inference, per-chunk scores are aggregated back into one document score and
thresholded, so reported metrics are still measured against the document-level
`hallucination` label. Span runs fork their artifacts to
`artifacts/<model_name>_spans/`, and their `metrics.json` gains a `span_config`
block and a `span_coverage` audit. `majority_voting` ignores the flag.

Chunk labels come from `build_train_chunk_examples`: spans are matched verbatim
first, then by a whitespace/case-normalized fallback mapped back to exact offsets.
No positive document is dropped — an unresolved chunk is promoted by highest
overlap, and a `yes` document whose spans never resolve (meta-annotations like
"Incomplete answer") weakly labels its longest chunk.

### Span-mode flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--use-spans` | `False` | `True` trains at the chunk level. |
| `--chunk-window` | `40` | Word-window size for chunking. |
| `--chunk-stride` | `20` | Word-window stride between chunks. |
| `--overlap-threshold` | `0.25` | Min fraction of a chunk a span must cover to label it. |
| `--aggregation` | `auto` | `auto`/`max`/`mean_topk`/`noisy_or`. `auto` picks the best on validation by PR-AUC (**neural only**; classical fall back to `max`). |
| `--top-k` | `3` | `K` for `mean_topk`. |
| `--response-threshold` | `0.5` | Fixed decision threshold on the aggregated score. |
| `--target-precision` | unset | **Neural only.** Tune the threshold to a precision target (falling back to the F1 point if recall collapses). |

Neural models (`encoder_head`, `LLM_train_head`, `LLM_LoRA`) have a train/val/test
split, so in span mode they auto-select their aggregation and decision threshold
(F1-maximizing on validation by default) and record it in an `operating_point`
block. Classical models have no validation split and are deliberately left at
`max` / `0.5`.

```powershell
python source\model_training\naive_bayes.py --use-spans True --export-metrics True
python source\model_training\tfidf_logistic_regression.py --use-spans True --aggregation noisy_or --export-metrics True
python source\model_training\encoder_head.py --use-spans True --target-precision 0.6 --dropout 0.1 --weight-decay 0.01 --export-metrics True
python source\model_training\LLM_LoRA.py --use-spans True --aggregation noisy_or --patience 3 --dropout 0.1 --export-metrics True
```

## Span Mode (`--use-spans`)

By default every model predicts hallucination at the whole `(user_query, chatgpt_response)`
document level. The opt-in `--use-spans True` flag instead reframes the unit of prediction to
`(user_query, chunk)` pairs, deriving chunk-level training labels from `hallucination_spans`.
When the flag is omitted (the default), every model produces byte-identical output to the
document-level baseline, so span-aware and baseline runs sit side by side for direct A/B
comparison.

At inference the per-chunk scores are aggregated back into a single document score and
thresholded, so the reported metrics are still measured against the unchanged document-level
`hallucination` label. `majority_voting` ignores the flag by design and remains the fixed
reference baseline.

Artifacts fork automatically: a span-aware run writes to `artifacts/<model_name>_spans/`
(e.g. `artifacts/naive_bayes_spans/`) alongside its baseline, and its `metrics.json` gains a
`span_config` block and a `span_coverage` audit (and, for the discriminative classical models,
the computed `class_weight`). The neural models additionally record an `operating_point` block
describing the aggregation and decision threshold chosen on validation (see below).

### How chunk labels are built (all models)

Training labels come from `hallucination_spans` via `build_train_chunk_examples`:

- **Span matching is fuzzy.** Each annotated span is located verbatim first, then, failing
  that, by a whitespace-collapsed and case-insensitive match that is mapped back to exact
  character offsets — so spans that differ only in spacing/casing still count.
- **No positive document is dropped.** A `yes` document whose spans resolve to no chunk still
  contributes: if no chunk clears `--overlap-threshold`, its highest-overlap chunk is promoted
  positive; if *no* span resolves at all (meta-annotations like "Incomplete answer" that
  describe hallucination *type*, not location), its single longest chunk is weakly labeled
  positive.
- **The `span_coverage` audit surfaces label quality.** Alongside the resolved/fallback counts
  it now reports `yes_records_meta_positive`, `positive_docs_all_negative` (should be `0`),
  `verbatim_spans` / `normalized_spans` / `unresolved_spans`, and `non_verbatim_span_rate`.

### Span-mode flags

These are added to every model (inert unless `--use-spans True` is passed):

- `--use-spans` - `True` trains at the chunk level; `False` (default) is the document-level baseline.
- `--chunk-window` - word-window size for chunking (default `40`).
- `--chunk-stride` - word-window stride between chunks (default `20`).
- `--overlap-threshold` - min fraction of a chunk a span must cover to label it hallucinated (default `0.25`).
- `--aggregation` - how chunk scores combine into a document score: `auto` (default), `max`, `mean_topk`, or `noisy_or`. `auto` picks the best of the three on the validation split by PR-AUC (**neural models only**; classical models fall back to `max`). Pass an explicit method to skip the sweep.
- `--top-k` - `K` for the `mean_topk` aggregation (default `3`).
- `--response-threshold` - fixed decision threshold on the aggregated response score (default `0.5`).
- `--target-precision` - when set to a value in `0-1`, tunes the threshold on the validation split to the lowest value reaching that precision, instead of the fixed `--response-threshold` (**neural models only**; unset by default).

### Neural-only span tuning & regularization

`encoder_head`, `LLM_train_head`, and `LLM_LoRA` have a train/val/test split, so they auto-select
their operating point and can regularize training. These flags are inert for the document-level
baseline (baseline output stays byte-identical):

- `--dropout` - dropout applied before the classification head in span mode (default `0.0`, i.e. off).
- `--patience` - early-stopping patience in epochs on `val_document_bce` (default `3`; `0` disables it).

The chosen aggregation and threshold (and the per-method validation PR-AUC when `auto` is used)
are written to the `operating_point` block of the span run's `metrics.json`. Classical models
are deliberately left at `max` / `0.5` — they have no validation split, and carving one from
their training data would shift them too far from their document-level baselines.

```powershell
# classical: chunk labels get the fuzzy-match + weak-label improvements automatically
python source\model_training\naive_bayes.py --use-spans True --export-metrics True
python source\model_training\tfidf_logistic_regression.py --use-spans True --aggregation noisy_or --export-metrics True
python source\model_training\tfidf_random_forest.py --use-spans True --chunk-window 40 --chunk-stride 20 --export-metrics True

# neural: auto aggregation is on by default; add threshold tuning + regularization
python source\model_training\encoder_head.py --use-spans True --export-metrics True
python source\model_training\encoder_head.py --use-spans True --target-precision 0.6 --dropout 0.1 --weight-decay 0.01 --export-metrics True
python source\model_training\LLM_LoRA.py --use-spans True --aggregation noisy_or --patience 3 --dropout 0.1 --export-metrics True
```
