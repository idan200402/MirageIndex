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
booster are all written from scratch in `source/`). Only the four neural models
depend on PyTorch.

## Requirements

- Python 3.12
- Classical models + `plot_stats.py`: **no third-party packages**.
- Neural models (`encoder_head`, `encoder_LoRA`, `LLM_train_head`, `LLM_LoRA`): `torch`,
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
| `encoder_LoRA.py` | LoRA adapters + head on ModernBERT attention projections. | torch |
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
python source\model_training\encoder_LoRA.py --export-metrics True
python source\model_training\LLM_train_head.py --export-metrics True
python source\model_training\LLM_LoRA.py --export-metrics True
```

Common flags (all models): `--data`, `--seed`, `--test-size`, `--export-metrics`,
`--artifacts-dir`. Neural models add `--base-model`, `--max-length`,
`--batch-size`, `--epochs`, `--learning-rate`, `--weight-decay`, `--dropout`,
`--patience`; `encoder_LoRA` and `LLM_LoRA` add `--lora-r`, `--lora-alpha`,
`--lora-dropout`, and `--lora-target-modules`.

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

Neural models (`encoder_head`, `encoder_LoRA`, `LLM_train_head`, `LLM_LoRA`) have
a train/val/test split, so in span mode they auto-select their aggregation and
decision threshold (F1-maximizing on validation by default) and record it in an
`operating_point` block. Classical models have no validation split and are
deliberately left at `max` / `0.5`.

```powershell
python source\model_training\naive_bayes.py --use-spans True --export-metrics True
python source\model_training\tfidf_logistic_regression.py --use-spans True --aggregation noisy_or --export-metrics True
python source\model_training\encoder_head.py --use-spans True --target-precision 0.6 --dropout 0.1 --weight-decay 0.01 --export-metrics True
python source\model_training\encoder_LoRA.py --use-spans True --aggregation noisy_or --patience 3 --dropout 0.1 --export-metrics True
python source\model_training\LLM_LoRA.py --use-spans True --aggregation noisy_or --patience 3 --dropout 0.1 --export-metrics True
```
