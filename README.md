# MirageIndex

Final course project for exploring hallucination-labeled examples and training simple baseline models.

## Folders

- `data/` - dataset files.
- `source/` - analysis and model-training code.
- `source/model_training/` - model scripts.
- `source/utils/` - shared helper logic for model training.
- `artifacts/` - generated model outputs such as exported metrics.

## Main Files

- `data/general_data.json` - The JSON dataset with `user_query`, `chatgpt_response`, `hallucination`, and `hallucination_spans`.
- `source/shallow_stats.py` - prints shallow dataset statistics and full example samples with different labels.
- `source/model_training/majority_voting.py` - baseline that predicts the majority label from the train split.
- `source/model_training/naive_bayes.py` - bag-of-words Naive Bayes baseline using `user_query` and `chatgpt_response`.
- `source/model_training/tfidf_logistic_regression.py` - TF-IDF text features with a logistic regression classifier.
- `source/utils/training_metrics.py` - shared metric calculation and optional JSON export logic.
- `source/utils/plot_stats.py` - creates an SVG plot comparing model F1 and accuracy scores from `artifacts/`.

## Run

```powershell
python source\shallow_stats.py
python source\model_training\majority_voting.py --export-metrics True
python source\model_training\naive_bayes.py --export-metrics True
python source\model_training\tfidf_logistic_regression.py --export-metrics True
python source\model_training\tfidf_random_forest.py --export-metrics True
python source\model_training\tfidf_xgboost.py --export-metrics True
python source\model_training\encoder_head.py --export-metrics True
python source\model_training\LLM_LoRA.py --export-metrics True
python source\model_training\LLM_train_head.py --export-metrics True
python source\utils\plot_stats.py
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
