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
the computed `class_weight`).

### Span-mode flags

These are added to every model (inert unless `--use-spans True` is passed):

- `--use-spans` - `True` trains at the chunk level; `False` (default) is the document-level baseline.
- `--chunk-window` - word-window size for chunking (default `40`).
- `--chunk-stride` - word-window stride between chunks (default `20`).
- `--overlap-threshold` - min fraction of a chunk a span must cover to label it hallucinated (default `0.5`).
- `--aggregation` - how chunk scores combine into a document score: `max` (default), `mean_topk`, or `noisy_or`.
- `--top-k` - `K` for the `mean_topk` aggregation (default `3`).
- `--response-threshold` - decision threshold on the aggregated response score (default `0.5`).

```powershell
python source\model_training\naive_bayes.py --use-spans True --export-metrics True
python source\model_training\tfidf_logistic_regression.py --use-spans True --aggregation noisy_or --export-metrics True
python source\model_training\tfidf_random_forest.py --use-spans True --chunk-window 40 --chunk-stride 20 --export-metrics True
python source\model_training\encoder_head.py --use-spans True --export-metrics True
```
