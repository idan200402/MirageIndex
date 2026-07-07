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
python source\model_training\LLM_LoRA.py --export-mertics True
python source\model_training\LLM_train_head.py --export-metrics True
python source\utils\plot_stats.py
```
