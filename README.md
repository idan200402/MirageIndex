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
- `source/utils/training_metrics.py` - shared metric calculation and optional JSON export logic.
