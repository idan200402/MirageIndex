import json
import os
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def project_relative_path(path: str | Path) -> str:
    """Convert a filesystem path into a portable, project-root-relative string.

    path: an absolute or relative path to a dataset or artifact.\\
    Returns the path relative to the project root using forward slashes, so the
    value written into exported artifacts is identical on every machine and
    operating system. Falls back to the absolute POSIX form only when the path
    lives on a different drive and cannot be expressed relative to the root.
    """
    absolute_path = Path(path).resolve()
    try:
        relative_path = os.path.relpath(absolute_path, PROJECT_ROOT)
    except ValueError:
        # os.path.relpath raises on Windows when the paths sit on different drives
        return absolute_path.as_posix()
    # normalize to forward slashes so exported JSON is byte-identical across OSes
    return Path(relative_path).as_posix()


def parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value

    normalized_value = value.strip().lower()
    if normalized_value in {"true", "1", "yes", "y"}:
        return True
    if normalized_value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Expected a boolean value, got {value!r}")


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_score: Sequence[float],
    positive_label: str = "yes",
) -> dict[str, float | None]:
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("y_true, y_pred, and y_score must have the same length")
    if not y_true:
        raise ValueError("Cannot calculate metrics for an empty dataset")

    true_positive = sum(actual == positive_label and predicted == positive_label for actual, predicted in zip(y_true, y_pred))
    false_positive = sum(actual != positive_label and predicted == positive_label for actual, predicted in zip(y_true, y_pred))
    false_negative = sum(actual == positive_label and predicted != positive_label for actual, predicted in zip(y_true, y_pred))
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    return {
        "accuracy": correct / len(y_true),
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "pr_auc": pr_auc(y_true, y_score, positive_label),
        "roc_auc": roc_auc(y_true, y_score, positive_label),
    }


def pr_auc(y_true: Sequence[str], y_score: Sequence[float], positive_label: str) -> float | None:
    positive_total = sum(label == positive_label for label in y_true)
    if positive_total == 0:
        return None

    paired_values = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0

    while index < len(paired_values):
        score = paired_values[index][0]
        group_positive = 0
        group_negative = 0

        while index < len(paired_values) and paired_values[index][0] == score:
            if paired_values[index][1] == positive_label:
                group_positive += 1
            else:
                group_negative += 1
            index += 1

        true_positive += group_positive
        false_positive += group_negative
        recall = true_positive / positive_total
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall

    return area


def roc_auc(y_true: Sequence[str], y_score: Sequence[float], positive_label: str) -> float | None:
    positives = [score for label, score in zip(y_true, y_score) if label == positive_label]
    negatives = [score for label, score in zip(y_true, y_score) if label != positive_label]

    if not positives or not negatives:
        return None

    greater_count = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                greater_count += 1.0
            elif positive_score == negative_score:
                greater_count += 0.5

    return greater_count / (len(positives) * len(negatives))


def export_metrics_json(
    model_name: str,
    metrics: dict[str, Any],
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    filename: str = "metrics.json",
) -> Path:
    output_dir = artifacts_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics.setdefault("trained_parameters", {})
    output_path = output_dir / filename
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def maybe_export_metrics_json(
    enabled: bool,
    model_name: str,
    metrics: dict[str, Any],
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Path | None:
    if not enabled:
        return None
    return export_metrics_json(model_name=model_name, metrics=metrics, artifacts_dir=artifacts_dir)
