import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, parse_bool


LABEL_FIELD = "hallucination"
MODEL_NAME = "majority_voting"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "general_data.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object on line {line_number}, got {type(item).__name__}")
            records.append(item)
        return records

    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Expected every list item to be a JSON object")
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Expected JSON object, JSON list, or JSONL file, got {type(data).__name__}")


def split_records(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("Need at least 2 records to split the dataset")

    shuffled_records = records[:]
    random.Random(seed).shuffle(shuffled_records)

    test_count = max(1, round(len(shuffled_records) * test_size))
    test_records = shuffled_records[:test_count]
    train_records = shuffled_records[test_count:]

    if not train_records:
        raise ValueError("Training split is empty. Use a smaller test_size.")

    return train_records, test_records


def choose_majority_label(records: list[dict[str, Any]]) -> str:
    label_counts = Counter(record.get(LABEL_FIELD) for record in records)
    label_counts.pop(None, None)

    if not label_counts:
        raise ValueError(f"No labels found in field {LABEL_FIELD!r}")

    return label_counts.most_common(1)[0][0]


def print_label_distribution(title: str, records: list[dict[str, Any]]) -> None:
    label_counts = Counter(record.get(LABEL_FIELD, "<missing>") for record in records)
    print(f"\n{title}")
    print("-" * len(title))
    for label, count in label_counts.most_common():
        percentage = count / len(records) * 100 if records else 0.0
        print(f"{label}: {count} ({percentage:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a majority-voting baseline model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset. Defaults to {DEFAULT_DATA_PATH}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed used to split the data.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of examples to use for testing.",
    )
    parser.add_argument(
        "--export-metrics",
        type=parse_bool,
        default=False,
        help="True exports test metrics JSON to artifacts/model_name. False skips export.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Model name used for artifact export. Defaults to {MODEL_NAME}.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory where exported metrics are written. Defaults to {DEFAULT_ARTIFACTS_DIR}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)
    prediction = choose_majority_label(train_records)
    y_true = [record.get(LABEL_FIELD) for record in test_records]
    y_pred = [prediction] * len(test_records)
    positive_score = 1.0 if prediction == "yes" else 0.0
    y_score = [positive_score] * len(test_records)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    metrics_payload = {
        "model_name": args.model_name,
        "data_path": str(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "majority_label": prediction,
        "metrics": metrics,
    }
    metrics_path = maybe_export_metrics_json(
        enabled=args.export_metrics,
        model_name=args.model_name,
        metrics=metrics_payload,
        artifacts_dir=args.artifacts_dir,
    )

    print("Majority Voting Baseline")
    print("------------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(train_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"majority_label: {prediction}")
    print(f"accuracy: {metrics['accuracy']:.4f} ({correct}/{len(test_records)})")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "pr_auc: undefined")
    print(f"roc_auc: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "roc_auc: undefined")
    if metrics_path is not None:
        print(f"metrics_path: {metrics_path}")

    print_label_distribution("Train Label Distribution", train_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
