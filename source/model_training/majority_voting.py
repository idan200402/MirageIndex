import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# utility function imports 
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, project_relative_path
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.general import add_common_parsing

# utility constant imports
from source.utils.data import LABEL_FIELD

# file specific constants
MODEL_NAME = "majority_voting"

def choose_majority_label(records: list[dict[str, Any]]) -> str:
    """Return the label that appears most often across the given records.

    records: dataset rows, each expected to carry a LABEL_FIELD entry.\\
    Returns the single most common label, raises ValueError if none are present.
    """
    # tally how many times each label value shows up
    label_counts = Counter(record.get(LABEL_FIELD) for record in records)
    # records missing a label contribute a None key, which is not a real class
    label_counts.pop(None, None)

    if not label_counts:
        raise ValueError(f"No labels found in field {LABEL_FIELD!r}")

    # most_common(1) returns [(label, count)], so index into the label
    return label_counts.most_common(1)[0][0]

def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    The majority-voting baseline only needs the arguments shared by every model plus a model name.
    Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a majority-voting baseline model.")
    
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # model specific parser
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Model name used for artifact export. Defaults to {MODEL_NAME}.",
    )
    return parser.parse_args()

def main() -> None:
    """Run the end-to-end evaluation pipeline for the majority-voting baseline.

    Loads the dataset, splits it, learns the majority label from the train split,
    predicts that same label for every test row, optionally exports the metrics JSON,
    and prints a human-readable summary. Takes no arguments and returns nothing.
    """
    args = parse_args()
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)
    # the only thing this baseline "learns" is the most common training label
    prediction = choose_majority_label(train_records)
    y_true = [record.get(LABEL_FIELD) for record in test_records]
    # every test row receives the same predicted label
    y_pred = [prediction] * len(test_records)
    # a constant prediction yields a constant positive score of 1.0 or 0.0
    positive_score = 1.0 if prediction == "yes" else 0.0
    y_score = [positive_score] * len(test_records)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    metrics_payload = {
        "model_name": args.model_name,
        "data_path": project_relative_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "majority_label": prediction,
        "trained_parameters": {
            "majority_label": prediction,
            "positive_label_score": positive_score,
        },
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
