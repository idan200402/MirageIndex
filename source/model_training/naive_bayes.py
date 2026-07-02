import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# utility function imports
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, parse_bool
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.text import tokenize, record_to_text

# utility constant imports
from source.utils.data import LABEL_FIELD, DEFAULT_DATA_PATH, DEFAULT_ARTIFACTS_DIR, DEFAULT_SEED, DEFAULT_TEST_SIZE
from source.utils.text import TEXT_FIELDS, POSITIVE_LABEL

# file specific constants
MODEL_NAME = "naive_bayes"

class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha
        self.labels: list[str] = []
        self.vocabulary: set[str] = set()
        self.label_counts: Counter[str] = Counter()
        self.token_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens_by_label: Counter[str] = Counter()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("Cannot train on an empty dataset")

        for text, label in zip(texts, labels):
            self.label_counts[label] += 1
            tokens = tokenize(text)
            self.vocabulary.update(tokens)
            self.token_counts_by_label[label].update(tokens)
            self.total_tokens_by_label[label] += len(tokens)

        self.labels = sorted(self.label_counts)

    def predict(self, texts: list[str]) -> list[str]:
        return [max(self._log_probabilities(text), key=self._log_probabilities(text).get) for text in texts]

    def predict_positive_scores(self, texts: list[str], positive_label: str) -> list[float]:
        return [self._probabilities(text).get(positive_label, 0.0) for text in texts]

    def _log_probabilities(self, text: str) -> dict[str, float]:
        total_examples = sum(self.label_counts.values())
        vocabulary_size = len(self.vocabulary)
        tokens = tokenize(text)
        log_probabilities = {}

        for label in self.labels:
            log_probability = math.log(self.label_counts[label] / total_examples)
            denominator = self.total_tokens_by_label[label] + self.alpha * vocabulary_size

            for token in tokens:
                token_count = self.token_counts_by_label[label][token]
                log_probability += math.log((token_count + self.alpha) / denominator)

            log_probabilities[label] = log_probability

        return log_probabilities

    def _probabilities(self, text: str) -> dict[str, float]:
        log_probabilities = self._log_probabilities(text)
        max_log_probability = max(log_probabilities.values())
        exp_values = {
            label: math.exp(log_probability - max_log_probability)
            for label, log_probability in log_probabilities.items()
        }
        normalizer = sum(exp_values.values())
        return {label: value / normalizer for label, value in exp_values.items()}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a Naive Bayes text baseline model.")
    # parsers relating to general model interactions
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
    # parsers relating specifically to naive bayes parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing value.",
    )
    # parsers relating to artifact exports and test matrices
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

    train_texts = [record_to_text(record) for record in train_records]
    train_labels = [record[LABEL_FIELD] for record in train_records]
    test_texts = [record_to_text(record) for record in test_records]
    y_true = [record[LABEL_FIELD] for record in test_records]

    model = MultinomialNaiveBayes(alpha=args.alpha)
    model.fit(train_texts, train_labels)

    y_pred = model.predict(test_texts)
    y_score = model.predict_positive_scores(test_texts, POSITIVE_LABEL)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    metrics_payload = {
        "model_name": args.model_name,
        "data_path": str(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "alpha": args.alpha,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "vocabulary_size": len(model.vocabulary),
        "text_fields": list(TEXT_FIELDS),
        "trained_parameters": {
            "alpha": args.alpha,
            "labels": model.labels,
            "label_counts": dict(model.label_counts),
            "total_tokens_by_label": dict(model.total_tokens_by_label),
            "vocabulary_size": len(model.vocabulary),
        },
        "metrics": metrics,
    }
    metrics_path = maybe_export_metrics_json(
        enabled=args.export_metrics,
        model_name=args.model_name,
        metrics=metrics_payload,
        artifacts_dir=args.artifacts_dir,
    )

    print("Naive Bayes Baseline")
    print("--------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"alpha: {args.alpha}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(train_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"vocabulary_size: {len(model.vocabulary)}")
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
