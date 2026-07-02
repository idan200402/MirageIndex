import argparse
import json
import math
import random
import re
import sys
from collections import Counter
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
MODEL_NAME = "tfidf_logistic_regression"


class TfidfVectorizer:
    def __init__(self, max_features: int = 20000, min_df: int = 1) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be greater than 0")
        if min_df <= 0:
            raise ValueError("min_df must be greater than 0")
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: list[str]) -> None:
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()

        for text in texts:
            tokens = tokenize(text)
            term_frequency.update(tokens)
            document_frequency.update(set(tokens))

        terms = [
            term
            for term, frequency in term_frequency.most_common()
            if document_frequency[term] >= self.min_df
        ][: self.max_features]

        self.vocabulary = {term: index for index, term in enumerate(terms)}
        document_count = len(texts)
        self.idf = [
            math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in terms
        ]

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        if not self.vocabulary:
            raise ValueError("Vectorizer must be fitted before transform")

        vectors = []
        for text in texts:
            counts: Counter[int] = Counter()
            for token in tokenize(text):
                index = self.vocabulary.get(token)
                if index is not None:
                    counts[index] += 1

            total_terms = sum(counts.values())
            vector = {}
            if total_terms:
                for index, count in counts.items():
                    vector[index] = (count / total_terms) * self.idf[index]

                norm = math.sqrt(sum(value * value for value in vector.values()))
                if norm:
                    vector = {index: value / norm for index, value in vector.items()}

            vectors.append(vector)

        return vectors

    def fit_transform(self, texts: list[str]) -> list[dict[int, float]]:
        self.fit(texts)
        return self.transform(texts)


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.5, epochs: int = 80, l2: float = 0.0001, seed: int = 42) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if l2 < 0:
            raise ValueError("l2 must be greater than or equal to 0")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int) -> None:
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")

        self.weights = [0.0] * feature_count
        self.bias = 0.0
        indices = list(range(len(vectors)))
        rng = random.Random(self.seed)

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for index in indices:
                vector = vectors[index]
                label = labels[index]
                probability = sigmoid(self._decision_function(vector))
                error = probability - label

                for feature_index, value in vector.items():
                    gradient = error * value + self.l2 * self.weights[feature_index]
                    self.weights[feature_index] -= self.learning_rate * gradient

                self.bias -= self.learning_rate * error

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        return [sigmoid(self._decision_function(vector)) for vector in vectors]

    def predict(self, vectors: list[dict[int, float]]) -> list[str]:
        return [
            POSITIVE_LABEL if score >= 0.5 else "no"
            for score in self.predict_positive_scores(vectors)
        ]

    def _decision_function(self, vector: dict[int, float]) -> float:
        return self.bias + sum(self.weights[index] * value for index, value in vector.items())


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)





def top_weighted_terms(vectorizer: TfidfVectorizer, model: LogisticRegression, limit: int = 10) -> dict[str, list[dict[str, float]]]:
    index_to_term = {index: term for term, index in vectorizer.vocabulary.items()}
    weighted_terms = [
        (index_to_term[index], weight)
        for index, weight in enumerate(model.weights)
        if index in index_to_term
    ]
    positive_terms = sorted(weighted_terms, key=lambda item: item[1], reverse=True)[:limit]
    negative_terms = sorted(weighted_terms, key=lambda item: item[1])[:limit]

    return {
        "positive_label_terms": [{"term": term, "weight": weight} for term, weight in positive_terms],
        "negative_label_terms": [{"term": term, "weight": weight} for term, weight in negative_terms],
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a TF-IDF logistic regression baseline model.")
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
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for terms.")
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Logistic regression learning rate.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--l2", type=float, default=0.0001, help="L2 regularization strength.")
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
    train_labels = [1 if record[LABEL_FIELD] == POSITIVE_LABEL else 0 for record in train_records]
    test_texts = [record_to_text(record) for record in test_records]
    y_true = [record[LABEL_FIELD] for record in test_records]

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    model = LogisticRegression(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
    )
    model.fit(train_vectors, train_labels, feature_count=len(vectorizer.vocabulary))

    y_pred = model.predict(test_vectors)
    y_score = model.predict_positive_scores(test_vectors)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    trained_parameters = {
        "text_fields": list(TEXT_FIELDS),
        "positive_label": POSITIVE_LABEL,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "l2": args.l2,
        "vocabulary_size": len(vectorizer.vocabulary),
        "bias": model.bias,
        "weights_count": len(model.weights),
        **top_weighted_terms(vectorizer, model),
    }
    metrics_payload = {
        "model_name": args.model_name,
        "data_path": str(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "trained_parameters": trained_parameters,
        "metrics": metrics,
    }
    metrics_path = maybe_export_metrics_json(
        enabled=args.export_metrics,
        model_name=args.model_name,
        metrics=metrics_payload,
        artifacts_dir=args.artifacts_dir,
    )

    print("TF-IDF Logistic Regression Baseline")
    print("-----------------------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"max_features: {args.max_features}")
    print(f"min_df: {args.min_df}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"epochs: {args.epochs}")
    print(f"l2: {args.l2}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(train_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"vocabulary_size: {len(vectorizer.vocabulary)}")
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
