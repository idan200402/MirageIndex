import argparse
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# utility class imports
from source.utils.text import TfidfVectorizer

# utility function imports
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, project_relative_path
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.text import record_to_text
from source.utils.general import add_common_parsing, sigmoid

# utility constant imports
from source.utils.data import LABEL_FIELD
from source.utils.text import TEXT_FIELDS, POSITIVE_LABEL

# file specific constants
MODEL_NAME = "tfidf_logistic_regression"

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
        """Train the weight vector and bias with stochastic gradient descent.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.
        labels: binary labels (1 = positive, 0 = negative), aligned with vectors.
        feature_count: size of the vocabulary / feature space.\\
        Returns nothing.\\
        The fitted weights and bias are stored on the instance.
        """
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.weights = [0.0] * feature_count
        self.bias = 0.0
        indices = list(range(len(vectors)))
        rng = random.Random(self.seed)

        for _ in range(self.epochs):
            # reshuffling each epoch decorrelates the update order between passes
            rng.shuffle(indices)
            for index in indices:
                vector = vectors[index]
                label = labels[index]
                # the error is the residual between predicted probability and the label
                probability = sigmoid(self._decision_function(vector))
                error = probability - label

                # only the features present in this row have a non-zero gradient
                for feature_index, value in vector.items():
                    # the l2 term pulls each weight back toward zero to curb overfitting
                    gradient = error * value + self.l2 * self.weights[feature_index]
                    self.weights[feature_index] -= self.learning_rate * gradient

                # the bias absorbs the class prior and is updated without regularization
                self.bias -= self.learning_rate * error

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        """Return the predicted positive-class probability for each input vector.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns one sigmoid-squashed probability in [0, 1] per vector.
        """
        return [sigmoid(self._decision_function(vector)) for vector in vectors]

    def predict(self, vectors: list[dict[int, float]]) -> list[str]:
        """Return a hard class label per vector by thresholding the positive score at 0.5.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns the positive label or "no" for each vector.
        """
        return [
            POSITIVE_LABEL if score >= 0.5 else "no"
            for score in self.predict_positive_scores(vectors)
        ]

    def _decision_function(self, vector: dict[int, float]) -> float:
        """Return the raw linear score (logit) for a single vector before the sigmoid.

        vector: one sparse TF-IDF row as a {feature_index: weight} map.\\
        Returns bias plus the dot product of the weights with the present features.
        """
        return self.bias + sum(self.weights[index] * value for index, value in vector.items())



def top_weighted_terms(vectorizer: TfidfVectorizer, model: LogisticRegression, limit: int = 10) -> dict[str, list[dict[str, float]]]:
    """Return the most influential vocabulary terms in each direction of the decision.

    vectorizer: the fitted vectorizer whose vocabulary maps terms to indices.
    model: the trained logistic regression whose weights rank the terms.
    limit: how many terms to keep per side.\\
    Returns a dict with the top positive-pushing and negative-pushing terms and weights.
    """
    # invert the vocabulary so we can label each weight with its term
    index_to_term = {index: term for term, index in vectorizer.vocabulary.items()}
    weighted_terms = [
        (index_to_term[index], weight)
        for index, weight in enumerate(model.weights)
        if index in index_to_term
    ]
    # the largest positive weights push toward the positive label
    positive_terms = sorted(weighted_terms, key=lambda item: item[1], reverse=True)[:limit]
    # the most negative weights push toward the negative label
    negative_terms = sorted(weighted_terms, key=lambda item: item[1])[:limit]

    return {
        "positive_label_terms": [{"term": term, "weight": weight} for term, weight in positive_terms],
        "negative_label_terms": [{"term": term, "weight": weight} for term, weight in negative_terms],
    }



def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the TF-IDF and logistic regression specific options. 
    Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a TF-IDF logistic regression baseline model.")

    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # model specific parser
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Model name used for artifact export. Defaults to {MODEL_NAME}.",
    )
    # parsers relating to tf-idf interactions
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for terms.")
    # parsers relating specifically to logistic regression parameters
    parser.add_argument("--learning-rate", type=float, default=0.5, help="Logistic regression learning rate.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--l2", type=float, default=0.0001, help="L2 regularization strength.")
   
    return parser.parse_args()

def main() -> None:
    """Run the end-to-end training and evaluation pipeline for this baseline.

    Loads the dataset, splits it, vectorizes the text, trains the logistic regression
    model, evaluates it on the test split, optionally exports the metrics JSON, and
    prints a human-readable summary. Takes no arguments and returns nothing.
    """
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
        "data_path": project_relative_path(args.data),
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
