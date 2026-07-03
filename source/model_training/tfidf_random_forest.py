import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# utility class imports
from source.utils.text import TfidfVectorizer

# utility function imports
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, parse_bool
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.text import record_to_text

# utility constant imports
from source.utils.data import LABEL_FIELD, DEFAULT_DATA_PATH, DEFAULT_ARTIFACTS_DIR, DEFAULT_SEED, DEFAULT_TEST_SIZE
from source.utils.text import TEXT_FIELDS, POSITIVE_LABEL

# file specific constants
MODEL_NAME = "tfidf_random_forest"

class DecisionTree:
    def __init__(self, max_depth: int, min_samples_split: int, max_features_split: int, seed: int):
        if(max_depth <= 0):
            raise ValueError("max_depth must be greater than 0")
        if(min_samples_split < 2):
            raise ValueError("min_samples_split must be at least 2")
        if(max_features_split <= 0):
            raise ValueError("max_features_split must be greater than 0")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_split = max_features_split
        self.seed = seed

        # creating the root of the tree
        self.root: "DecisionTree._Node | None" = None

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int) -> None:
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.feature_count = feature_count

        self._rng = random.Random(self.seed)
        # the list of indices of the samples the tree will be grown based on
        indices = list(range(len(vectors)))
        # create a tree root
        self.root = self._grow_tree(indices, vectors, labels, depth=0)

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        return [self._traverse_tree(vector, self.root) for vector in vectors]

    def _grow_tree(self, indices: list[int], vectors: list[dict[int, float]], labels: list[int], depth: int) -> "DecisionTree._Node":
        n = len(indices)
        # the fraction of samples at this node with a positive label
        p = sum(labels[index] for index in indices) / n

        # stopping conditions for the recursion
        if (depth >= self.max_depth or n < self.min_samples_split):
            return DecisionTree._Node(value=p)
        if (p == 0.0 or p == 1.0):
            return DecisionTree._Node(value=p)

        # drawing a random subset of features
        subset_size = min(self.max_features_split, self.feature_count)
        feature_subset = self._rng.sample(range(self.feature_count), k=subset_size)
        split = self._best_split(indices, vectors, labels, feature_subset)
        # no split improved impurity, so make this a leaf
        if split is None:
            return DecisionTree._Node(value=p)

        feature_index, threshold, left_indices, right_indices = split
        left = self._grow_tree(left_indices, vectors, labels, depth + 1)
        right = self._grow_tree(right_indices, vectors, labels, depth + 1)
        return DecisionTree._Node(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def _best_split(self, indices: list[int], vectors: list[dict[int, float]], labels: list[int], feature_subset: list[int]) -> tuple[int, float, list[int], list[int]] | None:
        parent_gini = self._gini(indices, labels)
        n = len(indices)
        best_gain = 0.0
        best_split = None

        for feature_index in feature_subset:
            # threshold 0.0 splits term-absent (left) from term-present (right)
            left_indices = []
            right_indices = []
            for index in indices:
                if vectors[index].get(feature_index, 0.0) <= 0.0:
                    left_indices.append(index)
                else:
                    right_indices.append(index)

            # a split that sends everything one way tells us nothing
            if not left_indices or not right_indices:
                continue

            left_gini = self._gini(left_indices, labels)
            right_gini = self._gini(right_indices, labels)
            weighted = (len(left_indices) / n) * left_gini + (len(right_indices) / n) * right_gini
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best_split = (feature_index, 0.0, left_indices, right_indices)

        return best_split

    def _gini(self, indices: list[int], labels: list[int]) -> float:
        p = sum(labels[index] for index in indices) / len(indices)
        return 1 - (p ** 2 + (1 - p) ** 2)

    def _traverse_tree(self, vector: dict[int, float], node: "DecisionTree._Node") -> float:
        # a leaf carries the positive-label probability
        if node.value is not None:
            return node.value
        if vector.get(node.feature_index, 0.0) <= node.threshold:
            return self._traverse_tree(vector, node.left)
        return self._traverse_tree(vector, node.right)

    class _Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value: float | None = None):
            # internal node has values for feature_index, threshold, left and right
            # leaf node has a value (probability)
            # node is a leaf iff value is not None
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

class RandomForest:
    def __init__(self, n_estimators: int, max_depth: int, min_samples_split: int, max_features_split: int, seed: int):
        if(n_estimators <= 0):
            raise ValueError("n_estimators must be greater than 0")
        if(max_depth <= 0):
            raise ValueError("max_depth must be greater than 0")
        if(min_samples_split < 2):
            raise ValueError("min_samples_split must be at least 2")
        if(max_features_split <= 0):
            raise ValueError("max_features_split must be greater than 0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_split = max_features_split
        self.seed = seed
        self.trees: list[DecisionTree] = []

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int) -> None:
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.trees = []
        n = len(vectors)
        for i in range(self.n_estimators):
            # each tree gets a distinct seed so its bootstrap and feature draws differ
            rng = random.Random(self.seed + i)
            # bootstrap sample: n rows drawn with replacement
            sample = [rng.randrange(n) for _ in range(n)]
            bootstrap_vectors = [vectors[index] for index in sample]
            bootstrap_labels = [labels[index] for index in sample]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features_split=self.max_features_split,
                seed=self.seed + i,
            )
            tree.fit(bootstrap_vectors, bootstrap_labels, feature_count)
            self.trees.append(tree)

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        totals = [0.0] * len(vectors)
        for tree in self.trees:
            for index, score in enumerate(tree.predict_positive_scores(vectors)):
                totals[index] += score
        return [total / len(self.trees) for total in totals]

    def predict(self, vectors: list[dict[int, float]]) -> list[str]:
        return [
            POSITIVE_LABEL if score >= 0.5 else "no"
            for score in self.predict_positive_scores(vectors)
        ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a TF-IDF random forest baseline model.")
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
    # parsers relating to tf-idf interactions
    parser.add_argument("--max-features", type=int, default=500, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for terms.")
    # parsers relating specifically to random-forest parameters
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=20, help="Maximum depth of each tree.")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split a node.")
    parser.add_argument("--max-features-split", type=int, default=32, help="Number of features considered at each split.")
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
    train_labels = [1 if record[LABEL_FIELD] == POSITIVE_LABEL else 0 for record in train_records]
    test_texts = [record_to_text(record) for record in test_records]
    y_true = [record[LABEL_FIELD] for record in test_records]

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    model = RandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features_split=args.max_features_split,
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
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "max_features_split": args.max_features_split,
        "vocabulary_size": len(vectorizer.vocabulary),
        "tree_count": len(model.trees),
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

    print("TF-IDF Random Forest Baseline")
    print("-----------------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"max_features: {args.max_features}")
    print(f"min_df: {args.min_df}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"min_samples_split: {args.min_samples_split}")
    print(f"max_features_split: {args.max_features_split}")
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

if __name__=="__main__":
    main()
