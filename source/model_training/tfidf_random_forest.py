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
from source.utils.training_metrics import classification_metrics, maybe_export_metrics_json, project_relative_path
from source.utils.data import load_records, split_records, print_label_distribution
from source.utils.text import record_to_text, build_train_chunk_examples
from source.utils.general import (
    add_common_parsing,
    spans_suffix,
    score_records_by_chunks,
    compute_class_weight_ratio,
    span_config_payload,
)

# utility constant imports
from source.utils.data import LABEL_FIELD
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

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int, class_weight: float = 1.0) -> None:
        """Grow a single decision tree from the given samples.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.
        labels: binary labels (1 = positive, 0 = negative), aligned with vectors.
        feature_count: size of the feature space that splits are drawn from.
        class_weight: per-example sample weight applied to positive-labeled samples in
        the impurity and leaf calculations (1.0, the default, reproduces the baseline).
        Returns nothing. \\
        The fitted tree is stored on self.root.
        """
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.feature_count = feature_count
        # positives carry class_weight, negatives carry 1.0; all-1.0 reproduces baseline
        self._sample_weights = [class_weight if label == 1 else 1.0 for label in labels]

        self._rng = random.Random(self.seed)
        # the list of indices of the samples the tree will be grown based on
        indices = list(range(len(vectors)))
        # create a tree root
        self.root = self._grow_tree(indices, vectors, labels, depth=0)

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        """Return this tree's positive-label probability for each input vector.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.
        Returns the leaf probability reached by each vector, in input order.
        """
        return [self._traverse_tree(vector, self.root) for vector in vectors]

    def _grow_tree(self, indices: list[int], vectors: list[dict[int, float]], labels: list[int], depth: int) -> "DecisionTree._Node":
        """Recursively build a node from the samples referenced by indices.

        indices: positions into vectors/labels handled at this node.
        depth: current recursion depth, checked against max_depth.\\
        Returns an internal node when an impurity-reducing split is found, otherwise a leaf.
        """
        n = len(indices)
        # the weight-adjusted fraction of samples at this node with a positive label
        p = self._positive_fraction(indices, labels)

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
        """Find the feature in feature_subset that most reduces Gini impurity.

        indices: samples to partition.
        feature_subset: candidate features to test. \\
        Returns (feature_index, threshold, left_indices, right_indices) for the best
        gain, or None when no split improves impurity.
        """
        parent_gini = self._gini(indices, labels)
        # combine child impurities by their weighted sizes so sample weights carry through
        total_weight = self._weight_sum(indices)
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
            left_weight = self._weight_sum(left_indices)
            right_weight = self._weight_sum(right_indices)
            weighted = (left_weight / total_weight) * left_gini + (right_weight / total_weight) * right_gini
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best_split = (feature_index, 0.0, left_indices, right_indices)

        return best_split

    def _positive_fraction(self, indices: list[int], labels: list[int]) -> float:
        """Return the weight-adjusted positive-label fraction of the given samples.

        indices: positions into labels/sample weights. labels: the binary label list.\\
        Returns the positive weight over the total weight, which equals the plain
        positive fraction when every sample weight is 1.0 (the baseline).
        """
        total_weight = self._weight_sum(indices)
        positive_weight = sum(self._sample_weights[index] for index in indices if labels[index] == 1)
        return positive_weight / total_weight

    def _weight_sum(self, indices: list[int]) -> float:
        """Return the total sample weight of the samples referenced by indices."""
        return sum(self._sample_weights[index] for index in indices)

    def _gini(self, indices: list[int], labels: list[int]) -> float:
        """Return the weighted binary Gini impurity of the samples referenced by indices.

        indices: positions into labels to measure.
        labels: the binary label list.\\
        Returns the impurity, which is 0 for a pure node and 0.5 for an even split.
        Reduces to the plain Gini impurity when every sample weight is 1.0.
        """
        p = self._positive_fraction(indices, labels)
        return 1 - (p ** 2 + (1 - p) ** 2)

    def _traverse_tree(self, vector: dict[int, float], node: "DecisionTree._Node") -> float:
        """Walk a single vector from node down to a leaf and return its probability.

        vector: one sparse TF-IDF row.
        node: current node in the recursion.\\
        Returns the leaf's stored positive-label probability.
        """
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

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int, class_weight: float = 1.0) -> None:
        """Train the forest by fitting each tree to its own bootstrap sample.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.
        labels: binary labels (1 = positive, 0 = negative), aligned with vectors.
        feature_count: size of the vocabulary / feature space.
        class_weight: per-example sample weight for positive chunks, forwarded to each
        tree's impurity calculation (1.0, the default, reproduces the baseline).\\
        Returns nothing.\\
        The fitted trees are stored on the instance.
        """
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
            tree.fit(bootstrap_vectors, bootstrap_labels, feature_count, class_weight=class_weight)
            self.trees.append(tree)

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        """Return the forest's positive-label probability for each input vector.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns one probability per vector, averaged across every tree in the forest.
        """
        totals = [0.0] * len(vectors)
        for tree in self.trees:
            for index, score in enumerate(tree.predict_positive_scores(vectors)):
                totals[index] += score
        return [total / len(self.trees) for total in totals]

    def predict(self, vectors: list[dict[int, float]]) -> list[str]:
        """Return a hard class label per vector by thresholding the positive score at 0.5.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns the positive label or "no" for each vector.
        """
        return [
            POSITIVE_LABEL if score >= 0.5 else "no"
            for score in self.predict_positive_scores(vectors)
        ]

def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the TF-IDF and random forest specific options.
    Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a TF-IDF random forest baseline model.")
    
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # model specific parser
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"Model name used for artifact export. Defaults to {MODEL_NAME}.",
    )
    # parsers relating to tf-idf interactions
    parser.add_argument("--max-features", type=int, default=500, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency for terms.")
    # parsers relating specifically to random-forest parameters
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=20, help="Maximum depth of each tree.")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split a node.")
    parser.add_argument("--max-features-split", type=int, default=32, help="Number of features considered at each split.")
    

    return parser.parse_args()

def main() -> None:
    """Run the end-to-end training and evaluation pipeline for this baseline.

    Loads the dataset, splits it, vectorizes the text, trains the random forest model,
    evaluates it on the test split, optionally exports the metrics JSON, and prints
    a human-readable summary. Takes no arguments and returns nothing.
    """
    args = parse_args()
    # fork artifacts to tfidf_random_forest_spans in span mode; inert otherwise
    args.model_name = args.model_name + spans_suffix(args)
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)

    y_true = [record[LABEL_FIELD] for record in test_records]
    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
    model = RandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features_split=args.max_features_split,
        seed=args.seed,
    )
    build_audit = None
    class_weight = 1.0

    if args.use_spans:
        # chunk-level training: each response chunk is one TF-IDF document (the query is
        # dropped, being constant per doc), scored by the same forest with positive-chunk weighting
        _queries, chunks, chunk_labels, build_audit = build_train_chunk_examples(
            train_records, args.chunk_window, args.chunk_stride, args.overlap_threshold
        )
        train_texts = list(chunks)
        train_vectors = vectorizer.fit_transform(train_texts)
        class_weight = compute_class_weight_ratio(chunk_labels)
        model.fit(train_vectors, chunk_labels, feature_count=len(vectorizer.vocabulary), class_weight=class_weight)
        y_score, y_pred = score_records_by_chunks(test_records, model, vectorizer, args)
    else:
        train_texts = [record_to_text(record) for record in train_records]
        train_labels = [1 if record[LABEL_FIELD] == POSITIVE_LABEL else 0 for record in train_records]
        test_texts = [record_to_text(record) for record in test_records]
        train_vectors = vectorizer.fit_transform(train_texts)
        test_vectors = vectorizer.transform(test_texts)
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
        "data_path": project_relative_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "trained_parameters": trained_parameters,
        "metrics": metrics,
    }
    if args.use_spans:
        # span-mode only additions; the baseline payload stays byte-identical
        metrics_payload["span_config"] = span_config_payload(args)
        metrics_payload["span_coverage"] = build_audit
        metrics_payload["class_weight"] = class_weight
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
