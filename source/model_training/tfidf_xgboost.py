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
from source.utils.text import record_to_text, build_train_chunk_examples
from source.utils.general import (
    add_common_parsing,
    sigmoid,
    spans_suffix,
    score_records_by_chunks,
    compute_class_weight_ratio,
    span_config_payload,
)

# utility constant imports
from source.utils.data import LABEL_FIELD
from source.utils.text import TEXT_FIELDS, POSITIVE_LABEL

# file specific constants
MODEL_NAME = "tfidf_xgboost"

class RegressionTree:
    def __init__(self, max_depth: int, min_samples_split: int, max_features_split: int, reg_lambda: float, gamma: float, seed: int):
        if(max_depth <= 0):
            raise ValueError("max_depth must be greater than 0")
        if(min_samples_split < 2):
            raise ValueError("min_samples_split must be at least 2")
        if(max_features_split <= 0):
            raise ValueError("max_features_split must be greater than 0")
        if(reg_lambda < 0):
            raise ValueError("reg_lambda must be greater than or equal to 0")
        if(gamma < 0):
            raise ValueError("gamma must be greater than or equal to 0")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_split = max_features_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.seed = seed

        # creating the root of the tree
        self.root: "RegressionTree._Node | None" = None

    def fit(self, vectors: list[dict[int, float]], gradients: list[float], hessians: list[float], feature_count: int) -> None:
        """Grow a single regression tree that fits the current boosting residuals.

        vectors: sparse TF-IDF rows, each a {feature_index: weight} mapping.
        gradients / hessians: per-sample first- and second-order logistic loss derivatives.
        feature_count: size of the feature space that splits are drawn from. \\
        Returns nothing. \\
        The fitted tree is stored on self.root.
        """
        if not (len(vectors) == len(gradients) == len(hessians)):
            raise ValueError("vectors, gradients and hessians must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.feature_count = feature_count

        self._rng = random.Random(self.seed)
        # the list of indices of the samples the tree will be grown based on
        indices = list(range(len(vectors)))
        # create a tree root
        self.root = self._grow_tree(indices, vectors, gradients, hessians, depth=0)

    def predict_margins(self, vectors: list[dict[int, float]]) -> list[float]:
        """Return this tree's additive margin contribution for each input vector.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns one leaf weight per vector, in the same order.
        """
        return [self._traverse_tree(vector, self.root) for vector in vectors]

    def _grow_tree(self, indices: list[int], vectors: list[dict[int, float]], gradients: list[float], hessians: list[float], depth: int) -> "RegressionTree._Node":
        """Recursively build a node from the samples referenced by indices.

        indices: positions into vectors/gradients/hessians handled at this node.
        depth: current recursion depth, checked against max_depth.\\
        Returns an internal node when a positive-gain split is found, otherwise a leaf.
        """
        # gradient and hessian totals drive both the leaf weight and the gain
        G = sum(gradients[index] for index in indices)
        H = sum(hessians[index] for index in indices)
        # the optimal leaf weight for this node under the regularized objective
        leaf_weight = -G / (H + self.reg_lambda)

        # stopping conditions for the recursion
        if (depth >= self.max_depth or len(indices) < self.min_samples_split):
            return RegressionTree._Node(value=leaf_weight)

        # drawing a random subset of features
        subset_size = min(self.max_features_split, self.feature_count)
        feature_subset = self._rng.sample(range(self.feature_count), k=subset_size)
        split = self._best_split(indices, vectors, gradients, hessians, feature_subset, G, H)
        # no split produced a positive gain, so make this a leaf
        if split is None:
            return RegressionTree._Node(value=leaf_weight)

        feature_index, left_indices, right_indices = split
        left = self._grow_tree(left_indices, vectors, gradients, hessians, depth + 1)
        right = self._grow_tree(right_indices, vectors, gradients, hessians, depth + 1)
        return RegressionTree._Node(feature_index=feature_index, threshold=0.0, left=left, right=right)

    def _best_split(self, indices: list[int], vectors: list[dict[int, float]], gradients: list[float], hessians: list[float], feature_subset: list[int], G: float, H: float) -> tuple[int, list[int], list[int]] | None:
        """Find the feature in feature_subset that yields the largest split gain.

        indices: samples to partition. feature_subset: candidate features to test.
        G / H: gradient and hessian totals over indices, reused to score each split.\\
        Returns (feature_index, left_indices, right_indices) for the best gain, or None.
        """
        best_gain = 0.0
        best_split = None

        for feature_index in feature_subset:
            # threshold 0.0 splits term-absent (left) from term-present (right)
            left_indices = []
            right_indices = []
            # only present rows are summed 
            # the absent side falls out by subtraction
            G_right = 0.0
            H_right = 0.0
            for index in indices:
                if vectors[index].get(feature_index, 0.0) <= 0.0:
                    left_indices.append(index)
                else:
                    right_indices.append(index)
                    G_right += gradients[index]
                    H_right += hessians[index]

            # a split that sends everything one way tells us nothing
            if not left_indices or not right_indices:
                continue

            G_left = G - G_right
            H_left = H - H_right
            gain = 0.5 * (
                G_left ** 2 / (H_left + self.reg_lambda)
                + G_right ** 2 / (H_right + self.reg_lambda)
                - G ** 2 / (H + self.reg_lambda)
            ) - self.gamma
            if gain > best_gain:
                best_gain = gain
                best_split = (feature_index, left_indices, right_indices)

        return best_split

    def _traverse_tree(self, vector: dict[int, float], node: "RegressionTree._Node") -> float:
        """Walk a single vector from node down to a leaf and return its margin weight.

        vector: one sparse TF-IDF row. node: current node in the recursion.\\
        Returns the leaf's stored margin weight.
        """
        # a leaf carries the additive margin weight
        if node.value is not None:
            return node.value
        if vector.get(node.feature_index, 0.0) <= node.threshold:
            return self._traverse_tree(vector, node.left)
        return self._traverse_tree(vector, node.right)

    class _Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value: float | None = None):
            # internal node has values for feature_index, threshold, left and right
            # leaf node has a value (margin weight)
            # node is a leaf iff value is not None
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

class XGBoost:
    def __init__(self, n_estimators: int, max_depth: int, min_samples_split: int, max_features_split: int, learning_rate: float, reg_lambda: float, gamma: float, seed: int):
        if(n_estimators <= 0):
            raise ValueError("n_estimators must be greater than 0")
        if(max_depth <= 0):
            raise ValueError("max_depth must be greater than 0")
        if(min_samples_split < 2):
            raise ValueError("min_samples_split must be at least 2")
        if(max_features_split <= 0):
            raise ValueError("max_features_split must be greater than 0")
        if(learning_rate <= 0):
            raise ValueError("learning_rate must be greater than 0")
        if(reg_lambda < 0):
            raise ValueError("reg_lambda must be greater than or equal to 0")
        if(gamma < 0):
            raise ValueError("gamma must be greater than or equal to 0")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_split = max_features_split
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.seed = seed
        self.trees: list[RegressionTree] = []
        self.base_score = 0.0

    def fit(self, vectors: list[dict[int, float]], labels: list[int], feature_count: int, class_weight: float = 1.0) -> None:
        """Train the boosted ensemble by fitting one tree per round to the loss gradients.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.
        labels: binary labels (1 = positive, 0 = negative), aligned with vectors.
        feature_count: size of the vocabulary / feature space.
        class_weight: multiplier on the gradient/hessian contribution of positive-labeled
        examples (1.0, the default, reproduces the unweighted baseline exactly).\\
        Returns nothing. \\
        The base score and fitted trees are stored on the instance.
        """
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("Cannot train on an empty dataset")
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than 0")

        self.trees = []
        n = len(vectors)
        # the base score is the log-odds of the training positive rate, clamped to avoid log(0)
        positive_rate = sum(labels) / n
        positive_rate = min(max(positive_rate, 1e-6), 1 - 1e-6)
        self.base_score = math.log(positive_rate / (1 - positive_rate))
        # positives carry class_weight, negatives carry 1.0; all-1.0 reproduces baseline
        sample_weights = [class_weight if label == 1 else 1.0 for label in labels]

        # every sample starts at the base margin and accumulates each tree's contribution
        margins = [self.base_score] * n
        for m in range(self.n_estimators):
            # first- and second-order derivatives of the logistic loss at the current margins,
            # scaled per example so positive chunks weigh more in the boosting objective
            probabilities = [sigmoid(margin) for margin in margins]
            gradients = [
                sample_weight * (probability - label)
                for probability, label, sample_weight in zip(probabilities, labels, sample_weights)
            ]
            hessians = [
                sample_weight * probability * (1 - probability)
                for probability, sample_weight in zip(probabilities, sample_weights)
            ]

            # each tree gets a distinct seed so its feature draws differ
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features_split=self.max_features_split,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                seed=self.seed + m,
            )
            tree.fit(vectors, gradients, hessians, feature_count)

            # shrink the new tree's leaf weights before folding them into the margins
            leaf_margins = tree.predict_margins(vectors)
            for index in range(n):
                margins[index] += self.learning_rate * leaf_margins[index]
            self.trees.append(tree)

    def predict_positive_scores(self, vectors: list[dict[int, float]]) -> list[float]:
        """Return the predicted positive-class probability for each input vector.

        vectors: sparse TF-IDF rows as {feature_index: weight} maps.\\
        Returns one probability in [0, 1] per vector, obtained by summing the base
        score and every tree's shrunken margin and passing the total through a sigmoid.
        """
        totals = [self.base_score] * len(vectors)
        for tree in self.trees:
            for index, margin in enumerate(tree.predict_margins(vectors)):
                totals[index] += self.learning_rate * margin
        return [sigmoid(total) for total in totals]

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

    Combines the arguments shared by every model with the TF-IDF and XGBoost
    specific options. Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a TF-IDF XGBoost baseline model.")

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
    # parsers relating specifically to xgboost parameters
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of boosting rounds (trees).")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of each tree.")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split a node.")
    parser.add_argument("--max-features-split", type=int, default=32, help="Number of features considered at each split.")
    parser.add_argument("--learning-rate", type=float, default=0.3, help="Shrinkage applied to each tree's contribution.")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization on leaf weights.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Minimum gain required to make a split.")

    return parser.parse_args()

def main() -> None:
    """Run the end-to-end training and evaluation pipeline for this baseline.

    Loads the dataset, splits it, vectorizes the text, trains the XGBoost model,
    evaluates it on the test split, optionally exports the metrics JSON, and prints
    a human-readable summary. Takes no arguments and returns nothing.
    """
    args = parse_args()
    # fork artifacts to tfidf_xgboost_spans in span mode; inert otherwise
    args.model_name = args.model_name + spans_suffix(args)
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)

    y_true = [record[LABEL_FIELD] for record in test_records]
    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
    model = XGBoost(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features_split=args.max_features_split,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        seed=args.seed,
    )
    build_audit = None
    class_weight = 1.0

    if args.use_spans:
        # cross-encoder fallback: (query, chunk) rendered as one document, TF-IDF
        # vectorized, then scored by the same booster with positive-chunk weighting
        queries, chunks, chunk_labels, build_audit = build_train_chunk_examples(
            train_records, args.chunk_window, args.chunk_stride, args.overlap_threshold
        )
        train_texts = [f"{query}\n{chunk}" for query, chunk in zip(queries, chunks)]
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
        "learning_rate": args.learning_rate,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "base_score": model.base_score,
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

    print("TF-IDF XGBoost Baseline")
    print("-----------------------")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"max_features: {args.max_features}")
    print(f"min_df: {args.min_df}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"min_samples_split: {args.min_samples_split}")
    print(f"max_features_split: {args.max_features_split}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"reg_lambda: {args.reg_lambda}")
    print(f"gamma: {args.gamma}")
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
