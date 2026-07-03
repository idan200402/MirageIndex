import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

        # creating the root of the tree
        self.root: DecisionTree._Node|None = None

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
        

    def _grow_tree(self, indices:list[int], vectors: list[dict[int, float]], labels: list[int], depth:int)-> DecisionTree._Node:
        n = len(vectors)
        # the fraction of samples with a positive label
        p = labels.count(1)/n

        # stopping conditions for the recursion
        if (depth >= self.max_depth or n < self.min_samples_split):
            return DecisionTree._Node(value=p)
        if (p == 0.0 or p == 1.0):
            return DecisionTree._Node(value=p)
        if self._best_split(indices, vectors, labels, "[NEED TO PLACE THE FEATURE SUBSET]") is None:
            return DecisionTree._Node(value=p)
        
        # drawing a random subset of features
        subset_size = min(self.max_features_split, self.feature_count)
        feature_subset = self._rng.sample(range(self.feature_count), k=subset_size)


    class _Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value: float=None):
            # internal node has values for feature_index, threshold, left and right
            # lead node has a value (probability)
            # node is a leaf iff value is not None
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

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

if __name__=="__main__":
    main()