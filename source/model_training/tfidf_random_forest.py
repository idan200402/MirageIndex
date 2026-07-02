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