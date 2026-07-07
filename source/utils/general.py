# standard library imports
import argparse
from pathlib import Path
import math

# utility function imports
from source.utils.training_metrics import parse_bool
# utility constant imports
from source.utils.data import DEFAULT_DATA_PATH, DEFAULT_ARTIFACTS_DIR, DEFAULT_SEED, DEFAULT_TEST_SIZE

def add_common_parsing(parser: argparse.ArgumentParser ) -> argparse.ArgumentParser:
    """Attach the command-line arguments shared by every model onto a parser.

    parser: an argparse.ArgumentParser to extend in place.\\
    Adds the dataset path, seed, test-size, metric-export flag, and artifacts
    directory options, then returns the same parser so calls can be chained.
    """
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

    # parsers relating to artifact exports and test matrices
    parser.add_argument(
        "--export-metrics",
        type=parse_bool,
        default=False,
        help="True exports test metrics JSON to artifacts/model_name. False skips export.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory where exported metrics are written. Defaults to {DEFAULT_ARTIFACTS_DIR}.",
    )

    return parser

def sigmoid(value: float) -> float:
    """Map a real-valued margin to a probability in the open interval (0, 1).

    value: the raw logit or margin to squash.\\
    Returns the logistic sigmoid of value, computed in a numerically stable way
    that avoids overflow for large-magnitude inputs.
    """
    # for non-negative inputs the standard form keeps exp arguments non-positive
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    # for negative inputs use the equivalent form so exp never overflows
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)