# imports 
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

# constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_FIELD = "hallucination"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "general_data.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a dataset file and return it as a list of records (dicts).

    Accepts three on-disk formats and normalizes them all to a list:
      - a JSON array of objects -> returned as-is
      - a single JSON object    -> wrapped in a one-element list
      - JSONL (one JSON object per line) -> parsed line by line, blank lines skipped

    The JSON array/object path is tried first; if the whole file is not valid
    JSON, it falls back to parsing line by line as JSONL.

    Raises ValueError if any line/item is not a JSON object, or if the top-level
    JSON is neither an object nor a list.
    """
    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object on line {line_number}, got {type(item).__name__}")
            records.append(item)
        return records

    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Expected every list item to be a JSON object")
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Expected JSON object, JSON list, or JSONL file, got {type(data).__name__}")

def split_records(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (train, test) using a deterministic shuffle.

    Shuffles a copy of the records with a fixed ``seed`` so the same seed always
    produces the same split (reproducible across runs and across models). The
    first ``round(len(records) * test_size)`` records become the test set
    (at least 1), and the rest become the train set.

    Args:
        records: the full dataset.
        test_size: fraction of records to use for testing, strictly between 0 and 1.
        seed: random seed controlling the shuffle.

    Raises ValueError if test_size is out of range, if there are fewer than 2
    records, or if the split would leave the train set empty.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("Need at least 2 records to split the dataset")

    shuffled_records = records[:]
    random.Random(seed).shuffle(shuffled_records)

    test_count = max(1, round(len(shuffled_records) * test_size))
    test_records = shuffled_records[:test_count]
    train_records = shuffled_records[test_count:]

    if not train_records:
        raise ValueError("Training split is empty. Use a smaller test_size.")

    return train_records, test_records

def print_label_distribution(title: str, records: list[dict[str, Any]]) -> None:
    """Print how many records fall under each label, with percentages.

    Reads the ``LABEL_FIELD`` ("hallucination") value from every record and
    prints a small table (label: count (percentage)) under an underlined title.
    Records missing the field are counted under "<missing>". Used to sanity-check
    that the train/test splits have a comparable class balance.
    """
    label_counts = Counter(record.get(LABEL_FIELD, "<missing>") for record in records)
    print(f"\n{title}")
    print("-" * len(title))
    for label, count in label_counts.most_common():
        percentage = count / len(records) * 100 if records else 0.0
        print(f"{label}: {count} ({percentage:.2f}%)")