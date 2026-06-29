import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_FIELD = "hallucination"
SPAN_FIELD = "hallucination_spans"
ID_FIELD = "ID"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "general_data.json"


def load_records(path: Path) -> list[dict[str, Any]]:
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


def word_count(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    return len(value.split())


def summarize_feature(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record.get(field) for record in records]
    missing = sum(value is None for value in values)
    non_empty = sum(value not in (None, "", [], {}) for value in values)
    types = Counter(type(value).__name__ for value in values)

    string_values = [value for value in values if isinstance(value, str)]
    char_lengths = [len(value) for value in string_values]
    word_lengths = [word_count(value) for value in string_values]

    return {
        "missing": missing,
        "non_empty": non_empty,
        "types": types,
        "min_chars": min(char_lengths) if char_lengths else None,
        "avg_chars": sum(char_lengths) / len(char_lengths) if char_lengths else None,
        "max_chars": max(char_lengths) if char_lengths else None,
        "avg_words": sum(word_lengths) / len(word_lengths) if word_lengths else None,
    }


def print_counter(title: str, counter: Counter[Any], total: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for label, count in counter.most_common():
        percentage = (count / total * 100) if total else 0
        print(f"{label}: {count} ({percentage:.2f}%)")


def print_full_examples(records: list[dict[str, Any]], label: str, limit: int) -> None:
    matching_records = [record for record in records if record.get(LABEL_FIELD) == label]
    print(f"\nFull {label!r} Examples (first {min(limit, len(matching_records))})")
    print("-" * 28)

    for index, record in enumerate(matching_records[:limit], start=1):
        spans = record.get(SPAN_FIELD) or []
        print(f"\nExample {index}")
        print(f"ID: {record.get(ID_FIELD, '<missing>')}")
        print(f"label: {record.get(LABEL_FIELD, '<missing>')}")
        print("user_query:")
        print(record.get("user_query", ""))
        print("chatgpt_response:")
        print(record.get("chatgpt_response", ""))
        print(f"hallucination_spans_count: {len(spans)}")
        if spans:
            print("hallucination_spans:")
            for span_index, span in enumerate(spans, start=1):
                print(f"[{span_index}] {span}")


def print_stats(records: list[dict[str, Any]], examples: int) -> None:
    fields = sorted({field for record in records for field in record})
    feature_fields = [field for field in fields if field not in {ID_FIELD, LABEL_FIELD, SPAN_FIELD}]

    print("Dataset")
    print("-------")
    print(f"examples: {len(records)}")
    print(f"fields: {', '.join(fields)}")
    print(f"features: {', '.join(feature_fields)}")
    print(f"label field: {LABEL_FIELD}")

    print("\nFeature Statistics")
    print("------------------")
    for field in feature_fields:
        summary = summarize_feature(records, field)
        type_summary = ", ".join(f"{name}={count}" for name, count in summary["types"].most_common())
        print(f"{field}:")
        print(f"  types: {type_summary}")
        print(f"  missing: {summary['missing']}")
        print(f"  non_empty: {summary['non_empty']}")
        if summary["avg_chars"] is not None:
            print(
                "  chars: "
                f"min={summary['min_chars']}, "
                f"avg={summary['avg_chars']:.2f}, "
                f"max={summary['max_chars']}"
            )
            print(f"  words_avg: {summary['avg_words']:.2f}")

    label_counts = Counter(record.get(LABEL_FIELD, "<missing>") for record in records)
    print_counter("Label Distribution", label_counts, len(records))

    span_counts = Counter(len(record.get(SPAN_FIELD) or []) for record in records)
    records_with_spans = sum(1 for record in records if record.get(SPAN_FIELD))
    total_spans = sum(len(record.get(SPAN_FIELD) or []) for record in records)

    print("\nHallucination Span Summary")
    print("--------------------------")
    print(f"examples_with_spans: {records_with_spans}")
    print(f"total_spans: {total_spans}")
    print_counter("Span Count Distribution", span_counts, len(records))

    print_full_examples(records, "yes", examples)
    print_full_examples(records, "no", examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print shallow statistics for general_data.json.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset. Defaults to {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of full examples to print for each label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.data)
    print_stats(records, max(args.examples, 0))


if __name__ == "__main__":
    main()
