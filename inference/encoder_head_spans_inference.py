import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from source.utils.LLM_train import (  # noqa: E402
    build_span_head,
    load_backbone,
    make_span_collate_fn,
    make_span_dataset_class,
    predict_chunk_scores,
    prepare_tokenizer,
    require_torch_and_encoder_transformers,
    select_device,
    select_dtype,
)
from source.utils.general import aggregate_document_score, document_label  # noqa: E402
from source.utils.text import (  # noqa: E402
    POSITIVE_LABEL,
    QUERY_FIELD,
    RESPONSE_FIELD,
    build_chunk_examples,
)


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "encoder_head_spans"
DEFAULT_CHECKPOINT_PATH = DEFAULT_ARTIFACT_DIR / "best_encoder_head.pt"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACT_DIR / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run document-level inference with the trained encoder_head span model."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-json",
        type=Path,
        help=(
            "JSON or JSONL file of records. Each record must contain user_query and "
            "chatgpt_response fields."
        ),
    )
    input_group.add_argument(
        "--response",
        help="Single response text to score. Must be used with --query.",
    )
    parser.add_argument("--query", help="Single user query for --response inference.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write inference results as JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to best_encoder_head.pt. Defaults to {DEFAULT_CHECKPOINT_PATH}.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=(
            "Path to encoder_head_spans metrics.json. Used for the selected span "
            f"operating point. Defaults to {DEFAULT_METRICS_PATH}."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Chunk scoring batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to use. Defaults to auto.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Backbone dtype. Defaults to auto.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("max", "mean_topk", "noisy_or"),
        default=None,
        help="Override the saved selected aggregation.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override saved top_k.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the saved selected document threshold.",
    )
    parser.add_argument(
        "--include-chunks",
        action="store_true",
        help="Include per-chunk scores and character spans in the output.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            records.append(item)
        return records

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    raise ValueError("Expected a JSON object, JSON list of objects, or JSONL file")


def load_records_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_json is not None:
        return load_json_records(args.input_json)
    if args.query is None:
        raise ValueError("--query is required when using --response")
    return [{QUERY_FIELD: args.query, RESPONSE_FIELD: args.response}]


def require_text_fields(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records, start=1):
        if QUERY_FIELD not in record:
            raise ValueError(f"Record {index} is missing {QUERY_FIELD!r}")
        if RESPONSE_FIELD not in record:
            raise ValueError(f"Record {index} is missing {RESPONSE_FIELD!r}")


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"metrics.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def make_head_from_checkpoint(checkpoint: dict[str, Any], nn: Any, device: Any) -> Any:
    hidden_size = int(checkpoint["hidden_size"])
    state_dict = checkpoint["head_state_dict"]
    # build_span_head returns either Linear (weight/bias) or Sequential(0/1.*).
    uses_sequential_head = any(key.startswith("1.") for key in state_dict)
    head = build_span_head(nn, hidden_size, dropout=0.0 if uses_sequential_head else 0.0)
    if uses_sequential_head:
        head = nn.Sequential(nn.Dropout(0.0), nn.Linear(hidden_size, 1))
    head.to(device)
    head.load_state_dict(state_dict)
    head.eval()
    return head


def resolve_runtime_config(args: argparse.Namespace, checkpoint: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    span_config = metrics.get("span_config", {})
    operating_point = metrics.get("operating_point", {})
    return {
        "base_model": checkpoint.get("base_model") or metrics.get("base_model"),
        "max_length": int(checkpoint.get("max_length") or metrics["trained_parameters"]["max_length"]),
        "chunk_window": int(checkpoint.get("chunk_window") or span_config.get("chunk_window", 40)),
        "chunk_stride": int(checkpoint.get("chunk_stride") or span_config.get("chunk_stride", 20)),
        "aggregation": args.aggregation or operating_point.get("selected_aggregation") or "max",
        "top_k": int(args.top_k if args.top_k is not None else span_config.get("top_k", 3)),
        "threshold": float(
            args.threshold
            if args.threshold is not None
            else operating_point.get("selected_threshold", span_config.get("response_threshold", 0.5))
        ),
        "batch_size": int(args.batch_size or metrics.get("trained_parameters", {}).get("batch_size", 8)),
    }


def score_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    backbone: Any,
    head: Any,
    config: dict[str, Any],
    torch: Any,
    DataLoader: Any,
    Dataset: Any,
    device: Any,
) -> list[dict[str, Any]]:
    queries, chunks, chunk_spans, doc_index, n_docs = build_chunk_examples(
        records,
        config["chunk_window"],
        config["chunk_stride"],
    )

    if chunks:
        dataset_class = make_span_dataset_class(torch, Dataset)
        collate_fn = make_span_collate_fn(tokenizer, config["max_length"], torch)
        chunk_loader = DataLoader(
            dataset_class(queries, chunks, [0.0] * len(chunks)),
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        chunk_scores = predict_chunk_scores(backbone, head, chunk_loader, device, torch)
    else:
        chunk_scores = []

    grouped_scores: list[list[float]] = [[] for _ in range(n_docs)]
    grouped_spans: list[list[tuple[int, int]]] = [[] for _ in range(n_docs)]
    grouped_chunks: list[list[str]] = [[] for _ in range(n_docs)]
    for score, span, chunk, index in zip(chunk_scores, chunk_spans, chunks, doc_index):
        grouped_scores[index].append(score)
        grouped_spans[index].append(span)
        grouped_chunks[index].append(chunk)

    results = []
    for index, record in enumerate(records):
        response_score = aggregate_document_score(
            grouped_scores[index],
            config["aggregation"],
            config["top_k"],
            chunk_spans=grouped_spans[index],
        )
        result = {
            "index": index,
            "prediction": document_label(response_score, config["threshold"], POSITIVE_LABEL),
            "hallucination_score": response_score,
            "threshold": config["threshold"],
            "aggregation": config["aggregation"],
            "top_k": config["top_k"],
            "chunk_count": len(grouped_scores[index]),
        }
        if "ID" in record:
            result["ID"] = record["ID"]
        if config.get("include_chunks"):
            result["chunks"] = [
                {
                    "span": [start, end],
                    "score": score,
                    "text": chunk,
                }
                for (start, end), score, chunk in zip(
                    grouped_spans[index],
                    grouped_scores[index],
                    grouped_chunks[index],
                )
            ]
        results.append(result)
    return results


def main() -> None:
    args = parse_args()
    records = load_records_from_args(args)
    require_text_fields(records)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    metrics = load_metrics(args.metrics)
    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_torch_and_encoder_transformers(
        "encoder_head_spans_inference"
    )
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = resolve_runtime_config(args, checkpoint, metrics)
    config["include_chunks"] = args.include_chunks

    tokenizer = prepare_tokenizer(config["base_model"], AutoTokenizer)
    backbone = load_backbone(config["base_model"], dtype, device, AutoModel)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = make_head_from_checkpoint(checkpoint, nn, device)

    results = score_records(
        records=records,
        tokenizer=tokenizer,
        backbone=backbone,
        head=head,
        config=config,
        torch=torch,
        DataLoader=DataLoader,
        Dataset=Dataset,
        device=device,
    )
    payload = {
        "model_name": "encoder_head_spans",
        "base_model": config["base_model"],
        "checkpoint": str(args.checkpoint),
        "metrics": str(args.metrics),
        "chunk_window": config["chunk_window"],
        "chunk_stride": config["chunk_stride"],
        "max_length": config["max_length"],
        "results": results,
    }

    output = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
        print(f"output_json: {args.output_json}")
    else:
        print(output)


if __name__ == "__main__":
    main()
