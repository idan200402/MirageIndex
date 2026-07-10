import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# shared LLM training utility function and constant imports
from source.utils.LLM_train import (
    DEFAULT_MODERNBERT_MODEL,
    add_learning_rate_grid_arg,
    add_llm_training_args,
    compute_pos_weight,
    count_parameters,
    evaluate_document_metrics,
    evaluate_loss,
    load_backbone,
    make_dataloaders,
    make_span_dataloaders,
    output_dir_for,
    parse_learning_rates,
    predict,
    prepare_tokenizer,
    require_torch_and_encoder_transformers,
    select_device,
    select_dtype,
    select_span_operating_point,
    set_seed,
    split_llm_records,
    train_frozen_head_grid_search,
    train_frozen_head_grid_search_spans,
    validate_llm_args,
)

# utility function and constant imports
from source.utils.data import LABEL_FIELD, print_label_distribution
from source.utils.general import add_common_parsing, spans_suffix, span_config_payload
from source.utils.text import POSITIVE_LABEL, TEXT_FIELDS
from source.utils.training_metrics import classification_metrics, export_metrics_json, project_relative_path


# file specific constants
MODEL_NAME = "encoder_head"
# default grid of learning rates swept during the frozen-head grid search
DEFAULT_LEARNING_RATES = "1e-3,1e-4,1e-5"


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the LLM training options for a
    frozen ModernBERT encoder, swapping the single learning rate for a learning rate
    grid. Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a binary classification head on frozen ModernBERT encoder hidden states."
    )
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # this model always uses a fixed seed and exports metrics by default
    parser.set_defaults(seed=42, export_metrics=True)
    # parsers shared by every LLM based training script, pointed at the ModernBERT family
    parser = add_llm_training_args(
        parser,
        MODEL_NAME,
        default_base_model=DEFAULT_MODERNBERT_MODEL,
        model_family="ModernBERT",
        include_learning_rate=False,
    )
    parser.set_defaults(epochs=8)
    # a grid of learning rates replaces the single learning rate argument
    parser = add_learning_rate_grid_arg(parser, DEFAULT_LEARNING_RATES)
    return parser.parse_args()


def validate_encoder_args(args: argparse.Namespace) -> None:
    """Validate the parsed arguments for the encoder head training run.

    args: the parsed argparse.Namespace to check.\\
    Runs the shared LLM argument checks and additionally requires at least one
    learning rate in the grid. Returns nothing and raises ValueError on invalid input.
    """
    validate_llm_args(args)
    # the grid search needs at least one learning rate to try
    if not parse_learning_rates(args.learning_rates):
        raise ValueError("learning_rates must contain at least one learning rate")


def main() -> None:
    """Run the end-to-end training and evaluation pipeline for the frozen encoder head.

    Loads and splits the dataset, tokenizes it, and runs a learning rate grid search
    that trains a linear head on top of the frozen ModernBERT encoder, keeping the head
    with the lowest validation loss. Evaluates the best head on the held-out test split,
    optionally exports the metrics JSON and head weights, and prints a summary.\\
    Takes no arguments and returns nothing.
    """
    args = parse_args()
    validate_encoder_args(args)
    # fork artifacts to encoder_head_spans in span mode; inert (suffix "") otherwise
    args.model_name = args.model_name + spans_suffix(args)
    # torch and encoder transformers are imported lazily so missing dependencies fail clearly
    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_torch_and_encoder_transformers(MODEL_NAME)
    set_seed(args.seed, torch)
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)
    # the learning rate grid to sweep during the search
    learning_rates = parse_learning_rates(args.learning_rates)

    # split the raw records into train, validation and test partitions
    encoder_train_records, val_records, test_records = split_llm_records(args)

    tokenizer = prepare_tokenizer(args.base_model, AutoTokenizer)

    output_dir = output_dir_for(args)
    if args.export_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_head_path = output_dir / "best_encoder_head.pt"

    def make_backbone() -> object:
        """Load and return a fresh frozen ModernBERT encoder backbone.

        Takes no arguments. Returns a newly loaded backbone placed on the selected
        device with the selected dtype, so the grid search can start each learning
        rate from a clean model.
        """
        return load_backbone(args.base_model, dtype, device, AutoModel)

    # the held-out document labels are the A/B comparison target in both modes
    y_true = [record[LABEL_FIELD] for record in test_records]
    build_audit = None

    if args.use_spans:
        # SPAN MODE: train the frozen-encoder head on (query, chunk) pairs and select by
        # the document-level aggregation, mirroring the baseline grid search structure.
        train_loader, val_bundle, test_bundle, train_labels, build_audit = make_span_dataloaders(
            encoder_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )
        pos_weight = compute_pos_weight(train_labels, torch)
        search_result = train_frozen_head_grid_search_spans(
            model_factory=make_backbone,
            nn=nn,
            torch=torch,
            train_loader=train_loader,
            val_bundle=val_bundle,
            args=args,
            device=device,
            learning_rates=learning_rates,
            pos_weight=pos_weight,
            checkpoint_path=best_head_path,
            checkpoint_metadata={
                "base_model": args.base_model,
                "positive_label": POSITIVE_LABEL,
                "text_fields": list(TEXT_FIELDS),
                "max_length": args.max_length,
                "chunk_window": args.chunk_window,
                "chunk_stride": args.chunk_stride,
                "overlap_threshold": args.overlap_threshold,
                "aggregation": args.aggregation,
                "top_k": args.top_k,
            },
        )
    else:
        train_loader, val_loader, test_loader = make_dataloaders(
            encoder_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )
        # sweep the learning rate grid and keep the head with the lowest validation loss
        search_result = train_frozen_head_grid_search(
            model_factory=make_backbone,
            nn=nn,
            torch=torch,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            device=device,
            learning_rates=learning_rates,
            checkpoint_path=best_head_path,
            checkpoint_metadata={
                "base_model": args.base_model,
                "positive_label": POSITIVE_LABEL,
                "text_fields": list(TEXT_FIELDS),
                "max_length": args.max_length,
            },
        )

    # unpack the best backbone, head and bookkeeping produced by the grid search
    backbone = search_result["backbone"]
    head = search_result["head"]
    best_val_loss = search_result["best_val_loss"]
    best_learning_rate = search_result["best_learning_rate"]
    best_epoch = search_result["best_epoch"]
    # fall back to the backbone config when the search did not report a hidden size
    hidden_size = search_result["hidden_size"] or backbone.config.hidden_size
    training_history = search_result["training_history"]

    # evaluate the best head on the held-out test split
    operating_point = None
    if args.use_spans:
        # pick the aggregation (if 'auto') and threshold (if --target-precision) on validation
        operating_point = select_span_operating_point(backbone, head, val_bundle, args, device, torch)
        # aggregate chunk scores to document predictions; document_bce stands in for test_loss
        eval_result = evaluate_document_metrics(
            backbone,
            head,
            test_bundle,
            args,
            device,
            torch,
            aggregation=operating_point["aggregation"],
            response_threshold=operating_point["threshold"],
        )
        y_pred = eval_result["y_pred"]
        y_score = eval_result["response_scores"]
        test_loss = eval_result["document_bce"]
    else:
        y_pred, y_score = predict(backbone, head, test_loader, device, torch)
        loss_fn = nn.BCEWithLogitsLoss()
        test_loss = evaluate_loss(backbone, head, test_loader, loss_fn, device, torch)

    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    # gather the run configuration, trained parameters and metrics into one exportable payload
    metrics_payload = {
        "model_name": args.model_name,
        "base_model": args.base_model,
        "data_path": project_relative_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "train_examples": len(encoder_train_records),
        "val_examples": len(val_records),
        "test_examples": len(test_records),
        "best_val_loss": best_val_loss,
        "best_learning_rate": best_learning_rate,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "training_history": training_history,
        "artifacts": {
            "best_head_weights": project_relative_path(best_head_path) if args.export_metrics else None,
        },
        "trained_parameters": {
            "frozen_encoder": args.base_model,
            "trainable_head": "torch.nn.Linear(hidden_size, 1)",
            "hidden_size": hidden_size,
            "head_parameters": count_parameters(head),
            "encoder_parameters_trainable": count_parameters(backbone, trainable_only=True),
            "positive_label": POSITIVE_LABEL,
            "text_fields": list(TEXT_FIELDS),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rates": list(learning_rates),
            "best_learning_rate": best_learning_rate,
            "best_epoch": best_epoch,
            "weight_decay": args.weight_decay,
            "torch_dtype": str(dtype),
            "device": str(device),
        },
        "metrics": metrics,
    }
    if args.use_spans:
        # span-mode only additions; the baseline payload stays byte-identical
        metrics_payload["span_config"] = span_config_payload(args)
        metrics_payload["span_coverage"] = build_audit
        metrics_payload["operating_point"] = {
            "requested_aggregation": args.aggregation,
            "selected_aggregation": operating_point["aggregation"],
            "aggregation_pr_auc": operating_point["aggregation_pr_auc"],
            "target_precision": args.target_precision,
            "tuned_threshold": operating_point["tuned_threshold"],
            "selected_threshold": operating_point["threshold"],
        }

    # only write the metrics JSON to disk when exporting is enabled
    metrics_path = None
    if args.export_metrics:
        metrics_path = export_metrics_json(
            model_name=args.model_name,
            metrics=metrics_payload,
            artifacts_dir=args.artifacts_dir,
        )

    # print a human-readable summary of the run and its evaluation metrics
    print("ModernBERT Frozen Encoder + Trainable Head")
    print("------------------------------------------")
    print(f"base_model: {args.base_model}")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"val_size: {args.val_size}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(encoder_train_records)}")
    print(f"val_examples: {len(val_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"learning_rates: {', '.join(f'{learning_rate:g}' for learning_rate in learning_rates)}")
    print(f"best_learning_rate: {best_learning_rate:g}" if best_learning_rate is not None else "best_learning_rate: undefined")
    print(f"best_epoch: {best_epoch}" if best_epoch is not None else "best_epoch: undefined")
    print(f"best_val_loss: {best_val_loss:.4f}")
    print(f"test_loss: {test_loss:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f} ({correct}/{len(test_records)})")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "pr_auc: undefined")
    print(f"roc_auc: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "roc_auc: undefined")
    if metrics_path is not None:
        print(f"metrics_path: {metrics_path}")
        print(f"best_head_weights: {best_head_path}")

    print_label_distribution("Train Label Distribution", encoder_train_records)
    print_label_distribution("Validation Label Distribution", val_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
