import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# shared LLM training utility function imports
from source.utils.LLM_train import (
    add_llm_training_args,
    compute_pos_weight,
    copy_state_dict_to_cpu,
    count_parameters,
    evaluate_document_metrics,
    evaluate_loss,
    freeze_module,
    load_backbone,
    make_dataloaders,
    make_span_dataloaders,
    output_dir_for,
    predict,
    prepare_tokenizer,
    require_torch_and_transformers,
    select_device,
    select_dtype,
    set_seed,
    split_llm_records,
    train_frozen_head_grid_search_spans,
    train_one_epoch,
    validate_llm_args,
)

# utility function and constant imports
from source.utils.data import LABEL_FIELD, print_label_distribution
from source.utils.general import add_common_parsing, spans_suffix, span_config_payload
from source.utils.text import POSITIVE_LABEL, TEXT_FIELDS
from source.utils.training_metrics import classification_metrics, export_metrics_json, project_relative_path


# file specific constants
MODEL_NAME = "LLM_train_head"


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the LLM training options
    used to fit a head on top of a frozen Qwen backbone. Returns the populated
    argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a binary classification head on frozen Qwen hidden states."
    )
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # this model always uses a fixed seed and exports metrics by default
    parser.set_defaults(seed=42, export_metrics=True)
    # parsers shared by every LLM based training script
    parser = add_llm_training_args(parser, MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end training and evaluation pipeline for the frozen backbone head.

    Loads and splits the dataset, tokenizes it, freezes the Qwen backbone, trains a
    single linear head over several epochs while tracking the best validation loss,
    evaluates the best head on the held-out test split, optionally exports the metrics
    JSON and head weights, and prints a human-readable summary.\\
    Takes no arguments and returns nothing.
    """
    args = parse_args()
    validate_llm_args(args)
    # fork artifacts to LLM_train_head_spans in span mode; inert (suffix "") otherwise
    args.model_name = args.model_name + spans_suffix(args)
    # torch and transformers are imported lazily so the script fails clearly when they are missing
    torch, nn, DataLoader, Dataset, AutoModelForCausalLM, AutoTokenizer = require_torch_and_transformers(MODEL_NAME)
    set_seed(args.seed, torch)
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)

    # split the raw records into train, validation and test partitions
    head_train_records, val_records, test_records = split_llm_records(args)

    tokenizer = prepare_tokenizer(args.base_model, AutoTokenizer)

    output_dir = output_dir_for(args)
    if args.export_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_head_path = output_dir / "best_head.pt"

    # the held-out document labels are the A/B comparison target in both modes
    y_true = [record[LABEL_FIELD] for record in test_records]
    build_audit = None

    if args.use_spans:
        # SPAN MODE: predict per (query, chunk) pair and aggregate to a document score.
        # A single-element learning-rate tuple reuses the shared span grid search.
        train_loader, val_bundle, test_bundle, train_labels, build_audit = make_span_dataloaders(
            head_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )
        pos_weight = compute_pos_weight(train_labels, torch)

        def make_backbone() -> object:
            """Load a fresh frozen causal-LM backbone for the span grid search."""
            return load_backbone(args.base_model, dtype, device, AutoModelForCausalLM)

        search_result = train_frozen_head_grid_search_spans(
            model_factory=make_backbone,
            nn=nn,
            torch=torch,
            train_loader=train_loader,
            val_bundle=val_bundle,
            args=args,
            device=device,
            learning_rates=(args.learning_rate,),
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
        backbone = search_result["backbone"]
        head = search_result["head"]
        hidden_size = search_result["hidden_size"] or backbone.config.hidden_size
        best_val_loss = search_result["best_val_loss"]
        training_history = search_result["training_history"]

        # aggregate chunk scores to document predictions for the test metric
        eval_result = evaluate_document_metrics(backbone, head, test_bundle, args, device, torch)
        y_pred = eval_result["y_pred"]
        y_score = eval_result["response_scores"]
        # document_bce stands in for the baseline's test_loss in span mode
        test_loss = eval_result["document_bce"]
    else:
        backbone = load_backbone(args.base_model, dtype, device, AutoModelForCausalLM)
        # put the backbone in eval mode and freeze it so only the head is trained
        backbone.eval()
        freeze_module(backbone)

        # a single linear head maps the pooled hidden state to one logit
        hidden_size = backbone.config.hidden_size
        head = nn.Linear(hidden_size, 1).to(device)
        # only the head parameters are handed to the optimizer
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader, val_loader, test_loader = make_dataloaders(
            head_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )

        # track the best head by validation loss across every epoch
        best_val_loss = float("inf")
        best_head_state = None
        training_history = []

        for epoch in range(1, args.epochs + 1):
            # the backbone stays frozen so only the head is updated this epoch
            train_loss = train_one_epoch(
                backbone,
                head,
                train_loader,
                optimizer,
                loss_fn,
                device,
                train_backbone=False,
            )
            val_loss = evaluate_loss(backbone, head, val_loader, loss_fn, device, torch)
            training_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # keep the head that reached the lowest validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_head_state = copy_state_dict_to_cpu(head)
                if args.export_metrics:
                    # persist the best head weights alongside the metadata needed to reload them
                    torch.save(
                        {
                            "head_state_dict": head.state_dict(),
                            "base_model": args.base_model,
                            "hidden_size": hidden_size,
                            "positive_label": POSITIVE_LABEL,
                            "text_fields": list(TEXT_FIELDS),
                            "max_length": args.max_length,
                            "epoch": epoch,
                            "val_loss": val_loss,
                        },
                        best_head_path,
                    )

            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # restore the best in-memory head seen during training
        if best_head_state is not None:
            head.load_state_dict(best_head_state)
        # prefer the exported checkpoint when metrics were exported to disk
        if args.export_metrics and best_head_path.exists():
            checkpoint = torch.load(best_head_path, map_location=device)
            head.load_state_dict(checkpoint["head_state_dict"])

        # evaluate the restored head on the held-out test split
        y_pred, y_score = predict(backbone, head, test_loader, device, torch)
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
        "train_examples": len(head_train_records),
        "val_examples": len(val_records),
        "test_examples": len(test_records),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "training_history": training_history,
        "artifacts": {
            "best_head_weights": project_relative_path(best_head_path) if args.export_metrics else None,
        },
        "trained_parameters": {
            "frozen_backbone": args.base_model,
            "trainable_head": "torch.nn.Linear(hidden_size, 1)",
            "hidden_size": hidden_size,
            "head_parameters": count_parameters(head),
            "backbone_parameters_trainable": count_parameters(backbone, trainable_only=True),
            "positive_label": POSITIVE_LABEL,
            "text_fields": list(TEXT_FIELDS),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
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

    # only write the metrics JSON to disk when exporting is enabled
    metrics_path = None
    if args.export_metrics:
        metrics_path = export_metrics_json(
            model_name=args.model_name,
            metrics=metrics_payload,
            artifacts_dir=args.artifacts_dir,
        )

    # print a human-readable summary of the run and its evaluation metrics
    print("Qwen Frozen Backbone + Trainable Head")
    print("-------------------------------------")
    print(f"base_model: {args.base_model}")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"val_size: {args.val_size}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(head_train_records)}")
    print(f"val_examples: {len(val_records)}")
    print(f"test_examples: {len(test_records)}")
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

    print_label_distribution("Train Label Distribution", head_train_records)
    print_label_distribution("Validation Label Distribution", val_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
