import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from source.utils.LLM_train import (
    DEFAULT_MODERNBERT_MODEL,
    add_learning_rate_grid_arg,
    add_llm_training_args,
    count_parameters,
    evaluate_loss,
    load_backbone,
    make_dataloaders,
    output_dir_for,
    parse_learning_rates,
    predict,
    prepare_tokenizer,
    require_torch_and_encoder_transformers,
    select_device,
    select_dtype,
    set_seed,
    split_llm_records,
    train_frozen_head_grid_search,
    validate_llm_args,
)
from source.utils.data import LABEL_FIELD, print_label_distribution
from source.utils.general import add_common_parsing
from source.utils.text import POSITIVE_LABEL, TEXT_FIELDS
from source.utils.training_metrics import classification_metrics, export_metrics_json


MODEL_NAME = "encoder_head"
DEFAULT_LEARNING_RATES = "1e-3,1e-4,1e-5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary classification head on frozen ModernBERT encoder hidden states."
    )
    parser = add_common_parsing(parser)
    parser.set_defaults(seed=42, export_metrics=True)
    parser = add_llm_training_args(
        parser,
        MODEL_NAME,
        default_base_model=DEFAULT_MODERNBERT_MODEL,
        model_family="ModernBERT",
        include_learning_rate=False,
    )
    parser.set_defaults(epochs=8)
    parser = add_learning_rate_grid_arg(parser, DEFAULT_LEARNING_RATES)
    return parser.parse_args()


def validate_encoder_args(args: argparse.Namespace) -> None:
    validate_llm_args(args)
    if not parse_learning_rates(args.learning_rates):
        raise ValueError("learning_rates must contain at least one learning rate")


def main() -> None:
    args = parse_args()
    validate_encoder_args(args)
    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_torch_and_encoder_transformers(MODEL_NAME)
    set_seed(args.seed, torch)
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)
    learning_rates = parse_learning_rates(args.learning_rates)

    encoder_train_records, val_records, test_records = split_llm_records(args)

    tokenizer = prepare_tokenizer(args.base_model, AutoTokenizer)
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

    output_dir = output_dir_for(args)
    if args.export_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_head_path = output_dir / "best_encoder_head.pt"

    def make_backbone() -> object:
        return load_backbone(args.base_model, dtype, device, AutoModel)

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

    backbone = search_result["backbone"]
    head = search_result["head"]
    best_val_loss = search_result["best_val_loss"]
    best_learning_rate = search_result["best_learning_rate"]
    best_epoch = search_result["best_epoch"]
    hidden_size = search_result["hidden_size"] or backbone.config.hidden_size
    training_history = search_result["training_history"]

    y_true = [record[LABEL_FIELD] for record in test_records]
    y_pred, y_score = predict(backbone, head, test_loader, device, torch)
    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss = evaluate_loss(backbone, head, test_loader, loss_fn, device, torch)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    metrics_payload = {
        "model_name": args.model_name,
        "base_model": args.base_model,
        "data_path": str(args.data),
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
            "best_head_weights": str(best_head_path) if args.export_metrics else None,
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

    metrics_path = None
    if args.export_metrics:
        metrics_path = export_metrics_json(
            model_name=args.model_name,
            metrics=metrics_payload,
            artifacts_dir=args.artifacts_dir,
        )

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
