import argparse
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# shared LLM training utility function imports
from source.utils.LLM_train import (
    DEFAULT_MODERNBERT_MODEL,
    add_learning_rate_grid_arg,
    add_llm_training_args,
    build_span_head,
    compute_pos_weight,
    count_parameters,
    evaluate_document_metrics,
    evaluate_loss,
    freeze_module,
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
    train_one_epoch,
    validate_llm_args,
)

# utility function and constant imports
from source.utils.data import LABEL_FIELD, print_label_distribution
from source.utils.general import add_common_parsing, spans_suffix, span_config_payload
from source.utils.text import POSITIVE_LABEL, TEXT_FIELDS
from source.utils.training_metrics import classification_metrics, export_metrics_json, project_relative_path


# file specific constants
MODEL_NAME = "encoder_LoRA"
# ModernBERT fuses q/k/v into Wqkv and keeps the attention output projection in Wo.
DEFAULT_LORA_TARGET_MODULES = "Wqkv,Wo"
# default learning rate grid swept during training
DEFAULT_LEARNING_RATES = "1e-5"
ATTENTION_PARENT_NAMES = {"attn", "attention", "self_attn"}


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the encoder training options and
    the LoRA hyperparameters. The default backbone is ModernBERT and the default LoRA
    targets are its attention projection modules, matching the Qwen LoRA script's scope.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary classification head plus LoRA adapters on ModernBERT "
            "attention projection modules."
        )
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
    parser.set_defaults(epochs=15)
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA adapter rank.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA adapter scaling alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout before LoRA adapters.")
    parser.add_argument(
        "--lora-target-modules",
        default=DEFAULT_LORA_TARGET_MODULES,
        help=(
            "Comma-separated linear module names to wrap with LoRA adapters. "
            "Defaults to ModernBERT's attention projections: Wqkv,Wo."
        ),
    )
    parser = add_learning_rate_grid_arg(parser, DEFAULT_LEARNING_RATES)
    return parser.parse_args()


def validate_lora_args(args: argparse.Namespace) -> None:
    """Validate the parsed arguments for the encoder LoRA training run."""
    validate_llm_args(args)
    if args.lora_r <= 0:
        raise ValueError("lora_r must be greater than 0")
    if args.lora_alpha <= 0:
        raise ValueError("lora_alpha must be greater than 0")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not parse_target_modules(args.lora_target_modules):
        raise ValueError("lora_target_modules must contain at least one module name")
    if not parse_learning_rates(args.learning_rates):
        raise ValueError("learning_rates must contain at least one learning rate")


def parse_target_modules(value: str) -> tuple[str, ...]:
    """Split a comma-separated module list into a clean tuple of module names."""
    return tuple(module_name.strip() for module_name in value.split(",") if module_name.strip())


def make_lora_linear_class(nn: Any, torch: Any) -> Any:
    """Build and return a LoRALinear class bound to the given torch and nn modules."""
    class LoRALinear(nn.Module):
        def __init__(self, base_layer: Any, rank: int, alpha: float, dropout: float) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout)

            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

            self.lora_a = nn.Parameter(
                torch.empty(
                    rank,
                    base_layer.in_features,
                    device=base_layer.weight.device,
                    dtype=torch.float32,
                )
            )
            self.lora_b = nn.Parameter(
                torch.zeros(
                    base_layer.out_features,
                    rank,
                    device=base_layer.weight.device,
                    dtype=torch.float32,
                )
            )
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, inputs: Any) -> Any:
            base_output = self.base_layer(inputs)
            lora_input = self.dropout(inputs).to(self.lora_a.dtype)
            lora_output = lora_input.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scaling
            return base_output + lora_output.to(base_output.dtype)

    return LoRALinear


def is_attention_parent(module_name: str) -> bool:
    """Return True when a module path looks like an attention block."""
    return any(part in ATTENTION_PARENT_NAMES for part in module_name.split("."))


def add_lora_to_attention_modules(
    backbone: Any,
    nn: Any,
    torch: Any,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Wrap every targeted attention projection layer in the backbone with LoRA."""
    LoRALinear = make_lora_linear_class(nn, torch)
    replaced_modules = []

    for module_name, module in list(backbone.named_modules()):
        for child_name, child in list(module.named_children()):
            if (
                child_name not in target_modules
                or not isinstance(child, nn.Linear)
                or not is_attention_parent(module_name)
            ):
                continue
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            replaced_name = f"{module_name}.{child_name}" if module_name else child_name
            replaced_modules.append(replaced_name)

    if not replaced_modules:
        raise ValueError(
            "No target attention modules were found. "
            f"Requested targets: {', '.join(target_modules)}"
        )

    return replaced_modules


def trainable_state_dict(module: Any) -> dict[str, Any]:
    """Return a CPU copy of only the trainable tensors in a module's state dict."""
    trainable_parameter_names = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if name in trainable_parameter_names
    }


def load_partial_state_dict(module: Any, partial_state_dict: dict[str, Any]) -> None:
    """Load a subset of parameters into a module without disturbing the rest."""
    state_dict = module.state_dict()
    state_dict.update(partial_state_dict)
    module.load_state_dict(state_dict)


def trainable_parameters(backbone: Any, head: Any) -> list[Any]:
    """Collect every parameter from the backbone and head that requires gradients."""
    return [
        parameter
        for parameter in list(backbone.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]


def main() -> None:
    """Run the end-to-end training and evaluation pipeline for ModernBERT LoRA."""
    args = parse_args()
    validate_lora_args(args)
    args.model_name = args.model_name + spans_suffix(args)

    torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer = require_torch_and_encoder_transformers(MODEL_NAME)
    set_seed(args.seed, torch)
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)
    target_modules = parse_target_modules(args.lora_target_modules)
    learning_rates = parse_learning_rates(args.learning_rates)

    lora_train_records, val_records, test_records = split_llm_records(args)
    tokenizer = prepare_tokenizer(args.base_model, AutoTokenizer)

    output_dir = output_dir_for(args)
    if args.export_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_lora_path = output_dir / "best_encoder_lora_head.pt"

    y_true = [record[LABEL_FIELD] for record in test_records]
    build_audit = None

    if args.use_spans:
        train_loader, val_bundle, test_bundle, train_labels, build_audit = make_span_dataloaders(
            lora_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )
        pos_weight = compute_pos_weight(train_labels, torch).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_loader, val_loader, test_loader = make_dataloaders(
            lora_train_records,
            val_records,
            test_records,
            tokenizer,
            args,
            torch,
            DataLoader,
            Dataset,
        )
        loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_head_state = None
    best_lora_state = None
    best_learning_rate = None
    best_epoch = None
    training_history = []
    hidden_size = None
    replaced_modules = []
    backbone = None
    head = None

    for learning_rate in learning_rates:
        set_seed(args.seed, torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        backbone = load_backbone(args.base_model, dtype, device, AutoModel)
        freeze_module(backbone)
        replaced_modules = add_lora_to_attention_modules(
            backbone=backbone,
            nn=nn,
            torch=torch,
            target_modules=target_modules,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        hidden_size = backbone.config.hidden_size
        head = build_span_head(nn, hidden_size, args.dropout if args.use_spans else 0.0).to(device)
        optimizer = torch.optim.AdamW(
            trainable_parameters(backbone, head),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

        run_best_val = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                backbone,
                head,
                train_loader,
                optimizer,
                loss_fn,
                device,
                train_backbone=True,
            )
            if args.use_spans:
                val_loss = evaluate_document_metrics(backbone, head, val_bundle, args, device, torch)["document_bce"]
            else:
                val_loss = evaluate_loss(backbone, head, val_loader, loss_fn, device, torch)
            training_history.append(
                {
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_learning_rate = learning_rate
                best_epoch = epoch
                best_head_state = trainable_state_dict(head)
                best_lora_state = trainable_state_dict(backbone)
                if args.export_metrics:
                    checkpoint_payload = {
                        "head_state_dict": head.state_dict(),
                        "lora_state_dict": best_lora_state,
                        "base_model": args.base_model,
                        "hidden_size": hidden_size,
                        "positive_label": POSITIVE_LABEL,
                        "text_fields": list(TEXT_FIELDS),
                        "max_length": args.max_length,
                        "learning_rate": learning_rate,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_target_modules": list(target_modules),
                        "replaced_modules": replaced_modules,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }
                    if args.use_spans:
                        checkpoint_payload.update(
                            {
                                "chunk_window": args.chunk_window,
                                "chunk_stride": args.chunk_stride,
                                "overlap_threshold": args.overlap_threshold,
                                "aggregation": args.aggregation,
                                "top_k": args.top_k,
                            }
                        )
                    torch.save(checkpoint_payload, best_lora_path)

            print(
                f"lr {learning_rate:g} epoch {epoch}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < run_best_val:
                run_best_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.use_spans and args.patience and epochs_without_improvement >= args.patience:
                    print(f"lr {learning_rate:g}: early stopping at epoch {epoch}")
                    break

    if backbone is None or head is None:
        raise ValueError("No learning-rate runs were executed")
    if best_head_state is not None:
        load_partial_state_dict(head, best_head_state)
    if best_lora_state is not None:
        load_partial_state_dict(backbone, best_lora_state)
    if args.export_metrics and best_lora_path.exists():
        checkpoint = torch.load(best_lora_path, map_location=device)
        head.load_state_dict(checkpoint["head_state_dict"])
        load_partial_state_dict(backbone, checkpoint["lora_state_dict"])

    operating_point = None
    if args.use_spans:
        operating_point = select_span_operating_point(backbone, head, val_bundle, args, device, torch)
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
        test_loss = evaluate_loss(backbone, head, test_loader, loss_fn, device, torch)

    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    trainable_backbone_parameters = count_parameters(backbone, trainable_only=True)
    trainable_head_parameters = count_parameters(head, trainable_only=True)
    metrics_payload = {
        "model_name": args.model_name,
        "base_model": args.base_model,
        "data_path": project_relative_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "train_examples": len(lora_train_records),
        "val_examples": len(val_records),
        "test_examples": len(test_records),
        "best_val_loss": best_val_loss,
        "best_learning_rate": best_learning_rate,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "training_history": training_history,
        "artifacts": {
            "best_encoder_lora_head_weights": project_relative_path(best_lora_path) if args.export_metrics else None,
        },
        "trained_parameters": {
            "frozen_encoder": args.base_model,
            "trainable_head": "torch.nn.Linear(hidden_size, 1)",
            "trainable_attention_adapters": "LoRA adapters on ModernBERT attention projection modules",
            "hidden_size": hidden_size,
            "head_parameters": trainable_head_parameters,
            "encoder_parameters_trainable": trainable_backbone_parameters,
            "total_trainable_parameters": trainable_head_parameters + trainable_backbone_parameters,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": list(target_modules),
            "replaced_modules": replaced_modules,
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
        metrics_payload["span_config"] = span_config_payload(args)
        metrics_payload["span_coverage"] = build_audit
        metrics_payload["operating_point"] = {
            "requested_aggregation": args.aggregation,
            "selected_aggregation": operating_point["aggregation"],
            "aggregation_pr_auc": operating_point["aggregation_pr_auc"],
            "target_precision": args.target_precision,
            "tuned_threshold": operating_point["tuned_threshold"],
            "objective": operating_point["objective"],
            "selected_threshold": operating_point["threshold"],
        }

    metrics_path = None
    if args.export_metrics:
        metrics_path = export_metrics_json(
            model_name=args.model_name,
            metrics=metrics_payload,
            artifacts_dir=args.artifacts_dir,
        )

    print("ModernBERT LoRA Attention Adapters + Trainable Head")
    print("---------------------------------------------------")
    print(f"base_model: {args.base_model}")
    print(f"data_path: {args.data}")
    print(f"seed: {args.seed}")
    print(f"test_size: {args.test_size}")
    print(f"val_size: {args.val_size}")
    print(f"export_metrics: {args.export_metrics}")
    print(f"train_examples: {len(lora_train_records)}")
    print(f"val_examples: {len(val_records)}")
    print(f"test_examples: {len(test_records)}")
    print(f"learning_rates: {', '.join(f'{learning_rate:g}' for learning_rate in learning_rates)}")
    print(f"best_learning_rate: {best_learning_rate:g}" if best_learning_rate is not None else "best_learning_rate: undefined")
    print(f"best_epoch: {best_epoch}" if best_epoch is not None else "best_epoch: undefined")
    print(f"lora_r: {args.lora_r}")
    print(f"lora_alpha: {args.lora_alpha}")
    print(f"lora_dropout: {args.lora_dropout}")
    print(f"lora_target_modules: {', '.join(target_modules)}")
    print(f"replaced_attention_modules: {len(replaced_modules)}")
    print(f"trainable_parameters: {trainable_head_parameters + trainable_backbone_parameters}")
    print(f"best_val_loss: {best_val_loss:.4f}")
    print(f"test_loss: {test_loss:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f} ({correct}/{len(test_records)})")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"pr_auc: {metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "pr_auc: undefined")
    print(f"roc_auc: {metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "roc_auc: undefined")
    if metrics_path is not None:
        print(f"metrics_path: {metrics_path}")
        print(f"best_encoder_lora_head_weights: {best_lora_path}")

    print_label_distribution("Train Label Distribution", lora_train_records)
    print_label_distribution("Validation Label Distribution", val_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
