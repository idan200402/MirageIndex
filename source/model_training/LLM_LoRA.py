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
    require_torch_and_transformers,
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
MODEL_NAME = "LLM_LoRA"
# attention projection modules wrapped with LoRA adapters by default
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj"
# default learning rate grid swept during training
DEFAULT_LEARNING_RATES = "1e-5"


def parse_args() -> argparse.Namespace:
    """Build the command-line argument parser and return the parsed arguments.

    Combines the arguments shared by every model with the LLM training options and the
    LoRA specific hyperparameters (rank, alpha, dropout, target modules, learning rate
    grid). Returns the populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary classification head plus LoRA adapters on Qwen attention "
            "projection modules."
        )
    )
    # parsers that are general to all models
    parser = add_common_parsing(parser)
    # this model always uses a fixed seed and exports metrics by default
    parser.set_defaults(seed=42, export_metrics=True)
    # parsers shared by every LLM based training script
    parser = add_llm_training_args(parser, MODEL_NAME, include_learning_rate=False)
    parser.set_defaults(epochs=15)
    # parsers relating specifically to the LoRA adapters
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA adapter rank.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA adapter scaling alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout before LoRA adapters.")
    parser.add_argument(
        "--lora-target-modules",
        default=DEFAULT_LORA_TARGET_MODULES,
        help="Comma-separated attention projection module names to wrap with LoRA adapters.",
    )
    # a grid of learning rates replaces the single learning rate argument
    parser = add_learning_rate_grid_arg(parser, DEFAULT_LEARNING_RATES)
    return parser.parse_args()


def validate_lora_args(args: argparse.Namespace) -> None:
    """Validate the parsed arguments for the LoRA training run.

    args: the parsed argparse.Namespace to check.\\
    Runs the shared LLM argument checks and additionally requires valid LoRA
    hyperparameters, at least one target module and at least one learning rate.
    Returns nothing and raises ValueError on invalid input.
    """
    validate_llm_args(args)
    if args.lora_r <= 0:
        raise ValueError("lora_r must be greater than 0")
    if args.lora_alpha <= 0:
        raise ValueError("lora_alpha must be greater than 0")
    # dropout is a probability so it must sit in the half-open interval [0, 1)
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not parse_target_modules(args.lora_target_modules):
        raise ValueError("lora_target_modules must contain at least one module name")
    if not parse_learning_rates(args.learning_rates):
        raise ValueError("learning_rates must contain at least one learning rate")


def parse_target_modules(value: str) -> tuple[str, ...]:
    """Split a comma-separated module list into a clean tuple of module names.

    value: the raw comma-separated string of target module names.\\
    Returns a tuple of the trimmed, non-empty module names in their original order.
    """
    # trim each entry and drop any empty pieces left by stray commas
    return tuple(module_name.strip() for module_name in value.split(",") if module_name.strip())


def make_lora_linear_class(nn: Any, torch: Any) -> Any:
    """Build and return a LoRALinear class bound to the given torch and nn modules.

    nn / torch: the lazily imported torch.nn and torch modules.\\
    Returns a LoRALinear class that wraps a frozen linear layer with a trainable
    low-rank adapter. The class is created inside this function so it can close over
    the imported modules without importing torch at module load time.
    """
    class LoRALinear(nn.Module):
        def __init__(self, base_layer: Any, rank: int, alpha: float, dropout: float) -> None:
            super().__init__()
            self.base_layer = base_layer
            self.rank = rank
            self.alpha = alpha
            # the low-rank update is scaled by alpha / rank before being added back
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout)

            # the wrapped layer stays frozen so only the adapter is trained
            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

            # lora_a projects the input down to the low-rank space
            self.lora_a = nn.Parameter(
                torch.empty(
                    rank,
                    base_layer.in_features,
                    device=base_layer.weight.device,
                    dtype=torch.float32,
                )
            )
            # lora_b projects back up to the output space and starts at zero
            self.lora_b = nn.Parameter(
                torch.zeros(
                    base_layer.out_features,
                    rank,
                    device=base_layer.weight.device,
                    dtype=torch.float32,
                )
            )
            # lora_b starts at zero so the adapter is a no-op until lora_a is initialized
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        def forward(self, inputs: Any) -> Any:
            # the frozen base layer produces the original output
            base_output = self.base_layer(inputs)
            # the low-rank branch runs in float32 through dropout, down then up projection
            lora_input = self.dropout(inputs).to(self.lora_a.dtype)
            lora_output = lora_input.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scaling
            # add the scaled adapter output back in the base layer's dtype
            return base_output + lora_output.to(base_output.dtype)

    return LoRALinear


def add_lora_to_attention_modules(
    backbone: Any,
    nn: Any,
    torch: Any,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Wrap every targeted attention projection layer in the backbone with a LoRA adapter.

    backbone: the frozen model whose attention projections are adapted in place.
    target_modules: names of the child linear layers to wrap (for example q_proj, v_proj).
    rank / alpha / dropout: LoRA hyperparameters passed to each adapter.\\
    Returns the list of fully qualified module names that were replaced, and raises
    ValueError when none of the requested targets are found.
    """
    LoRALinear = make_lora_linear_class(nn, torch)
    replaced_modules = []

    # walk every module and swap in an adapter for each targeted linear child
    for module_name, module in list(backbone.named_modules()):
        for child_name, child in list(module.named_children()):
            # only wrap children whose name is targeted and that are linear layers
            if child_name not in target_modules or not isinstance(child, nn.Linear):
                continue
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            # record the fully qualified name of the replaced module
            replaced_name = f"{module_name}.{child_name}" if module_name else child_name
            replaced_modules.append(replaced_name)

    # a run with no replacements means the target names never matched, so fail loudly
    if not replaced_modules:
        raise ValueError(
            "No target attention modules were found. "
            f"Requested targets: {', '.join(target_modules)}"
        )

    return replaced_modules


def trainable_state_dict(module: Any) -> dict[str, Any]:
    """Return a CPU copy of only the trainable tensors in a module's state dict.

    module: the module whose parameters are inspected.\\
    Returns a state dict containing just the entries whose parameters require gradients,
    detached and cloned onto the CPU so they can be stored without holding device memory.
    """
    # collect the names of parameters that are actually being trained
    trainable_parameter_names = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    # keep only those tensors, moved to the CPU as independent copies
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if name in trainable_parameter_names
    }


def load_partial_state_dict(module: Any, partial_state_dict: dict[str, Any]) -> None:
    """Load a subset of parameters into a module without disturbing the rest.

    module: the module to update in place.
    partial_state_dict: a mapping of parameter names to values, typically the trainable
    subset produced by trainable_state_dict. Returns nothing.
    """
    # merge the partial values over the current state, then load the combined dict
    state_dict = module.state_dict()
    state_dict.update(partial_state_dict)
    module.load_state_dict(state_dict)


def trainable_parameters(backbone: Any, head: Any) -> list[Any]:
    """Collect every parameter from the backbone and head that requires gradients.

    backbone / head: the two modules whose parameters are combined.\\
    Returns a flat list of the trainable parameters, ready to hand to an optimizer.
    """
    return [
        parameter
        for parameter in list(backbone.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]


def main() -> None:
    """Run the end-to-end training and evaluation pipeline for the LoRA adapters plus head.

    Loads and splits the dataset, tokenizes it, and for each learning rate reloads a frozen
    Qwen backbone, wraps its attention projections with LoRA adapters, and trains the adapters
    together with a linear head while tracking the best validation loss. Evaluates the best
    configuration on the held-out test split, optionally exports the metrics JSON and weights,
    and prints a summary.\\
    Takes no arguments and returns nothing.
    """
    args = parse_args()
    validate_lora_args(args)
    # fork artifacts to LLM_LoRA_spans in span mode; inert (suffix "") otherwise
    args.model_name = args.model_name + spans_suffix(args)
    # torch and transformers are imported lazily so missing dependencies fail clearly
    torch, nn, DataLoader, Dataset, AutoModelForCausalLM, AutoTokenizer = require_torch_and_transformers(MODEL_NAME)
    set_seed(args.seed, torch)
    device = select_device(args.device, torch)
    dtype = select_dtype(args.torch_dtype, device, torch)
    # the attention modules to adapt and the learning rate grid to sweep
    target_modules = parse_target_modules(args.lora_target_modules)
    learning_rates = parse_learning_rates(args.learning_rates)

    # split the raw records into train, validation and test partitions
    lora_train_records, val_records, test_records = split_llm_records(args)

    tokenizer = prepare_tokenizer(args.base_model, AutoTokenizer)

    output_dir = output_dir_for(args)
    if args.export_metrics:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_lora_path = output_dir / "best_lora_head.pt"

    # the held-out document labels are the A/B comparison target in both modes
    y_true = [record[LABEL_FIELD] for record in test_records]
    build_audit = None

    # span mode swaps the document dataloaders for chunk loaders/bundles and adds a
    # pos_weight to the loss; the grid loop below is otherwise structurally identical
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
        pos_weight = compute_pos_weight(train_labels, torch)
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

    # track the best adapter and head configuration by validation loss across the grid
    best_val_loss = float("inf")
    best_head_state = None
    best_lora_state = None
    best_learning_rate = None
    training_history = []
    hidden_size = None
    replaced_modules = []
    backbone = None
    head = None

    for learning_rate in learning_rates:
        # reset the seed so each learning rate starts from identical initial state
        set_seed(args.seed, torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # reload a fresh frozen backbone and wrap its attention projections with adapters
        backbone = load_backbone(args.base_model, dtype, device, AutoModelForCausalLM)
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

        # a single linear head maps the pooled hidden state to one logit; span mode can
        # regularize it with --dropout, while the baseline stays a plain Linear
        hidden_size = backbone.config.hidden_size
        head = build_span_head(nn, hidden_size, args.dropout if args.use_spans else 0.0).to(device)
        # the optimizer only sees the LoRA adapter and head parameters, not the frozen weights
        optimizer = torch.optim.AdamW(
            trainable_parameters(backbone, head),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

        # early stopping (span mode only) tracks val improvement per learning rate
        run_best_val = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            # train_backbone=True lets gradients flow into the LoRA adapters inside the backbone
            train_loss = train_one_epoch(
                backbone,
                head,
                train_loader,
                optimizer,
                loss_fn,
                device,
                train_backbone=True,
            )
            # span mode selects on the document-level aggregation (document_bce); the
            # baseline selects on the plain chunk-free validation loss
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

            # keep the head and adapter weights that reached the lowest validation loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_learning_rate = learning_rate
                best_head_state = trainable_state_dict(head)
                best_lora_state = trainable_state_dict(backbone)
                if args.export_metrics:
                    # persist the best head and adapter weights with the metadata needed to reload them
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
                        # record the chunk config so the checkpoint is self-describing
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

            # early stopping on the validation metric; span mode only so the baseline
            # keeps running every epoch (byte-identical)
            if val_loss < run_best_val:
                run_best_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.use_spans and args.patience and epochs_without_improvement >= args.patience:
                    print(f"lr {learning_rate:g}: early stopping at epoch {epoch}")
                    break

    # restore the best in-memory head and adapter weights seen during the grid search
    if best_head_state is not None:
        load_partial_state_dict(head, best_head_state)
    if best_lora_state is not None:
        load_partial_state_dict(backbone, best_lora_state)
    # prefer the exported checkpoint when metrics were exported to disk
    if args.export_metrics and best_lora_path.exists():
        checkpoint = torch.load(best_lora_path, map_location=device)
        head.load_state_dict(checkpoint["head_state_dict"])
        load_partial_state_dict(backbone, checkpoint["lora_state_dict"])

    # evaluate the restored configuration on the held-out test split
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
        test_loss = evaluate_loss(backbone, head, test_loader, loss_fn, device, torch)

    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score, positive_label=POSITIVE_LABEL)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    # count the adapter and head parameters that were actually trained
    trainable_backbone_parameters = count_parameters(backbone, trainable_only=True)
    trainable_head_parameters = count_parameters(head, trainable_only=True)
    # gather the run configuration, trained parameters and metrics into one exportable payload
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
        "test_loss": test_loss,
        "training_history": training_history,
        "artifacts": {
            "best_lora_head_weights": project_relative_path(best_lora_path) if args.export_metrics else None,
        },
        "trained_parameters": {
            "frozen_backbone": args.base_model,
            "trainable_head": "torch.nn.Linear(hidden_size, 1)",
            "trainable_attention_adapters": "LoRA adapters on attention projection modules",
            "hidden_size": hidden_size,
            "head_parameters": trainable_head_parameters,
            "backbone_parameters_trainable": trainable_backbone_parameters,
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
    print("Qwen LoRA Attention Adapters + Trainable Head")
    print("---------------------------------------------")
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
        print(f"best_lora_head_weights: {best_lora_path}")

    print_label_distribution("Train Label Distribution", lora_train_records)
    print_label_distribution("Validation Label Distribution", val_records)
    print_label_distribution("Test Label Distribution", test_records)


if __name__ == "__main__":
    main()
