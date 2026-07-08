# standard library imports
import math
import random
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable

# utility imports
from source.utils.data import LABEL_FIELD, load_records, split_records
from source.utils.text import (
    POSITIVE_LABEL,
    build_chunk_examples,
    build_train_chunk_examples,
    record_to_text,
)
from source.utils.general import aggregate_document_score, document_label, group_scores_by_doc


# a validation/test bundle for span-mode neural evaluation: a fixed-order chunk loader
# plus the metadata needed to group chunk scores back to their parent documents
SpanBundle = namedtuple("SpanBundle", ["chunk_loader", "chunk_spans", "doc_index", "n_docs", "doc_labels"])


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MODERNBERT_MODEL = "answerdotai/ModernBERT-large"


def require_torch_and_transformers(caller: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import torch and the causal-LM transformers pieces, failing loudly if missing.

    caller: name of the calling module, used to make the error message specific.\\
    Returns the (torch, nn, DataLoader, Dataset, AutoModelForCausalLM, AutoTokenizer)
    tuple so heavy dependencies are only imported when a model actually needs them.
    Raises ImportError with install guidance when torch or transformers cannot load.
    """
    # torch and transformers are optional heavyweight deps, so import them lazily
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        # surface a single actionable message instead of a raw import traceback
        raise ImportError(
            f"{caller} requires working torch and transformers installations. "
            "Install or repair them with: pip install torch transformers"
        ) from exc

    return torch, nn, DataLoader, Dataset, AutoModelForCausalLM, AutoTokenizer


def require_torch_and_encoder_transformers(caller: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import torch and the encoder transformers pieces, failing loudly if missing.

    caller: name of the calling module, used to make the error message specific.\\
    Returns the (torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer) tuple for
    encoder backbones. Raises ImportError with install guidance when torch or
    transformers cannot load.
    """
    # same lazy import as above but pulls the encoder AutoModel instead of the causal LM
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer
    except (ImportError, OSError) as exc:
        # surface a single actionable message instead of a raw import traceback
        raise ImportError(
            f"{caller} requires working torch and transformers installations. "
            "Install or repair them with: pip install torch transformers"
        ) from exc

    return torch, nn, DataLoader, Dataset, AutoModel, AutoTokenizer


def add_llm_training_args(
    parser: Any,
    model_name: str,
    default_base_model: str = DEFAULT_QWEN_MODEL,
    model_family: str = "Qwen",
    include_learning_rate: bool = True,
) -> Any:
    """Register the training arguments shared by the LLM-based models onto a parser.

    parser: the argparse parser to extend. model_name: default artifact name.
    default_base_model / model_family: backbone id and label shown in help text.
    include_learning_rate: whether to add a single --learning-rate option.\\
    Returns the same parser with the common LLM options attached.
    """
    parser.add_argument(
        "--model-name",
        default=model_name,
        help=f"Model name used for artifact export. Defaults to {model_name}.",
    )
    parser.add_argument(
        "--base-model",
        default=default_base_model,
        help=f"Hugging Face model id for the frozen {model_family} backbone. Defaults to {default_base_model}.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the train split to reserve for validation.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenized input length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training/evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Maximum number of training epochs.")
    # grid-search scripts omit this and add their own --learning-rates instead
    if include_learning_rate:
        parser.add_argument("--learning-rate", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
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
        help=f"Dtype for the {model_family} backbone.",
    )
    return parser


def add_learning_rate_grid_arg(parser: Any, default_learning_rates: str) -> Any:
    """Add the --learning-rates option used by grid-search training scripts.

    parser: the argparse parser to extend. default_learning_rates: comma-separated
    default string shown in the help text.\\
    Returns the same parser with the option attached.
    """
    parser.add_argument(
        "--learning-rates",
        default=default_learning_rates,
        help=(
            "Comma-separated learning rates to grid search. "
            f"Defaults to {default_learning_rates}."
        ),
    )
    return parser


def parse_learning_rates(value: str) -> tuple[float, ...]:
    """Parse a comma-separated string of learning rates into a tuple of floats.

    value: the raw "0.001,0.0005" style string from the command line.\\
    Returns the parsed learning rates as a tuple. Raises ValueError if any parsed
    rate is not greater than 0.
    """
    # split on commas and drop empty fragments left by trailing or doubled commas
    learning_rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("learning_rates must all be greater than 0")
    return learning_rates


def validate_llm_args(args: Any) -> None:
    """Check that the parsed LLM training arguments hold sane values.

    args: the argparse namespace to validate.\\
    Returns nothing. Raises ValueError on the first argument that is out of range
    (val_size, max_length, batch_size, epochs, learning_rate, or weight_decay).
    """
    if not 0 < args.val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if args.max_length <= 0:
        raise ValueError("max_length must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    # learning_rate is absent on grid-search runs, so only validate it when present
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")


def set_seed(seed: int, torch: Any) -> None:
    """Seed the Python and torch RNGs so a run is reproducible.

    seed: the integer seed to apply. torch: the injected torch module.\\
    Returns nothing. Also seeds every CUDA device when CUDA is available.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # cover the GPU generators too when training on CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str, torch: Any) -> Any:
    """Resolve the requested device string into a torch.device.

    device_arg: one of "auto", "cpu", or "cuda". torch: the injected torch module.\\
    Returns a torch.device, preferring CUDA when available under "auto". Raises
    ValueError if "cuda" is requested but no CUDA device is present.
    """
    # auto falls back to cpu whenever no gpu is available
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # an explicit cuda request must not silently degrade to cpu
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device_arg)


def select_dtype(dtype_arg: str, device: Any, torch: Any) -> Any:
    """Resolve the requested dtype string into a torch dtype for the backbone.

    dtype_arg: one of "auto", "float32", "float16", or "bfloat16". device: the
    target device. torch: the injected torch module.\\
    Returns a torch dtype, forcing float32 on CPU and defaulting to float16 under
    "auto" on other devices.
    """
    # half precision is unreliable on cpu, so cpu always runs in float32
    if dtype_arg == "float32" or device.type == "cpu":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto on a gpu defaults to float16
    return torch.float16


def split_llm_records(args: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the dataset and carve it into train, validation, and test splits.

    args: namespace holding the data path, test_size, val_size, and seed.\\
    Returns (train_records, val_records, test_records). The validation split is
    drawn from the train split so the test split never leaks into tuning.
    """
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)
    # carve validation out of the train split so the test split stays untouched
    model_train_records, val_records = split_records(train_records, args.val_size, args.seed)
    return model_train_records, val_records, test_records


def prepare_tokenizer(base_model: str, AutoTokenizer: Any) -> Any:
    """Load the tokenizer for a base model and guarantee it has a pad token.

    base_model: Hugging Face model id. AutoTokenizer: the injected loader class.\\
    Returns the ready-to-use tokenizer, falling back to the eos token for padding
    when the model defines no pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # many causal LMs ship without a pad token, so reuse eos for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(base_model: str, dtype: Any, device: Any, AutoModelForCausalLM: Any) -> Any:
    """Load a causal-LM backbone onto a device and disable its KV cache.

    base_model: Hugging Face model id. dtype / device: how and where to place it.
    AutoModelForCausalLM: the injected loader class.\\
    Returns the loaded backbone. Caching is turned off because training only needs
    the hidden states, not incremental generation.
    """
    backbone = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    backbone.to(device)
    # the kv cache only helps autoregressive generation, which we never do here
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    return backbone


def freeze_module(module: Any) -> None:
    """Freeze every parameter of a module so it is not updated during training.

    module: the torch module to freeze.\\
    Returns nothing. Sets requires_grad to False on all parameters in place.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def make_dataset_class(torch: Any, Dataset: Any) -> Any:
    """Build a torch Dataset subclass that yields text/label pairs from records.

    torch / Dataset: the injected torch module and base Dataset class.\\
    Returns a HallucinationDataset class. Each item exposes the record's flattened
    text and its binary label (1.0 for the positive label, 0.0 otherwise).
    """
    # defined inside so it can close over the injected torch and Dataset
    class HallucinationDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            # flatten each record's text fields up front so __getitem__ stays cheap
            self.texts = [record_to_text(record) for record in records]
            # map the categorical label onto the float target the loss expects
            self.labels = [
                1.0 if record[LABEL_FIELD] == POSITIVE_LABEL else 0.0
                for record in records
            ]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"text": self.texts[index], "label": torch.tensor(self.labels[index])}

    return HallucinationDataset


def make_collate_fn(tokenizer: Any, max_length: int, torch: Any) -> Any:
    """Build a collate function that tokenizes and batches records for a DataLoader.

    tokenizer: the tokenizer to apply. max_length: truncation length. torch: the
    injected torch module.\\
    Returns a collate_fn that pads and truncates a batch of items into model-ready
    tensors and attaches the stacked float labels under the "labels" key.
    """
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # tokenize the whole batch together so padding is to the batch max length
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # stack the per-item labels alongside the tokenized inputs
        encoded["labels"] = torch.stack([item["label"] for item in batch]).float()
        return encoded

    return collate_fn


def make_dataloaders(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    tokenizer: Any,
    args: Any,
    torch: Any,
    DataLoader: Any,
    Dataset: Any,
) -> tuple[Any, Any, Any]:
    """Wrap the three record splits into ready-to-iterate DataLoaders.

    train/val/test_records: the split datasets. tokenizer and args: supply the
    collate configuration and batch size. torch, DataLoader, Dataset: injected deps.\\
    Returns (train_loader, val_loader, test_loader). Only the train loader shuffles.
    """
    dataset_class = make_dataset_class(torch, Dataset)
    collate_fn = make_collate_fn(tokenizer, args.max_length, torch)
    # shuffle training data each epoch so batch composition varies
    train_loader = DataLoader(
        dataset_class(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # validation and test stay in a fixed order for stable, comparable metrics
    val_loader = DataLoader(
        dataset_class(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset_class(test_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Average a sequence of token embeddings, ignoring padding positions.

    last_hidden_state: (batch, tokens, hidden) embeddings. attention_mask: (batch,
    tokens) mask marking the real tokens.\\
    Returns one (batch, hidden) pooled vector per example. The token count is
    clamped to at least 1 to avoid dividing by zero on fully padded rows.
    """
    # broadcast the mask over the hidden dimension so padding contributes nothing
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    # clamp guards against a divide-by-zero on an all-padding row
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / token_counts


def forward_logits(backbone: Any, head: Any, batch: dict[str, Any], device: Any) -> Any:
    """Run a batch through the backbone and head to produce one logit per example.

    backbone: the frozen or trainable encoder. head: the classification head.
    batch: tokenized inputs with input_ids and attention_mask. device: target device.\\
    Returns a 1-D tensor of logits, one per example.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    # ask for hidden states because we pool them rather than use the LM head
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # pool the final layer into one vector per example
    pooled = mean_pool(outputs.hidden_states[-1], attention_mask)
    # match the head's dtype in case the backbone runs in half precision
    pooled = pooled.to(next(head.parameters()).dtype)
    # squeeze drops the trailing size-1 dimension so logits are 1-D
    return head(pooled).squeeze(-1)


def train_one_epoch(
    backbone: Any,
    head: Any,
    dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: Any,
    train_backbone: bool,
) -> float:
    """Train the head (and optionally the backbone) for a single pass over the data.

    backbone / head: the model parts. dataloader: batches to train on. optimizer /
    loss_fn: the optimization pieces. device: target device. train_backbone: whether
    the backbone runs in train mode.\\
    Returns the example-weighted average training loss for the epoch.
    """
    # the backbone only enters train mode when it is actually being fine-tuned
    backbone.train(mode=train_backbone)
    head.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(backbone, head, batch, device)
        loss = loss_fn(logits.float(), labels.float())
        loss.backward()
        optimizer.step()

        # weight each batch's loss by its size so the epoch average is exact
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate_loss(
    backbone: Any,
    head: Any,
    dataloader: Any,
    loss_fn: Any,
    device: Any,
    torch: Any,
) -> float:
    """Compute the average loss over a dataloader without updating any weights.

    backbone / head: the model parts. dataloader: batches to score. loss_fn: the
    loss to average. device: target device. torch: the injected torch module.\\
    Returns the example-weighted average loss, computed under no_grad in eval mode.
    """
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total_examples = 0

    # no_grad avoids building the autograd graph during evaluation
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            logits = forward_logits(backbone, head, batch, device)
            loss = loss_fn(logits.float(), labels.float())

            # weight each batch's loss by its size so the average is exact
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


def predict(
    backbone: Any,
    head: Any,
    dataloader: Any,
    device: Any,
    torch: Any,
) -> tuple[list[str], list[float]]:
    """Run inference over a dataloader and return hard labels and positive scores.

    backbone / head: the model parts. dataloader: batches to score. device: target
    device. torch: the injected torch module.\\
    Returns (y_pred, y_score) where y_score holds sigmoid probabilities and y_pred
    holds the positive label or "no" thresholded at 0.5.
    """
    backbone.eval()
    head.eval()
    y_pred = []
    y_score = []

    # inference needs no gradients, so run it under no_grad
    with torch.no_grad():
        for batch in dataloader:
            logits = forward_logits(backbone, head, batch, device)
            # turn logits into positive-class probabilities on the cpu
            scores = torch.sigmoid(logits.float()).detach().cpu().tolist()
            y_score.extend(scores)
            # threshold at 0.5 to get a hard label per example
            y_pred.extend(POSITIVE_LABEL if score >= 0.5 else "no" for score in scores)

    return y_pred, y_score


def copy_state_dict_to_cpu(module: Any) -> dict[str, Any]:
    """Snapshot a module's parameters as detached CPU tensors.

    module: the module whose state_dict to copy.\\
    Returns a new state dict with every tensor detached, moved to CPU, and cloned so
    later training steps cannot mutate the saved copy.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    """Count the parameters in a module.

    module: the module to inspect. trainable_only: when True count only parameters
    that require gradients.\\
    Returns the total number of (optionally trainable) parameter elements.
    """
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def output_dir_for(args: Any) -> Path:
    """Build the artifact output directory for a run.

    args: namespace holding artifacts_dir and model_name.\\
    Returns the artifacts_dir / model_name path. Does not create the directory.
    """
    return args.artifacts_dir / args.model_name


def train_frozen_head_grid_search(
    model_factory: Callable[[], Any],
    nn: Any,
    torch: Any,
    train_loader: Any,
    val_loader: Any,
    args: Any,
    device: Any,
    learning_rates: tuple[float, ...],
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Grid-search learning rates while training a linear head on a frozen backbone.

    model_factory: builds a fresh backbone per learning rate. nn / torch: injected
    deps. train_loader / val_loader: training and validation batches. args: training
    config. device: target device. learning_rates: the rates to try. checkpoint_path
    and checkpoint_metadata: where and what to save for the best run.\\
    Returns a dict with the final backbone and head (restored to the best validation
    state), the best val loss and its learning rate/epoch/hidden size, and the full
    per-epoch training history. Raises ValueError if no learning rate was run.
    """
    # track the best run seen across every learning rate
    best_val_loss = float("inf")
    best_head_state = None
    best_learning_rate = None
    best_epoch = None
    best_hidden_size = None
    training_history = []
    final_backbone = None
    final_head = None
    loss_fn = nn.BCEWithLogitsLoss()

    for learning_rate in learning_rates:
        # reseed so every learning rate trains from an identical starting point
        set_seed(args.seed, torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # a fresh frozen backbone per run, only the head is trained
        backbone = model_factory()
        backbone.eval()
        freeze_module(backbone)

        # a single linear layer maps the pooled embedding to one logit
        hidden_size = backbone.config.hidden_size
        head = nn.Linear(hidden_size, 1).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs + 1):
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
            training_history.append(
                {
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            # keep the best head by validation loss across all runs and epochs
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_learning_rate = learning_rate
                best_epoch = epoch
                best_hidden_size = hidden_size
                best_head_state = copy_state_dict_to_cpu(head)
                # persist the best checkpoint to disk when exports are enabled
                if args.export_metrics:
                    torch.save(
                        {
                            "head_state_dict": head.state_dict(),
                            "hidden_size": hidden_size,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            **checkpoint_metadata,
                        },
                        checkpoint_path,
                    )

            print(
                f"lr {learning_rate:g} epoch {epoch}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

        # remember the last run's parts to return after restoring the best head
        final_backbone = backbone
        final_head = head

    if final_backbone is None or final_head is None:
        raise ValueError("No learning-rate runs were executed")
    # restore the head to the best-scoring state seen during the search
    if best_head_state is not None:
        final_head.load_state_dict(best_head_state)
    # prefer the on-disk checkpoint when it exists so the export and return value agree
    if args.export_metrics and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        final_head.load_state_dict(checkpoint["head_state_dict"])

    return {
        "backbone": final_backbone,
        "head": final_head,
        "best_val_loss": best_val_loss,
        "best_learning_rate": best_learning_rate,
        "best_epoch": best_epoch,
        "hidden_size": best_hidden_size,
        "training_history": training_history,
    }


# ---------------------------------------------------------------------------
# Neural span-mode primitives (only used when --use-spans is set). The baseline
# neural path above is untouched.
# ---------------------------------------------------------------------------

def make_span_dataset_class(torch: Any, Dataset: Any) -> Any:
    """Build a torch Dataset subclass over pre-built (query, chunk, label) triples.

    torch / Dataset: the injected torch module and base Dataset class.\\
    Returns a SpanDataset class whose items expose the query, the chunk text, and the
    float label. Eval bundles pass placeholder labels since evaluation never reads them.
    """
    class SpanDataset(Dataset):
        def __init__(self, queries: list[str], chunks: list[str], labels: list[float]) -> None:
            self.queries = queries
            self.chunks = chunks
            self.labels = labels

        def __len__(self) -> int:
            return len(self.queries)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "query": self.queries[index],
                "chunk": self.chunks[index],
                "label": torch.tensor(float(self.labels[index])),
            }

    return SpanDataset


def make_span_collate_fn(tokenizer: Any, max_length: int, torch: Any) -> Any:
    """Build a cross-encoder collate function for (query, chunk) batches.

    tokenizer: the tokenizer to apply. max_length: truncation length. torch: the
    injected torch module.\\
    Returns a collate_fn that tokenizes each pair as [CLS] query [SEP] chunk [SEP]
    (passing text and text_pair builds token_type_ids automatically) and attaches the
    stacked float labels under the "labels" key, mirroring make_collate_fn.
    """
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # text + text_pair makes the tokenizer emit the cross-encoder segment layout
        encoded = tokenizer(
            [item["query"] for item in batch],
            [item["chunk"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.stack([item["label"] for item in batch]).float()
        return encoded

    return collate_fn


def _build_span_eval_bundle(
    records: list[dict[str, Any]],
    dataset_class: Any,
    collate_fn: Any,
    args: Any,
    DataLoader: Any,
) -> SpanBundle:
    """Build a fixed-order SpanBundle for validation/test span evaluation.

    records: the split's records, in order. dataset_class / collate_fn: the span dataset
    and collate function. args: supplies chunk config and batch size. DataLoader: injected.\\
    Returns a SpanBundle with a non-shuffling chunk loader plus the chunk spans, doc
    indices, doc count, and document labels needed to aggregate back to documents.
    """
    queries, chunks, chunk_spans, doc_index, n_docs = build_chunk_examples(
        records, args.chunk_window, args.chunk_stride
    )
    # eval never reads labels, so placeholders keep the dataset shape uniform
    chunk_loader = DataLoader(
        dataset_class(queries, chunks, [0.0] * len(queries)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    doc_labels = [record[LABEL_FIELD] for record in records]
    bundle = SpanBundle(
        chunk_loader=chunk_loader,
        chunk_spans=chunk_spans,
        doc_index=doc_index,
        n_docs=n_docs,
        doc_labels=doc_labels,
    )
    # DOC-TO-CHUNK ALIGNMENT INVARIANT checked at the bundle level
    assert len(bundle.doc_labels) == bundle.n_docs
    return bundle


def make_span_dataloaders(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    tokenizer: Any,
    args: Any,
    torch: Any,
    DataLoader: Any,
    Dataset: Any,
) -> tuple[Any, SpanBundle, SpanBundle, list[int], dict[str, int]]:
    """Wrap the three record splits into span-mode training and evaluation loaders.

    train/val/test_records: the split datasets. tokenizer / args: supply the collate
    config and batch size. torch, DataLoader, Dataset: injected deps.\\
    Returns (train_loader, val_bundle, test_bundle, train_labels, build_audit). The train
    loader shuffles over span-labeled chunks; the val/test bundles keep a fixed order so
    chunk scores stay aligned to their spans and documents. train_labels and build_audit
    feed pos_weight and the metrics payload respectively.
    """
    dataset_class = make_span_dataset_class(torch, Dataset)
    collate_fn = make_span_collate_fn(tokenizer, args.max_length, torch)

    # span-aware training chunks (the only place labels are formed)
    train_queries, train_chunks, train_labels, build_audit = build_train_chunk_examples(
        train_records, args.chunk_window, args.chunk_stride, args.overlap_threshold
    )
    train_loader = DataLoader(
        dataset_class(train_queries, train_chunks, [float(label) for label in train_labels]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_bundle = _build_span_eval_bundle(val_records, dataset_class, collate_fn, args, DataLoader)
    test_bundle = _build_span_eval_bundle(test_records, dataset_class, collate_fn, args, DataLoader)
    return train_loader, val_bundle, test_bundle, train_labels, build_audit


def compute_pos_weight(train_labels: list[int], torch: Any, clamp: tuple[float, float] = (1.0, 20.0)) -> Any:
    """Compute the BCEWithLogitsLoss pos_weight for the negative-skewed chunk labels.

    train_labels: the 0/1 chunk labels. torch: the injected torch module. clamp: bounds
    on the returned ratio.\\
    Returns a torch tensor holding neg_count / max(pos_count, 1), clamped to [low, high].
    """
    neg_count = sum(1 for label in train_labels if label == 0)
    pos_count = sum(1 for label in train_labels if label == 1)
    ratio = neg_count / max(pos_count, 1)
    ratio = min(clamp[1], max(clamp[0], ratio))
    return torch.tensor(ratio)


def predict_chunk_scores(backbone: Any, head: Any, chunk_loader: Any, device: Any, torch: Any) -> list[float]:
    """Run chunk inference and return sigmoid positive-scores in loader order.

    backbone / head: the model parts. chunk_loader: the fixed-order chunk batches.
    device: target device. torch: the injected torch module.\\
    Returns one score per chunk, aligned 1:1 with the bundle's chunk_spans/doc_index.
    Runs under no_grad in eval mode; the loader must never shuffle or reorder.
    """
    backbone.eval()
    head.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in chunk_loader:
            logits = forward_logits(backbone, head, batch, device)
            scores.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
    return scores


def document_bce(doc_scores: list[float], doc_labels: list[str], positive_label: str) -> float:
    """Compute mean binary cross-entropy between aggregated doc scores and labels.

    doc_scores: aggregated per-document positive scores. doc_labels: document labels.
    positive_label: the label mapped to target 1.0.\\
    Returns the mean BCE, with scores clamped away from 0/1 for numerical stability.
    Provides a smooth model-selection objective when chunk labels are unavailable.
    """
    epsilon = 1e-7
    total = 0.0
    for score, label in zip(doc_scores, doc_labels):
        target = 1.0 if label == positive_label else 0.0
        clamped = min(1 - epsilon, max(epsilon, score))
        total += -(target * math.log(clamped) + (1 - target) * math.log(1 - clamped))
    return total / len(doc_scores)


def evaluate_document_metrics(backbone: Any, head: Any, bundle: SpanBundle, args: Any, device: Any, torch: Any) -> dict[str, Any]:
    """Score a span bundle at the chunk level and aggregate to document predictions.

    backbone / head: the model parts. bundle: the SpanBundle to evaluate. args: supplies
    the aggregation config. device / torch: injected deps.\\
    Returns {"response_scores", "y_pred", "document_bce"}. Shared by both validation
    (model selection by document_bce) and final test scoring (against document labels).
    """
    scores = predict_chunk_scores(backbone, head, bundle.chunk_loader, device, torch)
    grouped_scores, grouped_spans = group_scores_by_doc(
        scores, bundle.chunk_spans, bundle.doc_index, bundle.n_docs
    )
    response_scores = [
        aggregate_document_score(chunk_scores, args.aggregation, args.top_k, chunk_spans=chunk_spans)
        for chunk_scores, chunk_spans in zip(grouped_scores, grouped_spans)
    ]
    y_pred = [document_label(score, args.response_threshold, POSITIVE_LABEL) for score in response_scores]
    return {
        "response_scores": response_scores,
        "y_pred": y_pred,
        "document_bce": document_bce(response_scores, bundle.doc_labels, POSITIVE_LABEL),
    }


def train_frozen_head_grid_search_spans(
    model_factory: Callable[[], Any],
    nn: Any,
    torch: Any,
    train_loader: Any,
    val_bundle: SpanBundle,
    args: Any,
    device: Any,
    learning_rates: tuple[float, ...],
    pos_weight: Any,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Grid-search learning rates for a frozen-backbone head in span mode.

    Mirrors train_frozen_head_grid_search but (a) uses a pos_weight-weighted
    BCEWithLogitsLoss over the chunk train_loader, and (b) selects the best epoch/lr by
    the LOWEST document_bce from evaluate_document_metrics on the validation bundle
    instead of chunk validation loss. Saves the same checkpoint shape plus the chunk
    metadata supplied in checkpoint_metadata.\\
    Returns the same keys as the baseline grid search (best_val_loss holds the best
    document_bce) so callers stay uniform. Raises ValueError if no learning rate ran.
    """
    # best_val_loss tracks the best (lowest) document_bce across all runs
    best_val_loss = float("inf")
    best_head_state = None
    best_learning_rate = None
    best_epoch = None
    best_hidden_size = None
    training_history = []
    final_backbone = None
    final_head = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for learning_rate in learning_rates:
        # reseed so every learning rate trains from an identical starting point
        set_seed(args.seed, torch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # a fresh frozen backbone per run, only the head is trained
        backbone = model_factory()
        backbone.eval()
        freeze_module(backbone)

        hidden_size = backbone.config.hidden_size
        head = nn.Linear(hidden_size, 1).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                backbone,
                head,
                train_loader,
                optimizer,
                loss_fn,
                device,
                train_backbone=False,
            )
            # select on the same document-level aggregation used for the test metric
            eval_result = evaluate_document_metrics(backbone, head, val_bundle, args, device, torch)
            val_document_bce = eval_result["document_bce"]
            training_history.append(
                {
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_document_bce": val_document_bce,
                }
            )

            if val_document_bce < best_val_loss:
                best_val_loss = val_document_bce
                best_learning_rate = learning_rate
                best_epoch = epoch
                best_hidden_size = hidden_size
                best_head_state = copy_state_dict_to_cpu(head)
                if args.export_metrics:
                    torch.save(
                        {
                            "head_state_dict": head.state_dict(),
                            "hidden_size": hidden_size,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "val_document_bce": val_document_bce,
                            **checkpoint_metadata,
                        },
                        checkpoint_path,
                    )

            print(
                f"lr {learning_rate:g} epoch {epoch}: "
                f"train_loss={train_loss:.4f} val_document_bce={val_document_bce:.4f}"
            )

        final_backbone = backbone
        final_head = head

    if final_backbone is None or final_head is None:
        raise ValueError("No learning-rate runs were executed")
    if best_head_state is not None:
        final_head.load_state_dict(best_head_state)
    if args.export_metrics and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        final_head.load_state_dict(checkpoint["head_state_dict"])

    return {
        "backbone": final_backbone,
        "head": final_head,
        "best_val_loss": best_val_loss,
        "best_learning_rate": best_learning_rate,
        "best_epoch": best_epoch,
        "hidden_size": best_hidden_size,
        "training_history": training_history,
    }
