import random
from pathlib import Path
from typing import Any

from source.utils.data import LABEL_FIELD, load_records, split_records
from source.utils.text import POSITIVE_LABEL, record_to_text


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-0.6B"


def require_torch_and_transformers(caller: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"{caller} requires working torch and transformers installations. "
            "Install or repair them with: pip install torch transformers"
        ) from exc

    return torch, nn, DataLoader, Dataset, AutoModelForCausalLM, AutoTokenizer


def add_llm_training_args(parser: Any, model_name: str) -> Any:
    parser.add_argument(
        "--model-name",
        default=model_name,
        help=f"Model name used for artifact export. Defaults to {model_name}.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_QWEN_MODEL,
        help=f"Hugging Face model id for the frozen Qwen backbone. Defaults to {DEFAULT_QWEN_MODEL}.",
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
        help="Dtype for the Qwen backbone.",
    )
    return parser


def validate_llm_args(args: Any) -> None:
    if not 0 < args.val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if args.max_length <= 0:
        raise ValueError("max_length must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device_arg)


def select_dtype(dtype_arg: str, device: Any, torch: Any) -> Any:
    if dtype_arg == "float32" or device.type == "cpu":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float16


def split_llm_records(args: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_records(args.data)
    train_records, test_records = split_records(records, args.test_size, args.seed)
    model_train_records, val_records = split_records(train_records, args.val_size, args.seed)
    return model_train_records, val_records, test_records


def prepare_tokenizer(base_model: str, AutoTokenizer: Any) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(base_model: str, dtype: Any, device: Any, AutoModelForCausalLM: Any) -> Any:
    backbone = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    backbone.to(device)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    return backbone


def freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def make_dataset_class(torch: Any, Dataset: Any) -> Any:
    class HallucinationDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.texts = [record_to_text(record) for record in records]
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
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
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
    dataset_class = make_dataset_class(torch, Dataset)
    collate_fn = make_collate_fn(tokenizer, args.max_length, torch)
    train_loader = DataLoader(
        dataset_class(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
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
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / token_counts


def forward_logits(backbone: Any, head: Any, batch: dict[str, Any], device: Any) -> Any:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    pooled = mean_pool(outputs.hidden_states[-1], attention_mask)
    pooled = pooled.to(next(head.parameters()).dtype)
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
    backbone.eval()
    head.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            logits = forward_logits(backbone, head, batch, device)
            loss = loss_fn(logits.float(), labels.float())

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
    backbone.eval()
    head.eval()
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch in dataloader:
            logits = forward_logits(backbone, head, batch, device)
            scores = torch.sigmoid(logits.float()).detach().cpu().tolist()
            y_score.extend(scores)
            y_pred.extend(POSITIVE_LABEL if score >= 0.5 else "no" for score in scores)

    return y_pred, y_score


def copy_state_dict_to_cpu(module: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def output_dir_for(args: Any) -> Path:
    return args.artifacts_dir / args.model_name
