import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


RESULT_COLUMNS = [
    "part",
    "experiment_name",
    "mode",
    "lr",
    "d_model",
    "n_heads",
    "num_layers",
    "ff_dim",
    "dropout",
    "weight_tying",
    "total_params",
    "trainable_params",
    "train_time_seconds",
    "tokens_per_second",
    "peak_memory_mb",
    "dev_ppl",
    "test_ppl",
    "checkpoint_path",
    "decision",
    "notes",
    "best_epoch",
    "final_train_loss",
    "best_dev_loss",
    "train_dev_gap",
    "overfitting_note",
]


def load_gpt2_tokenizer():
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "LM/partA requires transformers for GPT2TokenizerFast. "
            "Install the project requirements or run in the university environment."
        ) from exc
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def read_ptb_token_ids(path: Path, tokenizer) -> list[int]:
    token_ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
            token_ids.append(tokenizer.eos_token_id)
    return token_ids


def chunk_token_ids(token_ids: list[int], block_size: int, pad_id: int) -> torch.Tensor:
    chunk_size = block_size + 1
    chunks = []
    for start in range(0, len(token_ids), block_size):
        chunk = token_ids[start : start + chunk_size]
        if len(chunk) < 2:
            continue
        if len(chunk) < chunk_size:
            chunk = chunk + [pad_id] * (chunk_size - len(chunk))
        chunks.append(chunk)
    return torch.tensor(chunks, dtype=torch.long)


class PTBGPT2Dataset(Dataset):
    def __init__(self, token_matrix: torch.Tensor, pad_id: int) -> None:
        self.tokenized = token_matrix
        self.pad_id = pad_id
        self.input_ids = self.tokenized[:, :-1].contiguous()
        labels = self.tokenized[:, 1:].contiguous()
        self.labels = labels.masked_fill(labels == pad_id, -100)
        self.non_pad_tokens = int((self.labels != -100).sum().item())

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        labels = self.labels[index]
        return {
            "input_ids": self.input_ids[index],
            "labels": labels,
            "num_tokens": torch.tensor((labels != -100).sum().item(), dtype=torch.long),
        }


def build_datasets(project_dir: Path, tokenizer, block_size: int) -> dict[str, PTBGPT2Dataset]:
    data_dir = project_dir / "dataset" / "PennTreeBank"
    pad_id = tokenizer.pad_token_id
    datasets = {}
    for split, filename in {
        "train": "ptb.train.txt",
        "dev": "ptb.valid.txt",
        "test": "ptb.test.txt",
    }.items():
        token_ids = read_ptb_token_ids(data_dir / filename, tokenizer)
        token_matrix = chunk_token_ids(token_ids, block_size=block_size, pad_id=pad_id)
        datasets[split] = PTBGPT2Dataset(token_matrix, pad_id=pad_id)
    return datasets


def make_dataloader(dataset: Dataset, config: dict, args, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        generator=generator if shuffle else None,
    )


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def safe_ppl(loss: float) -> float:
    return float(math.exp(min(loss, 50.0)))


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(input_ids)
        loss = lm_loss(logits, labels)
        tokens = int(batch["num_tokens"].sum().item())
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if max_batches and step >= max_batches:
            break
    avg_loss = total_loss / max(total_tokens, 1)
    return {"loss": avg_loss, "ppl": safe_ppl(avg_loss), "tokens": total_tokens}


def base_config() -> dict:
    return {
        "batch_size": 32,
        "block_size": 128,
        "epochs": 10,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "ff_dim": 512,
        "dropout": 0.1,
        "weight_tying": True,
        "norm_type": "layernorm",
        "activation": "gelu",
        "lambda_x0": 0.0,
        "max_train_batches": None,
        "max_eval_batches": None,
    }


def with_overrides(name: str, overrides: dict, mode: str, notes: str = "") -> dict:
    cfg = base_config()
    cfg.update(overrides)
    cfg["experiment_name"] = name
    cfg["mode"] = mode
    cfg["experiment_notes"] = notes
    return cfg


def smoke_experiments() -> list[dict]:
    return [
        with_overrides(
            "smoke_shapes_loss_ppl_checkpoint_csv",
            {
                "batch_size": 4,
                "block_size": 32,
                "epochs": 1,
                "d_model": 64,
                "n_heads": 2,
                "num_layers": 1,
                "ff_dim": 256,
                "lr": 5e-4,
                "max_train_batches": 2,
                "max_eval_batches": 2,
            },
            mode="smoke",
            notes="1-2 batch harness verification.",
        )
    ]


def core_experiments() -> list[dict]:
    return [
        with_overrides("baseline_lr_5e-4", {"lr": 5e-4}, "core", "Baseline fixed architecture."),
        with_overrides("lr_sweep_1e-3", {"lr": 1e-3}, "core", "LR sweep, fixed architecture."),
        with_overrides("lr_sweep_3e-4", {"lr": 3e-4}, "core", "LR sweep, fixed architecture."),
        with_overrides("ablation_d_model_192", {"d_model": 192}, "core", "One-at-a-time d_model ablation."),
        with_overrides("ablation_n_heads_8", {"n_heads": 8}, "core", "One-at-a-time n_heads ablation."),
        with_overrides("ablation_num_layers_3", {"num_layers": 3}, "core", "One-at-a-time num_layers ablation."),
        with_overrides("ablation_ff_dim_768", {"ff_dim": 768}, "core", "One-at-a-time ff_dim ablation."),
        with_overrides("dropout_0_0", {"dropout": 0.0}, "core", "Required dropout ablation."),
        with_overrides(
            "dropout_0_2_weight_tying",
            {"dropout": 0.2, "weight_tying": True},
            "core",
            "Required dropout + weight tying regularization experiment.",
        ),
    ]


def extra_experiments() -> list[dict]:
    return [
        with_overrides("extra_rmsnorm", {"norm_type": "rmsnorm"}, "full", "Optional RMSNorm vs LayerNorm."),
        with_overrides("extra_relu2", {"activation": "relu2"}, "full", "Optional ReLU^2 vs GELU."),
        with_overrides(
            "extra_depth_dial_1",
            {"num_layers": 1, "d_model": 96, "n_heads": 3, "ff_dim": 384, "lr": 7e-4},
            "full",
            "nanochat-inspired depth dial depth=1.",
        ),
        with_overrides(
            "extra_depth_dial_2",
            {"num_layers": 2, "d_model": 128, "n_heads": 4, "ff_dim": 512, "lr": 5e-4},
            "full",
            "nanochat-inspired depth dial depth=2.",
        ),
        with_overrides(
            "extra_depth_dial_3",
            {"num_layers": 3, "d_model": 192, "n_heads": 6, "ff_dim": 768, "lr": 3e-4},
            "full",
            "nanochat-inspired depth dial depth=3.",
        ),
        with_overrides(
            "extra_x0_residual",
            {"lambda_x0": 0.1},
            "full",
            "Optional x0 residual injection after blocks.",
        ),
    ]


def experiments_for_mode(mode: str) -> tuple[list[dict], list[dict]]:
    if mode == "smoke":
        return smoke_experiments(), []
    if mode == "core":
        return core_experiments(), []
    if mode == "full":
        return core_experiments(), extra_experiments()
    raise ValueError(f"Unknown mode={mode}")
