import csv
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


MODEL_NAME = "openai-community/gpt2"

RESULT_COLUMNS = [
    "part",
    "experiment_name",
    "mode",
    "rank",
    "alpha",
    "target_modules",
    "lr",
    "total_params",
    "trainable_params",
    "trainable_percent",
    "train_time_seconds",
    "tokens_per_second",
    "peak_memory_mb",
    "dev_ppl",
    "test_ppl",
    "partA_best_ppl",
    "improves_over_partA",
    "checkpoint_path",
    "notes",
]


def load_gpt2_tokenizer():
    try:
        from transformers import GPT2TokenizerFast
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("LM/partB requires transformers for GPT2TokenizerFast.") from exc
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gpt2_lm_model():
    try:
        from transformers import GPT2LMHeadModel
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("LM/partB requires transformers for GPT2LMHeadModel.") from exc
    return GPT2LMHeadModel.from_pretrained(MODEL_NAME)


def read_ptb_token_ids(path: Path, tokenizer) -> list[int]:
    token_ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
            token_ids.append(tokenizer.eos_token_id)
    return token_ids


def chunk_token_ids(token_ids: list[int], block_size: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = block_size + 1
    chunks = []
    masks = []
    for start in range(0, len(token_ids), block_size):
        chunk = token_ids[start : start + chunk_size]
        if len(chunk) < 2:
            continue
        mask = [1] * len(chunk)
        if len(chunk) < chunk_size:
            pad_len = chunk_size - len(chunk)
            chunk = chunk + [pad_id] * pad_len
            mask = mask + [0] * pad_len
        chunks.append(chunk)
        masks.append(mask)
    return torch.tensor(chunks, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


class PTBGPT2Dataset(Dataset):
    def __init__(self, token_matrix: torch.Tensor, attention_matrix: torch.Tensor) -> None:
        self.tokenized = token_matrix
        self.attention = attention_matrix
        self.input_ids = self.tokenized[:, :-1].contiguous()
        self.attention_mask = self.attention[:, :-1].contiguous()
        labels = self.tokenized[:, 1:].contiguous()
        label_mask = self.attention[:, 1:].contiguous()
        self.labels = labels.masked_fill(label_mask == 0, -100)
        self.non_pad_tokens = int((self.labels != -100).sum().item())

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        labels = self.labels[index]
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": labels,
            "num_tokens": torch.tensor((labels != -100).sum().item(), dtype=torch.long),
        }


def build_datasets(project_dir: Path, tokenizer, block_size: int) -> dict[str, PTBGPT2Dataset]:
    data_dir = project_dir / "dataset" / "PennTreeBank"
    datasets = {}
    for split, filename in {
        "train": "ptb.train.txt",
        "dev": "ptb.valid.txt",
        "test": "ptb.test.txt",
    }.items():
        token_ids = read_ptb_token_ids(data_dir / filename, tokenizer)
        token_matrix, attention_matrix = chunk_token_ids(token_ids, block_size, tokenizer.pad_token_id)
        datasets[split] = PTBGPT2Dataset(token_matrix, attention_matrix)
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


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


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
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = forward_logits(model, input_ids, attention_mask)
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
        "batch_size": 8,
        "block_size": 128,
        "epochs": 3,
        "lr": 5e-4,
        "weight_decay": 0.0,
        "rank": 4,
        "alpha": 8,
        "target_modules": "qkv",
        "lora_dropout": 0.0,
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
            "smoke_lora_qkv_r2_alpha4",
            {
                "batch_size": 2,
                "block_size": 64,
                "epochs": 1,
                "rank": 2,
                "alpha": 4,
                "max_train_batches": 2,
                "max_eval_batches": 2,
            },
            "smoke",
            "1-2 batch LoRA injection, forward, loss, checkpoint and CSV verification.",
        )
    ]


def core_experiments() -> list[dict]:
    return [
        with_overrides("rank1_alpha2_qkv", {"rank": 1, "alpha": 2}, "core", "Mandatory rank/alpha sweep."),
        with_overrides("rank2_alpha4_qkv", {"rank": 2, "alpha": 4}, "core", "Mandatory rank/alpha sweep."),
        with_overrides("rank4_alpha8_qkv", {"rank": 4, "alpha": 8}, "core", "Mandatory rank/alpha sweep."),
        with_overrides("rank8_alpha16_qkv", {"rank": 8, "alpha": 16}, "core", "Mandatory rank/alpha sweep."),
    ]


def extra_experiments() -> list[dict]:
    return [
        with_overrides("extra_rank16_alpha32_qkv", {"rank": 16, "alpha": 32}, "full", "Optional larger rank."),
        with_overrides("extra_q_only_r4_alpha8", {"target_modules": "q", "rank": 4, "alpha": 8}, "full", "LoRA target ablation: Q only."),
        with_overrides("extra_k_only_r4_alpha8", {"target_modules": "k", "rank": 4, "alpha": 8}, "full", "LoRA target ablation: K only."),
        with_overrides("extra_v_only_r4_alpha8", {"target_modules": "v", "rank": 4, "alpha": 8}, "full", "LoRA target ablation: V only."),
        with_overrides(
            "extra_qkv_dropout_r4_alpha8",
            {"target_modules": "qkv", "rank": 4, "alpha": 8, "lora_dropout": 0.05},
            "full",
            "Position-agnostic gated-style regularization via LoRA dropout.",
        ),
    ]


def experiments_for_mode(mode: str) -> tuple[list[dict], list[dict]]:
    if mode == "smoke":
        return smoke_experiments(), []
    if mode == "core":
        return core_experiments(), []
    if mode == "full":
        return core_experiments(), extra_experiments()
    if mode == "extras":
        return [], extra_experiments()
    raise ValueError(f"Unknown mode={mode}")


def read_part_a_best(project_dir: Path) -> dict:
    path = project_dir.parent / "partA" / "results" / "results_partA.csv"
    if not path.exists():
        return {"dev_ppl": math.nan, "test_ppl": math.nan, "source": ""}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    valid_rows = []
    for row in rows:
        try:
            valid_rows.append((float(row["dev_ppl"]), float(row.get("test_ppl", "nan")), row))
        except (KeyError, ValueError):
            continue
    if not valid_rows:
        return {"dev_ppl": math.nan, "test_ppl": math.nan, "source": str(path)}
    dev_ppl, test_ppl, row = min(valid_rows, key=lambda item: item[0])
    return {
        "dev_ppl": dev_ppl,
        "test_ppl": test_ppl,
        "source": str(path),
        "experiment_name": row.get("experiment_name", ""),
    }
