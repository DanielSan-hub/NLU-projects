import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False


def get_device(requested: str, allow_cpu: bool = False) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        if allow_cpu:
            print("WARNING: CUDA requested but unavailable; falling back to CPU because --allow-cpu was passed.")
            return torch.device("cpu")
        raise RuntimeError("CUDA was requested but is not available. Pass --allow-cpu to fall back to CPU.")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def append_result_csv(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    requested_fieldnames = list(fieldnames) if fieldnames else list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            existing_rows = list(reader)
        merged_fieldnames = list(header)
        for key in requested_fieldnames:
            if key not in merged_fieldnames:
                merged_fieldnames.append(key)
        if merged_fieldnames != header:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
        requested_fieldnames = merged_fieldnames
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=requested_fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def create_run_dir(results_dir: Path, mode: str, seed: int, resume: bool) -> Path:
    latest_file = results_dir / "latest_run.txt"
    if resume:
        if not latest_file.exists():
            raise FileNotFoundError(f"--resume was passed, but {latest_file} does not exist.")
        run_dir = Path(latest_file.read_text(encoding="utf-8").strip())
        if not run_dir.is_absolute():
            run_dir = results_dir / run_dir
        return run_dir

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{timestamp}_{mode}_seed{seed}"
    run_dir = results_dir / base
    suffix = 1
    while run_dir.exists():
        run_dir = results_dir / f"{base}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    latest_file.write_text(run_dir.name, encoding="utf-8")
    return run_dir


def get_tensorboard_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - depends on optional package
        print(f"WARNING: TensorBoard logging requested but unavailable: {exc}")
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def write_summary(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
