import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import torch

from functions import (
    RESULT_COLUMNS,
    build_datasets,
    evaluate,
    experiments_for_mode,
    lm_loss,
    load_gpt2_tokenizer,
    make_dataloader,
)
from model import GPT2LM
from utils import (
    append_result_csv,
    count_parameters,
    create_run_dir,
    get_device,
    get_tensorboard_writer,
    load_checkpoint,
    peak_memory_mb,
    reset_peak_memory,
    save_checkpoint,
    save_json,
    set_seed,
    write_summary,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-Project 1A: scratch GPT2 language modeling on PennTreeBank.")
    parser.add_argument("--mode", choices=["smoke", "core", "full", "extras"], default="smoke")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-tensorboard", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_epoch_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "dev_loss", "dev_ppl", "train_dev_gap", "is_best"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(model, loader, optimizer, device, config, amp_state) -> dict:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=amp_state["enabled"])
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = model(input_ids)
            loss = lm_loss(logits, labels)

        if not torch.isfinite(loss):
            if amp_state["enabled"]:
                print("WARNING: non-finite loss under AMP; disabling AMP and retrying this batch in fp32.")
                amp_state["enabled"] = False
                amp_state["scaler"] = None
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = lm_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered with AMP disabled.")

        if amp_state["enabled"]:
            amp_state["scaler"].scale(loss).backward()
            amp_state["scaler"].unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            amp_state["scaler"].step(optimizer)
            amp_state["scaler"].update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        tokens = int(batch["num_tokens"].sum().item())
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if config["max_train_batches"] and step >= config["max_train_batches"]:
            break

    return {"loss": total_loss / max(total_tokens, 1), "tokens": total_tokens}


def build_model(config: dict, vocab_size: int, device: torch.device) -> GPT2LM:
    model = GPT2LM(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
        weight_tying=config["weight_tying"],
        norm_type=config["norm_type"],
        activation=config["activation"],
        lambda_x0=config["lambda_x0"],
    ).to(device)
    if config["weight_tying"]:
        assert model.lm_head.weight is model.token_embed.weight
    return model


def train_experiment(
    config: dict,
    run_dir: Path,
    datasets: dict,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    result_csv: Path,
    baseline_row: dict | None,
    best_dev_ppl_so_far: float,
    writer_enabled: bool,
) -> dict:
    set_seed(args.seed)
    exp_dir = run_dir / config["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.update(
        {
            "part": "LM/partA",
            "seed": args.seed,
            "requested_device": args.device,
            "actual_device": device.type,
            "amp_requested": args.amp,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "grad_clip": 1.0,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "tokenizer": "gpt2",
        }
    )
    save_json(exp_dir / "config.json", config)
    writer = get_tensorboard_writer(exp_dir / "tensorboard", writer_enabled)

    train_loader = make_dataloader(datasets["train"], config, args, shuffle=True)
    dev_loader = make_dataloader(datasets["dev"], config, args, shuffle=False)
    test_loader = make_dataloader(datasets["test"], config, args, shuffle=False)

    model = build_model(config, vocab_size=len(tokenizer), device=device)
    total_params, trainable_params = count_parameters(model)
    print(
        f"{config['experiment_name']}: params before tying={model.param_count_before_tying} "
        f"after tying={model.param_count_after_tying} trainable={trainable_params}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    amp_state = {"enabled": amp_enabled, "scaler": scaler}

    start_epoch = 0
    best_epoch = 0
    best_dev_loss = math.inf
    best_dev_ppl = math.inf
    final_train_loss = math.inf
    epoch_rows: list[dict] = []
    last_checkpoint = exp_dir / "last.pt"
    best_checkpoint = exp_dir / "best.pt"

    if args.resume and last_checkpoint.exists():
        checkpoint = load_checkpoint(last_checkpoint, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if amp_state["scaler"] is not None and checkpoint.get("scaler_state"):
            amp_state["scaler"].load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_dev_loss = checkpoint.get("best_dev_loss", math.inf)
        best_dev_ppl = checkpoint.get("best_dev_ppl", math.inf)
        final_train_loss = checkpoint.get("final_train_loss", math.inf)
        epoch_rows = checkpoint.get("epoch_rows", [])
        amp_state["enabled"] = checkpoint.get("amp_enabled", amp_state["enabled"])

    reset_peak_memory(device)
    train_start = time.perf_counter()
    total_train_tokens = 0

    for epoch in range(start_epoch, config["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, amp_state)
        dev_metrics = evaluate(model, dev_loader, device, config["max_eval_batches"])
        total_train_tokens += train_metrics["tokens"]
        final_train_loss = train_metrics["loss"]
        train_dev_gap = dev_metrics["loss"] - train_metrics["loss"]
        is_best = dev_metrics["ppl"] < best_dev_ppl
        if is_best:
            best_epoch = epoch + 1
            best_dev_loss = dev_metrics["loss"]
            best_dev_ppl = dev_metrics["ppl"]

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 6),
            "dev_loss": round(dev_metrics["loss"], 6),
            "dev_ppl": round(dev_metrics["ppl"], 6),
            "train_dev_gap": round(train_dev_gap, 6),
            "is_best": is_best,
        }
        epoch_rows.append(epoch_row)
        checkpoint_state = {
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_dev_loss": best_dev_loss,
            "best_dev_ppl": best_dev_ppl,
            "final_train_loss": final_train_loss,
            "epoch_rows": epoch_rows,
            "amp_enabled": amp_state["enabled"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": amp_state["scaler"].state_dict() if amp_state["scaler"] is not None else None,
            "config": config,
        }
        save_checkpoint(last_checkpoint, checkpoint_state)
        if is_best:
            save_checkpoint(best_checkpoint, checkpoint_state)

        print(
            f"{config['experiment_name']} epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} dev_loss={dev_metrics['loss']:.4f} "
            f"dev_ppl={dev_metrics['ppl']:.2f} gap={train_dev_gap:.4f}"
        )
        if writer:
            step = epoch + 1
            writer.add_scalar("train/loss", train_metrics["loss"], step)
            writer.add_scalar("dev/loss", dev_metrics["loss"], step)
            writer.add_scalar("dev/ppl", dev_metrics["ppl"], step)
            writer.add_scalar("train_dev_gap", train_dev_gap, step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)

    train_time = time.perf_counter() - train_start
    tokens_per_second = total_train_tokens / train_time if train_time > 0 else 0.0
    write_epoch_log(exp_dir / "epoch_log.csv", epoch_rows)

    if best_checkpoint.exists():
        best_state = load_checkpoint(best_checkpoint, device)
        model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate(model, test_loader, device, config["max_eval_batches"])
    peak_mb = peak_memory_mb(device)
    train_dev_gap = best_dev_loss - final_train_loss

    if writer:
        step = max(best_epoch, 1)
        writer.add_scalar("train_time_seconds", train_time, step)
        writer.add_scalar("tokens_per_second", tokens_per_second, step)
        writer.add_scalar("peak_memory_mb", peak_mb, step)
        writer.flush()
        writer.close()

    overfitting_note = "none"
    likely_overfit = False
    if baseline_row and config["experiment_name"].startswith(("ablation_d_model", "ablation_num_layers")):
        likely_overfit = (
            final_train_loss < float(baseline_row["final_train_loss"])
            and best_dev_ppl > float(baseline_row["dev_ppl"])
        )
        if likely_overfit:
            overfitting_note = "larger/deeper model improved train loss but worsened dev PPL"

    if likely_overfit:
        decision = "reject"
        notes = "likely overfitting or optimization instability on small PTB"
    elif baseline_row is None and config["experiment_name"] == "baseline_lr_5e-4":
        decision = "baseline"
        notes = "reference fixed architecture"
    elif best_dev_ppl < best_dev_ppl_so_far:
        decision = "accept"
        notes = "best dev PPL so far"
    elif best_dev_ppl < 250:
        decision = "accept"
        notes = "meets PPL target but is not the best run so far"
    else:
        decision = "reject"
        notes = config.get("experiment_notes") or "worse dev PPL than current best"

    row = {
        "part": "LM/partA",
        "experiment_name": config["experiment_name"],
        "mode": config["mode"],
        "lr": config["lr"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "num_layers": config["num_layers"],
        "ff_dim": config["ff_dim"],
        "dropout": config["dropout"],
        "weight_tying": config["weight_tying"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_seconds": round(train_time, 3),
        "tokens_per_second": round(tokens_per_second, 3),
        "peak_memory_mb": round(peak_mb, 3),
        "dev_ppl": round(best_dev_ppl, 6),
        "test_ppl": round(test_metrics["ppl"], 6),
        "checkpoint_path": relative(best_checkpoint),
        "decision": decision,
        "notes": notes,
        "best_epoch": best_epoch,
        "final_train_loss": round(final_train_loss, 6),
        "best_dev_loss": round(best_dev_loss, 6),
        "train_dev_gap": round(train_dev_gap, 6),
        "overfitting_note": overfitting_note,
    }
    append_result_csv(result_csv, row, fieldnames=RESULT_COLUMNS)
    write_summary(
        exp_dir / "summary.txt",
        [
            f"LM/partA experiment: {config['experiment_name']}",
            f"Mode/device/seed: {config['mode']} / {device.type} / {args.seed}",
            f"Architecture: d_model={config['d_model']} n_heads={config['n_heads']} "
            f"num_layers={config['num_layers']} ff_dim={config['ff_dim']} dropout={config['dropout']}",
            f"Weight tying: {config['weight_tying']}",
            f"Parameters before/after tying: {model.param_count_before_tying} / {model.param_count_after_tying}",
            f"Best epoch: {best_epoch}",
            f"Final train loss: {final_train_loss:.4f}",
            f"Best dev loss/PPL: {best_dev_loss:.4f} / {best_dev_ppl:.2f}",
            f"Test PPL from best checkpoint: {test_metrics['ppl']:.2f}",
            f"Train-dev gap: {train_dev_gap:.4f}",
            f"Decision: {decision}",
            f"Notes: {notes}",
        ],
    )
    return row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device, args.allow_cpu)

    core_exps, extra_exps = experiments_for_mode(args.mode)
    all_exps = core_exps + extra_exps
    if not all_exps:
        raise RuntimeError(f"No experiments configured for mode={args.mode}")

    run_dir = create_run_dir(PROJECT_DIR / "results", args.mode, args.seed, args.resume)
    save_json(
        run_dir / "config.json",
        {
            "part": "LM/partA",
            "mode": args.mode,
            "seed": args.seed,
            "device": device.type,
            "experiments": [exp["experiment_name"] for exp in all_exps],
        },
    )

    tokenizer = load_gpt2_tokenizer()
    max_block_size = max(exp["block_size"] for exp in all_exps)
    datasets = build_datasets(PROJECT_DIR, tokenizer, block_size=max_block_size)
    if any(exp["block_size"] != max_block_size for exp in all_exps):
        raise ValueError("All LM/partA experiments in a single run must share block_size for cached datasets.")

    baseline_row = None
    best_row = None
    best_dev_ppl_so_far = math.inf
    rows = []

    for exp in core_exps:
        result_csv = PROJECT_DIR / "results" / "results_partA.csv"
        row = train_experiment(
            exp,
            run_dir,
            datasets,
            tokenizer,
            args,
            device,
            result_csv,
            baseline_row=baseline_row,
            best_dev_ppl_so_far=best_dev_ppl_so_far,
            writer_enabled=args.log_tensorboard,
        )
        rows.append(row)
        if row["experiment_name"] == "baseline_lr_5e-4":
            baseline_row = row
        if float(row["dev_ppl"]) < best_dev_ppl_so_far:
            best_dev_ppl_so_far = float(row["dev_ppl"])
            best_row = row

    for exp in extra_exps:
        result_csv = PROJECT_DIR / "results" / "results_partA_extra.csv"
        row = train_experiment(
            exp,
            run_dir,
            datasets,
            tokenizer,
            args,
            device,
            result_csv,
            baseline_row=baseline_row,
            best_dev_ppl_so_far=best_dev_ppl_so_far,
            writer_enabled=args.log_tensorboard,
        )
        rows.append(row)
        if float(row["dev_ppl"]) < best_dev_ppl_so_far:
            best_dev_ppl_so_far = float(row["dev_ppl"])
            best_row = row

    if best_row is None:
        best_row = min(rows, key=lambda r: float(r["dev_ppl"]))

    write_summary(
        run_dir / "summary.txt",
        [
            "Mini-Project 1A: scratch GPT2 PTB language modeling",
            f"Mode/device/seed: {args.mode} / {device.type} / {args.seed}",
            f"Final best configuration: {best_row['experiment_name']}",
            f"Best dev PPL: {best_row['dev_ppl']}",
            f"Best test PPL: {best_row['test_ppl']}",
            f"Checkpoint: {best_row['checkpoint_path']}",
            "See results/results_partA.csv and per-experiment epoch_log.csv for overfitting-aware traces.",
        ],
    )
    print(
        "Final best configuration: "
        f"{best_row['experiment_name']} dev_ppl={best_row['dev_ppl']} "
        f"test_ppl={best_row['test_ppl']} checkpoint={best_row['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
