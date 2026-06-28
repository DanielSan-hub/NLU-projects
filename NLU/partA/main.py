import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import torch

from functions import (
    RESULT_COLUMNS,
    build_intent_slot_ontology,
    evaluate,
    experiments_for_mode,
    make_dataloaders,
    multitask_loss,
)
from model import GPT2NLU
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
    parser = argparse.ArgumentParser(description="Mini-Project 2A: scratch GPT2 for ATIS multitask NLU.")
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
    fieldnames = ["epoch", "train_loss", "slot_loss", "intent_loss", "dev_loss", "intent_acc_dev", "slot_f1_dev", "is_best"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_model(config: dict, metadata: dict, device: torch.device) -> GPT2NLU:
    return GPT2NLU(
        vocab_size=metadata["vocab_size"],
        n_slots=metadata["n_slots"],
        n_intents=metadata["n_intents"],
        max_length=config["max_tokens"] + 1,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config["dropout"],
        pad_id=metadata["pad_id"],
    ).to(device)


def verify_smoke_batch(model, batch, config, metadata, device) -> dict:
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    cls_index = batch["cls_index"].to(device)
    slot_labels = batch["slot_labels"].to(device)
    intent_labels = batch["intent_labels"].to(device)
    with torch.no_grad():
        slot_logits, intent_logits = model(input_ids, attention_mask, cls_index)
        total_loss, slot_loss, intent_loss = multitask_loss(
            slot_logits,
            intent_logits,
            slot_labels,
            intent_labels,
            metadata["slot_pad_id"],
        )
    assert tuple(slot_logits.shape[:2]) == tuple(input_ids.shape)
    assert slot_logits.size(-1) == metadata["n_slots"]
    assert tuple(intent_logits.shape) == (input_ids.size(0), metadata["n_intents"])
    assert torch.isfinite(total_loss)
    assert torch.isfinite(slot_loss)
    assert torch.isfinite(intent_loss)
    assert torch.all(cls_index == attention_mask.sum(dim=1).to(device) - 1), "CLS must be the final valid token."
    return {
        "input_shape": tuple(input_ids.shape),
        "slot_logits_shape": tuple(slot_logits.shape),
        "intent_logits_shape": tuple(intent_logits.shape),
        "total_loss": float(total_loss.item()),
        "slot_loss": float(slot_loss.item()),
        "intent_loss": float(intent_loss.item()),
    }


def train_one_epoch(model, loader, optimizer, device, config, metadata, amp_state) -> dict:
    model.train()
    total_loss = 0.0
    total_slot_loss = 0.0
    total_intent_loss = 0.0
    total_examples = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        cls_index = batch["cls_index"].to(device, non_blocking=True)
        slot_labels = batch["slot_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=amp_state["enabled"])
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            slot_logits, intent_logits = model(input_ids, attention_mask, cls_index)
            loss, slot_loss, intent_loss = multitask_loss(
                slot_logits,
                intent_logits,
                slot_labels,
                intent_labels,
                metadata["slot_pad_id"],
            )

        if not torch.isfinite(loss):
            if amp_state["enabled"]:
                print("WARNING: non-finite loss under AMP; disabling AMP and retrying this batch in fp32.")
                amp_state["enabled"] = False
                amp_state["scaler"] = None
                optimizer.zero_grad(set_to_none=True)
                slot_logits, intent_logits = model(input_ids, attention_mask, cls_index)
                loss, slot_loss, intent_loss = multitask_loss(
                    slot_logits,
                    intent_logits,
                    slot_labels,
                    intent_labels,
                    metadata["slot_pad_id"],
                )
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

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_slot_loss += slot_loss.item() * batch_size
        total_intent_loss += intent_loss.item() * batch_size
        total_examples += batch_size
        if config["max_train_batches"] and step >= config["max_train_batches"]:
            break

    return {
        "loss": total_loss / max(total_examples, 1),
        "slot_loss": total_slot_loss / max(total_examples, 1),
        "intent_loss": total_intent_loss / max(total_examples, 1),
    }


def train_experiment(
    config: dict,
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    result_csv: Path,
    best_score_so_far: float,
) -> dict:
    set_seed(args.seed)
    exp_dir = run_dir / config["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.update(
        {
            "part": "NLU/partA",
            "seed": args.seed,
            "requested_device": args.device,
            "actual_device": device.type,
            "amp_requested": args.amp,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "grad_clip": 1.0,
        }
    )

    train_loader, dev_loader, test_loader, metadata = make_dataloaders(PROJECT_DIR, config, args)
    config.update({k: v for k, v in metadata.items() if not k.startswith("id2")})
    save_json(exp_dir / "config.json", config)
    save_json(exp_dir / "labels.json", {"id2slot": metadata["id2slot"], "id2intent": metadata["id2intent"]})
    writer = get_tensorboard_writer(exp_dir / "tensorboard", args.log_tensorboard)

    model = build_model(config, metadata, device)
    total_params, trainable_params = count_parameters(model)
    first_batch = next(iter(train_loader))
    smoke_check = verify_smoke_batch(model, first_batch, config, metadata, device)
    save_json(exp_dir / "smoke_checks.json", smoke_check)
    print(
        f"{config['experiment_name']}: input_shape={smoke_check['input_shape']} "
        f"slot_loss={smoke_check['slot_loss']:.4f} intent_loss={smoke_check['intent_loss']:.4f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    amp_state = {"enabled": amp_enabled, "scaler": scaler}

    start_epoch = 0
    best_epoch = 0
    best_score = -math.inf
    best_dev_metrics = {"intent_acc": 0.0, "slot_f1": 0.0, "loss": math.inf}
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
        best_score = checkpoint.get("best_score", -math.inf)
        best_dev_metrics = checkpoint.get("best_dev_metrics", best_dev_metrics)
        epoch_rows = checkpoint.get("epoch_rows", [])
        amp_state["enabled"] = checkpoint.get("amp_enabled", amp_state["enabled"])

    ontology = build_intent_slot_ontology(train_loader, metadata) if config["use_ontology_metrics"] else None
    reset_peak_memory(device)
    train_start = time.perf_counter()
    for epoch in range(start_epoch, config["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, metadata, amp_state)
        dev_metrics = evaluate(model, dev_loader, device, config, metadata, config["max_eval_batches"])
        score = (dev_metrics["intent_acc"] + dev_metrics["slot_f1"]) / 2
        is_best = score > best_score
        if is_best:
            best_epoch = epoch + 1
            best_score = score
            best_dev_metrics = dev_metrics
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 6),
            "slot_loss": round(train_metrics["slot_loss"], 6),
            "intent_loss": round(train_metrics["intent_loss"], 6),
            "dev_loss": round(dev_metrics["loss"], 6),
            "intent_acc_dev": round(dev_metrics["intent_acc"], 6),
            "slot_f1_dev": round(dev_metrics["slot_f1"], 6),
            "is_best": is_best,
        }
        epoch_rows.append(epoch_row)
        checkpoint = {
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_dev_metrics": best_dev_metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": amp_state["scaler"].state_dict() if amp_state["scaler"] is not None else None,
            "epoch_rows": epoch_rows,
            "amp_enabled": amp_state["enabled"],
            "config": config,
            "metadata": metadata,
        }
        save_checkpoint(last_checkpoint, checkpoint)
        if is_best:
            save_checkpoint(best_checkpoint, checkpoint)

        print(
            f"{config['experiment_name']} epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} dev_intent_acc={dev_metrics['intent_acc']:.3f} "
            f"dev_slot_f1={dev_metrics['slot_f1']:.3f}"
        )
        if writer:
            step = epoch + 1
            writer.add_scalar("train/loss", train_metrics["loss"], step)
            writer.add_scalar("train/slot_loss", train_metrics["slot_loss"], step)
            writer.add_scalar("train/intent_loss", train_metrics["intent_loss"], step)
            writer.add_scalar("dev/loss", dev_metrics["loss"], step)
            writer.add_scalar("dev/intent_acc", dev_metrics["intent_acc"], step)
            writer.add_scalar("dev/slot_f1", dev_metrics["slot_f1"], step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)

    train_time = time.perf_counter() - train_start
    write_epoch_log(exp_dir / "epoch_log.csv", epoch_rows)
    if best_checkpoint.exists():
        best_state = load_checkpoint(best_checkpoint, device)
        model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate(model, test_loader, device, config, metadata, config["max_eval_batches"], ontology=ontology)
    peak_mb = peak_memory_mb(device)
    if writer:
        step = max(best_epoch, 1)
        writer.add_scalar("train_time_seconds", train_time, step)
        writer.add_scalar("peak_memory_mb", peak_mb, step)
        writer.close()

    score = (best_dev_metrics["intent_acc"] + best_dev_metrics["slot_f1"]) / 2
    if score > best_score_so_far:
        decision = "accept"
        notes = "best dev intent/slot combined score so far"
    else:
        decision = "reject"
        notes = config["experiment_notes"] or "worse dev combined score than current best"
    if config["use_ontology_metrics"]:
        notes += (
            f"; optional illegal_slot_rate={test_metrics['illegal_slot_rate']:.4f} "
            f"frame_validity={test_metrics['frame_validity']:.4f}"
        )

    row = {
        "part": "NLU/partA",
        "experiment_name": config["experiment_name"],
        "mode": config["mode"],
        "lr": config["lr"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "num_layers": config["num_layers"],
        "ff_dim": config["ff_dim"],
        "dropout": config["dropout"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_time_seconds": round(train_time, 3),
        "peak_memory_mb": round(peak_mb, 3),
        "intent_acc_dev": round(best_dev_metrics["intent_acc"], 6),
        "slot_f1_dev": round(best_dev_metrics["slot_f1"], 6),
        "intent_acc_test": round(test_metrics["intent_acc"], 6),
        "slot_f1_test": round(test_metrics["slot_f1"], 6),
        "semantic_frame_acc_test": round(test_metrics["semantic_frame_acc"], 6),
        "checkpoint_path": relative(best_checkpoint),
        "decision": decision,
        "notes": notes,
    }
    append_result_csv(result_csv, row, fieldnames=RESULT_COLUMNS)
    write_summary(
        exp_dir / "summary.txt",
        [
            f"NLU/partA experiment: {config['experiment_name']}",
            f"Architecture: d_model={config['d_model']} n_heads={config['n_heads']} "
            f"num_layers={config['num_layers']} ff_dim={config['ff_dim']} dropout={config['dropout']}",
            f"Best epoch: {best_epoch}",
            f"Best dev intent accuracy: {best_dev_metrics['intent_acc']:.4f}",
            f"Best dev slot CoNLL F1: {best_dev_metrics['slot_f1']:.4f}",
            f"Test intent accuracy: {test_metrics['intent_acc']:.4f}",
            f"Test slot CoNLL F1: {test_metrics['slot_f1']:.4f}",
            f"Test semantic frame accuracy: {test_metrics['semantic_frame_acc']:.4f}",
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
    run_dir = create_run_dir(PROJECT_DIR / "results", args.mode, args.seed, args.resume)
    save_json(
        run_dir / "config.json",
        {
            "part": "NLU/partA",
            "mode": args.mode,
            "seed": args.seed,
            "device": device.type,
            "experiments": [exp["experiment_name"] for exp in all_exps],
        },
    )

    rows = []
    best_row = None
    best_score_so_far = -math.inf
    for exp in core_exps:
        row = train_experiment(
            exp,
            run_dir,
            args,
            device,
            PROJECT_DIR / "results" / "results_partA.csv",
            best_score_so_far,
        )
        rows.append(row)
        score = (float(row["intent_acc_dev"]) + float(row["slot_f1_dev"])) / 2
        if score > best_score_so_far:
            best_score_so_far = score
            best_row = row

    for exp in extra_exps:
        row = train_experiment(
            exp,
            run_dir,
            args,
            device,
            PROJECT_DIR / "results" / "results_partA_extra.csv",
            best_score_so_far,
        )
        rows.append(row)
        score = (float(row["intent_acc_dev"]) + float(row["slot_f1_dev"])) / 2
        if score > best_score_so_far:
            best_score_so_far = score
            best_row = row

    if best_row is None:
        best_row = max(rows, key=lambda item: (float(item["intent_acc_dev"]) + float(item["slot_f1_dev"])) / 2)
    write_summary(
        run_dir / "summary.txt",
        [
            "Mini-Project 2A: scratch GPT2 for ATIS multitask NLU",
            f"Mode/device/seed: {args.mode} / {device.type} / {args.seed}",
            f"Final best configuration: {best_row['experiment_name']}",
            f"Best dev intent accuracy: {best_row['intent_acc_dev']}",
            f"Best dev slot CoNLL F1: {best_row['slot_f1_dev']}",
            f"Test intent accuracy: {best_row['intent_acc_test']}",
            f"Test slot CoNLL F1: {best_row['slot_f1_test']}",
            f"Checkpoint: {best_row['checkpoint_path']}",
        ],
    )
    print(
        "Final best configuration: "
        f"{best_row['experiment_name']} dev_intent_acc={best_row['intent_acc_dev']} "
        f"dev_slot_f1={best_row['slot_f1_dev']} test_intent_acc={best_row['intent_acc_test']} "
        f"test_slot_f1={best_row['slot_f1_test']} checkpoint={best_row['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
