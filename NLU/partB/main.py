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
    make_dataloaders,
    model_configs_for_mode,
    multitask_loss,
    verify_alignment_and_shapes,
)
from model import load_pretrained_multitask_model, load_tokenizer
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
    parser = argparse.ArgumentParser(description="Mini-Project 2B: pretrained BERT/GPT2 multitask NLU on ATIS.")
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


def train_one_epoch(model, loader, optimizer, device, config, amp_state) -> dict:
    model.train()
    total_loss = 0.0
    total_slot_loss = 0.0
    total_intent_loss = 0.0
    total_examples = 0
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        slot_labels = batch["slot_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=amp_state["enabled"])
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            slot_logits, intent_logits = model(input_ids, attention_mask)
            loss, slot_loss, intent_loss = multitask_loss(
                slot_logits,
                intent_logits,
                slot_labels,
                intent_labels,
                config["lambda_slot"],
                config["lambda_intent"],
            )

        if not torch.isfinite(loss):
            if amp_state["enabled"]:
                print("WARNING: non-finite loss under AMP; disabling AMP and retrying this batch in fp32.")
                amp_state["enabled"] = False
                amp_state["scaler"] = None
                optimizer.zero_grad(set_to_none=True)
                slot_logits, intent_logits = model(input_ids, attention_mask)
                loss, slot_loss, intent_loss = multitask_loss(
                    slot_logits,
                    intent_logits,
                    slot_labels,
                    intent_labels,
                    config["lambda_slot"],
                    config["lambda_intent"],
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


def run_model(config: dict, run_dir: Path, args, device: torch.device, result_csv: Path) -> dict:
    set_seed(args.seed)
    exp_dir = run_dir / config["model_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.update(
        {
            "part": "NLU/partB",
            "seed": args.seed,
            "requested_device": args.device,
            "actual_device": device.type,
            "amp_requested": args.amp,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "grad_clip": 1.0,
        }
    )

    tokenizer = load_tokenizer(config["pretrained_model"])
    train_loader, dev_loader, test_loader, metadata = make_dataloaders(PROJECT_DIR, config, args, tokenizer)
    model = load_pretrained_multitask_model(
        config["pretrained_model"],
        tokenizer,
        n_slots=metadata["n_slots"],
        n_intents=metadata["n_intents"],
        pooling=config["pooling"],
        dropout=config["dropout"],
    ).to(device)
    save_json(exp_dir / "config.json", {**config, **{k: v for k, v in metadata.items() if k.startswith("n_")}})
    save_json(exp_dir / "labels.json", {"id2slot": metadata["id2slot"], "id2intent": metadata["id2intent"]})
    writer = get_tensorboard_writer(exp_dir / "tensorboard", args.log_tensorboard)

    first_batch = next(iter(train_loader))
    checks = verify_alignment_and_shapes(model, first_batch, device, config, metadata, config["model_name"])
    save_json(exp_dir / "alignment_and_shape_checks.json", checks)
    print(
        f"{config['model_name']}: input_shape={checks['input_shape']} "
        f"slot_logits={checks['slot_logits_shape']} intent_logits={checks['intent_logits_shape']} "
        f"ignored_labels={checks['ignored_slot_labels']} pooling={checks['pooling']}"
    )

    total_params, trainable_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    amp_state = {"enabled": amp_enabled, "scaler": scaler}

    start_epoch = 0
    best_epoch = 0
    best_score = -math.inf
    best_dev_metrics = {"loss": math.inf, "intent_acc": 0.0, "slot_f1": 0.0}
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

    ontology = build_intent_slot_ontology(train_loader, metadata) if config.get("use_ontology_gate") else None
    reset_peak_memory(device)
    train_start = time.perf_counter()
    for epoch in range(start_epoch, config["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, amp_state)
        dev_metrics = evaluate(model, dev_loader, device, config, metadata)
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
        }
        save_checkpoint(last_checkpoint, checkpoint)
        if is_best:
            save_checkpoint(best_checkpoint, checkpoint)

        print(
            f"{config['model_name']} epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"dev_intent_acc={dev_metrics['intent_acc']:.3f} dev_slot_f1={dev_metrics['slot_f1']:.3f}"
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
    test_metrics = evaluate(model, test_loader, device, config, metadata, ontology=ontology)
    peak_mb = peak_memory_mb(device)
    if writer:
        step = max(best_epoch, 1)
        writer.add_scalar("train_time_seconds", train_time, step)
        writer.add_scalar("peak_memory_mb", peak_mb, step)
        writer.close()

    notes = (
        f"subtoken labels use -100 for non-first subtokens/special/pad; "
        f"pooling={config['pooling']}; ignored_labels_in_smoke_batch={checks['ignored_slot_labels']}"
    )
    if config.get("use_ontology_gate"):
        notes += f"; optional illegal_slot_rate={test_metrics['illegal_slot_rate']:.4f}"
    row = {
        "part": "NLU/partB",
        "model_name": config["model_name"],
        "pretrained_model": config["pretrained_model"],
        "mode": config["mode"],
        "lr": config["lr"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "lambda_slot": config["lambda_slot"],
        "lambda_intent": config["lambda_intent"],
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
        "notes": notes,
    }
    append_result_csv(result_csv, row, fieldnames=RESULT_COLUMNS)
    write_summary(
        exp_dir / "summary.txt",
        [
            f"NLU/partB model: {config['model_name']}",
            f"Pretrained model: {config['pretrained_model']}",
            f"Pooling: {config['pooling']}",
            f"Best epoch: {best_epoch}",
            f"Best dev intent accuracy: {best_dev_metrics['intent_acc']:.4f}",
            f"Best dev slot CoNLL F1: {best_dev_metrics['slot_f1']:.4f}",
            f"Test intent accuracy: {test_metrics['intent_acc']:.4f}",
            f"Test slot CoNLL F1: {test_metrics['slot_f1']:.4f}",
            f"Test semantic frame accuracy: {test_metrics['semantic_frame_acc']:.4f}",
            f"Checkpoint: {relative(best_checkpoint)}",
        ],
    )
    return row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device, args.allow_cpu)
    core_configs, extra_configs = model_configs_for_mode(args.mode)
    all_configs = core_configs + extra_configs
    run_dir = create_run_dir(PROJECT_DIR / "results", args.mode, args.seed, args.resume)
    save_json(
        run_dir / "config.json",
        {
            "part": "NLU/partB",
            "mode": args.mode,
            "seed": args.seed,
            "device": device.type,
            "models": [cfg["model_name"] for cfg in all_configs],
        },
    )

    rows = []
    for cfg in core_configs:
        rows.append(run_model(cfg, run_dir, args, device, PROJECT_DIR / "results" / "results_partB.csv"))
    for cfg in extra_configs:
        rows.append(run_model(cfg, run_dir, args, device, PROJECT_DIR / "results" / "results_partB_extra.csv"))

    print("\nComparison table:")
    print("model_name\tintent_acc_dev\tslot_f1_dev\tintent_acc_test\tslot_f1_test")
    for row in rows:
        print(
            f"{row['model_name']}\t{row['intent_acc_dev']}\t{row['slot_f1_dev']}\t"
            f"{row['intent_acc_test']}\t{row['slot_f1_test']}"
        )
    best = max(rows, key=lambda item: (float(item["intent_acc_dev"]) + float(item["slot_f1_dev"])) / 2)
    write_summary(
        run_dir / "summary.txt",
        [
            "Mini-Project 2B: pretrained BERT and GPT2 for ATIS multitask NLU",
            f"Mode/device/seed: {args.mode} / {device.type} / {args.seed}",
            f"Best dev model: {best['model_name']}",
            f"Best dev intent accuracy: {best['intent_acc_dev']}",
            f"Best dev slot CoNLL F1: {best['slot_f1_dev']}",
            f"Test intent accuracy: {best['intent_acc_test']}",
            f"Test slot CoNLL F1: {best['slot_f1_test']}",
        ],
    )


if __name__ == "__main__":
    main()
