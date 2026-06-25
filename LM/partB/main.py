import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import torch

from functions import (
    MODEL_NAME,
    RESULT_COLUMNS,
    build_datasets,
    evaluate,
    experiments_for_mode,
    forward_logits,
    lm_loss,
    load_gpt2_lm_model,
    load_gpt2_tokenizer,
    make_dataloader,
    read_part_a_best,
)
from model import (
    assert_lora_training_safety,
    freeze_pretrained_parameters,
    inject_lora_into_gpt2_c_attn,
    inspect_c_attn_modules,
    load_lora_state_dict,
    lora_state_dict,
)
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
    parser = argparse.ArgumentParser(description="Mini-Project 1B: pretrained GPT-2 with manual LoRA on PTB.")
    parser.add_argument("--mode", choices=["smoke", "core", "full"], default="smoke")
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
    fieldnames = ["epoch", "train_loss", "dev_loss", "dev_ppl", "is_best"]
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
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=amp_state["enabled"])
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits = forward_logits(model, input_ids, attention_mask)
            loss = lm_loss(logits, labels)

        if not torch.isfinite(loss):
            if amp_state["enabled"]:
                print("WARNING: non-finite loss under AMP; disabling AMP and retrying this batch in fp32.")
                amp_state["enabled"] = False
                amp_state["scaler"] = None
                optimizer.zero_grad(set_to_none=True)
                logits = forward_logits(model, input_ids, attention_mask)
                loss = lm_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered with AMP disabled.")

        if amp_state["enabled"]:
            amp_state["scaler"].scale(loss).backward()
            amp_state["scaler"].unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["grad_clip"])
            amp_state["scaler"].step(optimizer)
            amp_state["scaler"].update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["grad_clip"])
            optimizer.step()

        tokens = int(batch["num_tokens"].sum().item())
        total_loss += loss.item() * tokens
        total_tokens += tokens
        if config["max_train_batches"] and step >= config["max_train_batches"]:
            break

    return {"loss": total_loss / max(total_tokens, 1), "tokens": total_tokens}


def verify_forward_and_loss(model, batch, device, vocab_size: int) -> dict:
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    with torch.no_grad():
        logits = forward_logits(model, input_ids, attention_mask)
        loss = lm_loss(logits, labels)
        labels_with_ignored_position = labels.clone()
        labels_with_ignored_position[0, -1] = -100
        ignored_loss = lm_loss(logits, labels_with_ignored_position)
    expected = (input_ids.size(0), input_ids.size(1), vocab_size)
    assert tuple(logits.shape) == expected, f"Expected logits shape {expected}, got {tuple(logits.shape)}"
    assert torch.isfinite(loss), "Forward loss is not finite."
    assert torch.isfinite(ignored_loss), "Padding/ignore-index loss check is not finite."
    return {
        "logits_shape": tuple(logits.shape),
        "loss": float(loss.item()),
        "padding_ignored_loss": float(ignored_loss.item()),
    }


def train_experiment(
    config: dict,
    run_dir: Path,
    datasets: dict,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    result_csv: Path,
    part_a_best: dict,
    writer_enabled: bool,
) -> dict:
    set_seed(args.seed)
    exp_dir = run_dir / config["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.update(
        {
            "part": "LM/partB",
            "seed": args.seed,
            "requested_device": args.device,
            "actual_device": device.type,
            "amp_requested": args.amp,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "grad_clip": 1.0,
            "pretrained_model": MODEL_NAME,
            "tokenizer": MODEL_NAME,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
    save_json(exp_dir / "config.json", config)
    writer = get_tensorboard_writer(exp_dir / "tensorboard", writer_enabled)

    train_loader = make_dataloader(datasets["train"], config, args, shuffle=True)
    dev_loader = make_dataloader(datasets["dev"], config, args, shuffle=False)
    test_loader = make_dataloader(datasets["test"], config, args, shuffle=False)
    probe_batch = next(iter(train_loader))

    model = load_gpt2_lm_model()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    freeze_pretrained_parameters(model)

    c_attn_before = inspect_c_attn_modules(model)
    if not c_attn_before:
        raise RuntimeError("Could not find GPT-2 c_attn modules for LoRA injection.")
    save_json(exp_dir / "c_attn_inspection.json", {"modules": c_attn_before})
    print("c_attn modules before LoRA:")
    for item in c_attn_before:
        print(f"  {item['name']} type={item['type']} weight_shape={item['weight_shape']} bias_shape={item['bias_shape']}")

    model.eval()
    probe_input = probe_batch["input_ids"].to(device)
    probe_attention = probe_batch["attention_mask"].to(device)
    with torch.no_grad():
        base_logits = forward_logits(model, probe_input, probe_attention).detach().float().cpu()

    reports = inject_lora_into_gpt2_c_attn(
        model=model,
        rank=config["rank"],
        alpha=config["alpha"],
        target_sections=config["target_modules"],
        dropout=config["lora_dropout"],
        sample_input_ids=probe_input,
    )
    save_json(
        exp_dir / "lora_injection_report.json",
        {"reports": [report.__dict__ for report in reports]},
    )
    trainable_names = assert_lora_training_safety(model)
    print("Trainable parameter names:")
    for name in trainable_names:
        print(f"  {name}")
    (exp_dir / "trainable_params.txt").write_text("\n".join(trainable_names) + "\n", encoding="utf-8")

    with torch.no_grad():
        lora_logits = forward_logits(model, probe_input, probe_attention).detach().float().cpu()
    max_abs_diff = float((base_logits - lora_logits).abs().max().item())
    print(f"Step-0 base vs LoRA max_abs_diff={max_abs_diff:.8f}")
    if max_abs_diff > 1e-5:
        raise AssertionError(f"LoRA step-0 delta should be near zero, got max_abs_diff={max_abs_diff}")

    forward_check = verify_forward_and_loss(model, probe_batch, device, len(tokenizer))
    print(
        f"Forward check: logits_shape={forward_check['logits_shape']} "
        f"loss={forward_check['loss']:.4f} padding_ignored_loss={forward_check['padding_ignored_loss']:.4f}"
    )
    save_json(
        exp_dir / "safety_checks.json",
        {
            "max_abs_diff_base_vs_lora_step0": max_abs_diff,
            "forward_check": forward_check,
            "padding_ignored": True,
            "all_original_gpt2_parameters_frozen": True,
        },
    )

    total_params, trainable_params = count_parameters(model)
    trainable_percent = 100.0 * trainable_params / max(total_params, 1)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print(f"Trainable percent: {trainable_percent:.6f}%")
    assert trainable_params > 0
    assert trainable_params < total_params

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    amp_state = {"enabled": amp_enabled, "scaler": scaler}

    start_epoch = 0
    best_epoch = 0
    best_dev_loss = math.inf
    best_dev_ppl = math.inf
    epoch_rows: list[dict] = []
    best_checkpoint = exp_dir / "best_lora_adapters.pt"
    last_checkpoint = exp_dir / "last_lora_adapters.pt"

    if args.resume and last_checkpoint.exists():
        checkpoint = load_checkpoint(last_checkpoint, device)
        load_lora_state_dict(model, checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if amp_state["scaler"] is not None and checkpoint.get("scaler_state"):
            amp_state["scaler"].load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_epoch = checkpoint.get("best_epoch", 0)
        best_dev_loss = checkpoint.get("best_dev_loss", math.inf)
        best_dev_ppl = checkpoint.get("best_dev_ppl", math.inf)
        epoch_rows = checkpoint.get("epoch_rows", [])
        amp_state["enabled"] = checkpoint.get("amp_enabled", amp_state["enabled"])

    reset_peak_memory(device)
    train_start = time.perf_counter()
    total_train_tokens = 0

    for epoch in range(start_epoch, config["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, amp_state)
        dev_metrics = evaluate(model, dev_loader, device, config["max_eval_batches"])
        total_train_tokens += train_metrics["tokens"]
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
            "is_best": is_best,
        }
        epoch_rows.append(epoch_row)
        checkpoint = {
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_dev_loss": best_dev_loss,
            "best_dev_ppl": best_dev_ppl,
            "adapter_state": lora_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": amp_state["scaler"].state_dict() if amp_state["scaler"] is not None else None,
            "epoch_rows": epoch_rows,
            "amp_enabled": amp_state["enabled"],
            "config": config,
        }
        save_checkpoint(last_checkpoint, checkpoint)
        if is_best:
            save_checkpoint(
                best_checkpoint,
                {
                    "adapter_state": lora_state_dict(model),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_dev_loss": best_dev_loss,
                    "best_dev_ppl": best_dev_ppl,
                    "max_abs_diff_base_vs_lora_step0": max_abs_diff,
                    "trainable_parameter_names": trainable_names,
                },
            )
            save_json(exp_dir / "config_best_lora.json", config)

        print(
            f"{config['experiment_name']} epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} dev_loss={dev_metrics['loss']:.4f} "
            f"dev_ppl={dev_metrics['ppl']:.2f}"
        )
        if writer:
            step = epoch + 1
            writer.add_scalar("train/loss", train_metrics["loss"], step)
            writer.add_scalar("dev/loss", dev_metrics["loss"], step)
            writer.add_scalar("dev/ppl", dev_metrics["ppl"], step)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)

    train_time = time.perf_counter() - train_start
    tokens_per_second = total_train_tokens / train_time if train_time > 0 else 0.0
    write_epoch_log(exp_dir / "epoch_log.csv", epoch_rows)

    if best_checkpoint.exists():
        best_state = load_checkpoint(best_checkpoint, device)
        load_lora_state_dict(model, best_state["adapter_state"])
    test_metrics = evaluate(model, test_loader, device, config["max_eval_batches"])
    peak_mb = peak_memory_mb(device)
    if writer:
        step = max(best_epoch, 1)
        writer.add_scalar("train_time_seconds", train_time, step)
        writer.add_scalar("tokens_per_second", tokens_per_second, step)
        writer.add_scalar("peak_memory_mb", peak_mb, step)
        writer.flush()
        writer.close()

    part_a_best_ppl = float(part_a_best["dev_ppl"]) if not math.isnan(float(part_a_best["dev_ppl"])) else math.nan
    improves_over_part_a = False if math.isnan(part_a_best_ppl) else best_dev_ppl < part_a_best_ppl
    notes = (
        f"manual LoRA on fused c_attn Q/K/V sections; step0_max_abs_diff={max_abs_diff:.8g}; "
        f"partA_best_test_ppl={part_a_best.get('test_ppl', math.nan)}"
    )
    if config["mode"] == "full":
        notes += "; optional efficiency reporting: "
        if not math.isnan(part_a_best_ppl):
            improvement = part_a_best_ppl - best_dev_ppl
            notes += (
                f"dev_ppl_improvement_per_trainable_param={improvement / max(trainable_params, 1):.8g}, "
                f"dev_ppl_improvement_per_minute={improvement / max(train_time / 60.0, 1e-12):.8g}"
            )
        else:
            notes += "partA baseline unavailable"

    row = {
        "part": "LM/partB",
        "experiment_name": config["experiment_name"],
        "mode": config["mode"],
        "rank": config["rank"],
        "alpha": config["alpha"],
        "target_modules": f"c_attn:{config['target_modules']}",
        "lr": config["lr"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": round(trainable_percent, 6),
        "train_time_seconds": round(train_time, 3),
        "tokens_per_second": round(tokens_per_second, 3),
        "peak_memory_mb": round(peak_mb, 3),
        "dev_ppl": round(best_dev_ppl, 6),
        "test_ppl": round(test_metrics["ppl"], 6),
        "partA_best_ppl": "" if math.isnan(part_a_best_ppl) else round(part_a_best_ppl, 6),
        "improves_over_partA": improves_over_part_a,
        "checkpoint_path": relative(best_checkpoint),
        "notes": notes,
    }
    append_result_csv(result_csv, row, fieldnames=RESULT_COLUMNS)
    write_summary(
        exp_dir / "summary.txt",
        [
            f"LM/partB experiment: {config['experiment_name']}",
            f"Model: {MODEL_NAME}",
            f"Rank/alpha/targets: {config['rank']} / {config['alpha']} / {config['target_modules']}",
            f"Frozen pretrained parameters: yes",
            f"Trainable params: {trainable_params} ({trainable_percent:.6f}%)",
            f"Base vs LoRA step-0 max abs diff: {max_abs_diff:.8g}",
            f"Best epoch: {best_epoch}",
            f"Best dev loss/PPL: {best_dev_loss:.4f} / {best_dev_ppl:.2f}",
            f"Test PPL from best adapters: {test_metrics['ppl']:.2f}",
            f"PartA best dev PPL: {part_a_best_ppl}",
            f"Improves over PartA: {improves_over_part_a}",
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
            "part": "LM/partB",
            "mode": args.mode,
            "seed": args.seed,
            "device": device.type,
            "pretrained_model": MODEL_NAME,
            "experiments": [exp["experiment_name"] for exp in all_exps],
        },
    )

    tokenizer = load_gpt2_tokenizer()
    max_block_size = max(exp["block_size"] for exp in all_exps)
    datasets = build_datasets(PROJECT_DIR, tokenizer, max_block_size)
    if any(exp["block_size"] != max_block_size for exp in all_exps):
        raise ValueError("All LM/partB experiments in a single run must share block_size for cached datasets.")

    part_a_best = read_part_a_best(PROJECT_DIR)
    if math.isnan(float(part_a_best["dev_ppl"])):
        print("PartA best PPL unavailable; comparison will be left blank.")
    else:
        print(
            f"PartA best scratch dev PPL={part_a_best['dev_ppl']:.4f} "
            f"test PPL={part_a_best['test_ppl']:.4f} ({part_a_best.get('experiment_name', '')})"
        )

    rows = []
    for exp in core_exps:
        rows.append(
            train_experiment(
                exp,
                run_dir,
                datasets,
                tokenizer,
                args,
                device,
                PROJECT_DIR / "results" / "results_partB.csv",
                part_a_best,
                writer_enabled=args.log_tensorboard,
            )
        )
    for exp in extra_exps:
        rows.append(
            train_experiment(
                exp,
                run_dir,
                datasets,
                tokenizer,
                args,
                device,
                PROJECT_DIR / "results" / "results_partB_extra.csv",
                part_a_best,
                writer_enabled=args.log_tensorboard,
            )
        )

    best_row = min(rows, key=lambda row: float(row["dev_ppl"]))
    write_summary(
        run_dir / "summary.txt",
        [
            "Mini-Project 1B: pretrained GPT-2 + manual LoRA",
            f"Mode/device/seed: {args.mode} / {device.type} / {args.seed}",
            f"Final best configuration: {best_row['experiment_name']}",
            f"Best dev PPL: {best_row['dev_ppl']}",
            f"Best test PPL: {best_row['test_ppl']}",
            f"Improves over PartA: {best_row['improves_over_partA']}",
            f"Checkpoint: {best_row['checkpoint_path']}",
        ],
    )
    print(
        "Final best configuration: "
        f"{best_row['experiment_name']} dev_ppl={best_row['dev_ppl']} "
        f"test_ppl={best_row['test_ppl']} improves_over_partA={best_row['improves_over_partA']} "
        f"checkpoint={best_row['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
