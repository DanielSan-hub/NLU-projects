import csv
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PART_RESULT_FILES = [
    "LM/partA/results/results_partA.csv",
    "LM/partB/results/results_partB.csv",
    "NLU/partA/results/results_partA.csv",
    "NLU/partB/results/results_partB.csv",
]
PART_BY_RESULT = {
    "LM/partA/results/results_partA.csv": "LM/partA",
    "LM/partB/results/results_partB.csv": "LM/partB",
    "NLU/partA/results/results_partA.csv": "NLU/partA",
    "NLU/partB/results/results_partB.csv": "NLU/partB",
}


def read_rows(path: Path, part_name: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["source_part"] = part_name
        row["source_csv"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return rows


def ordered_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "source_part",
        "source_csv",
        "part",
        "experiment_name",
        "model_name",
        "pretrained_model",
        "mode",
        "lr",
        "rank",
        "alpha",
        "d_model",
        "n_heads",
        "num_layers",
        "ff_dim",
        "dropout",
        "total_params",
        "trainable_params",
        "train_time_seconds",
        "tokens_per_second",
        "peak_memory_mb",
        "dev_ppl",
        "test_ppl",
        "intent_acc_dev",
        "slot_f1_dev",
        "intent_acc_test",
        "slot_f1_test",
        "checkpoint_path",
        "decision",
        "notes",
    ]
    seen = set()
    fieldnames: list[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames or ["source_csv"]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def first_float(row: dict[str, str], names: list[str]) -> tuple[str, float] | None:
    for name in names:
        value = as_float(row.get(name))
        if value is not None:
            return name, value
    return None


def score_row(row: dict[str, str]) -> tuple[float, str, float] | None:
    part = row.get("source_part", "")
    if part.startswith("LM/"):
        metric = first_float(row, ["dev_ppl", "test_ppl", "ppl", "valid_ppl"])
        if metric is None:
            return None
        name, value = metric
        return -value, name, value

    if part.startswith("NLU/"):
        slot = first_float(row, ["slot_f1_dev", "dev_slot_f1", "slot_f1_test", "test_slot_f1"])
        intent = first_float(row, ["intent_acc_dev", "dev_intent_acc", "intent_acc_test", "test_intent_acc"])
        if slot and intent:
            value = (slot[1] + intent[1]) / 2.0
            return value, "mean(dev intent acc, dev slot f1)", value
        if slot:
            return slot[1], slot[0], slot[1]
        if intent:
            return intent[1], intent[0], intent[1]
    return None


def row_label(row: dict[str, str]) -> str:
    return (
        row.get("experiment_name")
        or row.get("model_name")
        or row.get("pretrained_model")
        or row.get("target_modules")
        or "unknown"
    )


def best_rows_by_part(rows: list[dict[str, str]]) -> dict[str, tuple[dict[str, str], str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("source_part", "unknown")].append(row)

    best: dict[str, tuple[dict[str, str], str, float]] = {}
    for part, part_rows in grouped.items():
        scored = []
        for row in part_rows:
            score = score_row(row)
            if score is not None:
                scored.append((score[0], score[1], score[2], row))
        if scored:
            _, metric_name, metric_value, row = max(scored, key=lambda item: item[0])
            best[part] = (row, metric_name, metric_value)
    return best


def clean_cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "/")


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
    missing: list[str],
    best: dict[str, tuple[dict[str, str], str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.get("source_part", "unknown") for row in rows)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Master Results Summary",
        "",
        f"Generated: {timestamp}",
        "",
        "## Files",
        "",
        f"- Rows collected: {len(rows)}",
        f"- Master CSV: `results/master_results.csv`",
        "",
        "## Missing Result Files",
        "",
    ]
    if missing:
        lines.extend(f"- `{rel}`" for rel in missing)
    else:
        lines.append("- None")

    lines.extend(["", "## Rows By Part", ""])
    for part in PART_BY_RESULT.values():
        lines.append(f"- `{part}`: {counts.get(part, 0)}")

    lines.extend(
        [
            "",
            "## Best Result Per Part",
            "",
            "| Part | Experiment | Mode | Metric | Value | Source |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for part in PART_BY_RESULT.values():
        item = best.get(part)
        if item is None:
            lines.append(f"| {part} | missing metric |  |  |  |  |")
            continue
        row, metric_name, metric_value = item
        lines.append(
            "| "
            + " | ".join(
                [
                    clean_cell(part),
                    clean_cell(row_label(row)),
                    clean_cell(row.get("mode", "")),
                    clean_cell(metric_name),
                    f"{metric_value:.6g}",
                    clean_cell(row.get("source_csv", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for result_file in PART_RESULT_FILES:
        path = ROOT / result_file
        if not path.exists():
            missing.append(result_file)
            continue
        rows.extend(read_rows(path, PART_BY_RESULT[result_file]))

    master_csv = ROOT / "results" / "master_results.csv"
    summary_md = ROOT / "results" / "summary.md"
    write_rows(master_csv, rows)
    best = best_rows_by_part(rows)
    write_summary(summary_md, rows, missing, best)

    if missing:
        print("Missing result CSVs:")
        for rel in missing:
            print(f"  - {rel}")
    print(f"Collected {len(rows)} rows into {master_csv.relative_to(ROOT)}")
    print(f"Wrote summary to {summary_md.relative_to(ROOT)}")

    for part in PART_BY_RESULT.values():
        item = best.get(part)
        if item is None:
            print(f"Best {part}: unavailable")
            continue
        row, metric_name, metric_value = item
        print(
            f"Best {part}: {row_label(row)} "
            f"(mode={row.get('mode', '')}, {metric_name}={metric_value:.6g})"
        )


if __name__ == "__main__":
    main()
