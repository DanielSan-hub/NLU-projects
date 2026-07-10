from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULT_ROOT = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results"
if not RESULT_ROOT.exists():
    RESULT_ROOT = PROJECT_ROOT

OUT_MD = PROJECT_ROOT / "reports" / "efficiency_tables.md"
OUT_TEX = PROJECT_ROOT / "reports" / "efficiency_tables.tex"


CSV_FILES = [
    ("LM/partA", "LM/partA/results/results_partA.csv"),
    ("LM/partA", "LM/partA/results/results_partA_extra.csv"),
    ("LM/partB", "LM/partB/results/results_partB.csv"),
    ("LM/partB", "LM/partB/results/results_partB_extra.csv"),
    ("NLU/partA", "NLU/partA/results/results_partA.csv"),
    ("NLU/partA", "NLU/partA/results/results_partA_extra.csv"),
    ("NLU/partB", "NLU/partB/results/results_partB.csv"),
    ("NLU/partB", "NLU/partB/results/results_partB_extra.csv"),
]


def read_csv(rel: str) -> list[dict]:
    with (RESULT_ROOT / rel).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ff(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except Exception:
        return default


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def params(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(int(value))


def mode_order(mode: str) -> int:
    return {"core": 0, "full": 1, "extras": 1, "smoke": 2}.get(mode, 9)


def experiment_name(row: dict) -> str:
    return row.get("experiment_name") or row.get("model_name") or "unknown"


def metric_summary(part: str, row: dict) -> str:
    if part.startswith("LM"):
        return f"dev/test PPL {ff(row, 'dev_ppl'):.2f}/{ff(row, 'test_ppl'):.2f}"
    return (
        f"dev slot {ff(row, 'slot_f1_dev'):.3f}, "
        f"test slot {ff(row, 'slot_f1_test'):.3f}, "
        f"frame {ff(row, 'semantic_frame_acc_test'):.3f}"
    )


def gather_rows() -> list[dict]:
    rows: list[dict] = []
    for part, rel in CSV_FILES:
        for row in read_csv(rel):
            if row.get("mode") == "smoke":
                continue
            rows.append({"part": part, "source": rel, **row})
    rows.sort(key=lambda r: (r["part"], mode_order(r.get("mode", "")), experiment_name(r)))
    return rows


def suite_totals(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    groups = []
    for part in ["LM/partA", "LM/partB", "NLU/partA", "NLU/partB"]:
        groups.append((part, [r for r in rows if r["part"] == part]))
    groups.append(("LM total", [r for r in rows if r["part"].startswith("LM")]))
    groups.append(("NLU total", [r for r in rows if r["part"].startswith("NLU")]))
    groups.append(("Project total", rows))
    return groups


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_markdown(rows: list[dict]) -> str:
    lines = [
        "# GPU, Memory And Time Tables",
        "",
        "Generated from final VM result CSV files. Smoke runs are excluded because they are functional checks, not performance experiments.",
        "",
        "Important interpretation note: `peak_memory_mb` is `torch.cuda.max_memory_allocated`, i.e. peak tensor memory allocated by PyTorch during the experiment. It is not the full `nvidia-smi` process footprint and does not include all CUDA caching/reserved memory. It is still useful for comparing experiments within this project.",
        "",
        "## Hardware Context",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["GPU", "NVIDIA Tesla T4"],
                ["GPU memory", "16GB"],
                ["VM size", "Azure Standard_NC8as_T4_v3"],
                ["Driver / CUDA shown by nvidia-smi", "555.42.06 / 12.5"],
                ["System RAM", "about 54GiB"],
                ["Execution", "sequential, unattended via tmux/scripts"],
                ["Core/extras AMP", "enabled when launched through scripts unless overridden"],
            ],
        ),
        "",
        "## Suite-Level Totals",
        "",
    ]
    total_rows = []
    for name, group in suite_totals(rows):
        total_time = sum(ff(r, "train_time_seconds") for r in group)
        peak = max((ff(r, "peak_memory_mb") for r in group), default=0.0)
        total_rows.append(
            [
                name,
                str(len(group)),
                fmt(total_time, 1),
                fmt(total_time / 60.0, 2),
                fmt(peak, 1),
            ]
        )
    lines.append(markdown_table(["Scope", "Experiments", "Train time s", "Train time min", "Max peak CUDA MB"], total_rows))
    lines.append("")
    lines.append("## Per-Experiment Efficiency")
    lines.append("")
    for part in ["LM/partA", "LM/partB", "NLU/partA", "NLU/partB"]:
        part_rows = [r for r in rows if r["part"] == part]
        body = []
        for r in part_rows:
            tok_s = ff(r, "tokens_per_second")
            body.append(
                [
                    experiment_name(r),
                    r.get("mode", ""),
                    fmt(ff(r, "train_time_seconds"), 2),
                    fmt(ff(r, "peak_memory_mb"), 1),
                    "" if tok_s == 0 else fmt(tok_s, 0),
                    params(ff(r, "trainable_params")),
                    metric_summary(part, r),
                ]
            )
        lines.extend(
            [
                f"### {part}",
                "",
                markdown_table(
                    ["Experiment", "Mode", "Time s", "Peak CUDA MB", "Tokens/s", "Trainable params", "Metric"],
                    body,
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def make_latex(rows: list[dict]) -> str:
    selected_names = {
        "LM/partA": ["baseline_lr_5e-4", "ablation_d_model_192"],
        "LM/partB": ["rank1_alpha2_qkv", "rank8_alpha16_qkv", "extra_rank16_alpha32_qkv"],
        "NLU/partA": ["baseline_lr_5e-4", "ablation_n_heads_8", "ablation_d_model_192"],
        "NLU/partB": ["bert", "gpt2", "gpt2_mean_pool", "bert_ontology_report"],
    }
    selected = []
    for part, names in selected_names.items():
        part_rows = [r for r in rows if r["part"] == part]
        for name in names:
            for r in part_rows:
                if experiment_name(r) == name:
                    selected.append(r)
                    break
    lines = [
        "% Compact efficiency table generated from reports/make_efficiency_tables.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrrl}",
        "\\hline",
        "Part & Experiment & Time(s) & Peak MB & Tok/s & Trainable & Metric \\\\",
        "\\hline",
    ]
    for r in selected:
        part = latex_escape(r["part"])
        exp = latex_escape(experiment_name(r))
        tok_s = ff(r, "tokens_per_second")
        metric = metric_summary(r["part"], r)
        lines.append(
            f"{part} & {exp} & {ff(r, 'train_time_seconds'):.1f} & {ff(r, 'peak_memory_mb'):.0f} & "
            f"{('-' if tok_s == 0 else f'{tok_s:.0f}')} & {params(ff(r, 'trainable_params'))} & {latex_escape(metric)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Training time and peak CUDA tensor memory measured on the university Tesla T4 16GB VM. Peak MB is PyTorch max allocated memory, not full nvidia-smi process memory.}",
            "\\label{tab:efficiency}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = gather_rows()
    OUT_MD.write_text(make_markdown(rows), encoding="utf-8")
    OUT_TEX.write_text(make_latex(rows), encoding="utf-8")
    final_reports = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results" / "reports"
    if final_reports.exists():
        (final_reports / OUT_MD.name).write_text(OUT_MD.read_text(encoding="utf-8"), encoding="utf-8")
        (final_reports / OUT_TEX.name).write_text(OUT_TEX.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
