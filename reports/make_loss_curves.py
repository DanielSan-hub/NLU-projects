from __future__ import annotations

import csv
import html
import math
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULT_ROOT = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results"
if not RESULT_ROOT.exists():
    RESULT_ROOT = PROJECT_ROOT
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


COLORS = [
    "#2f6fbb",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#7b5ea7",
    "#4eaaa6",
    "#9c755f",
    "#edc948",
    "#af7aa1",
]
BG = "#ffffff"
DARK = "#20242a"
MUTED = "#59616f"
GRID = "#d9dee7"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def read_epoch_log(rel: str) -> list[dict]:
    path = RESULT_ROOT / rel
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_epoch_log(part: str, run_marker: str, experiment: str) -> str:
    matches = sorted((RESULT_ROOT / part / "results").glob(f"*{run_marker}*/{experiment}/epoch_log.csv"))
    if not matches:
        raise FileNotFoundError(f"No epoch_log for {part} {run_marker} {experiment}")
    return str(matches[-1].relative_to(RESULT_ROOT)).replace("\\", "/")


def val(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return math.nan


def nice_label(name: str) -> str:
    replacements = {
        "baseline_lr_5e-4": "baseline",
        "lr_sweep_1e-3": "lr=1e-3",
        "lr_sweep_3e-4": "lr=3e-4",
        "ablation_d_model_192": "d_model=192",
        "ablation_n_heads_8": "heads=8",
        "ablation_num_layers_3": "layers=3",
        "ablation_ff_dim_768": "ff=768",
        "dropout_0_0": "dropout=0",
        "dropout_0_2_weight_tying": "dropout=0.2",
        "dropout_before_heads_0_2": "dropout=0.2",
        "rank1_alpha2_qkv": "LoRA r1",
        "rank2_alpha4_qkv": "LoRA r2",
        "rank4_alpha8_qkv": "LoRA r4",
        "rank8_alpha16_qkv": "LoRA r8",
        "extra_rank16_alpha32_qkv": "LoRA r16 extra",
        "bert": "BERT core",
        "gpt2": "GPT2 core",
        "bert_ontology_report": "BERT ontology",
        "gpt2_mean_pool": "GPT2 mean",
        "extra_semantic_frame_metrics": "semantic extra",
    }
    return replacements.get(name, name.replace("_", " "))


class SVG:
    def __init__(self, width: int, height: int, title: str, subtitle: str = "") -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
            f'<rect width="{width}" height="{height}" fill="{BG}"/>',
            "<style>",
            "text{font-family:Inter,Segoe UI,Arial,sans-serif;}",
            ".title{font-weight:700;fill:#20242a;}",
            ".subtitle{fill:#59616f;}",
            "</style>",
        ]
        self.text(60, 42, title, size=24, weight="700", cls="title")
        if subtitle:
            self.text(60, 68, subtitle, size=14, fill=MUTED, cls="subtitle")

    def line(self, x1, y1, x2, y2, stroke=DARK, width=1, dash="") -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')

    def rect(self, x, y, w, h, fill, stroke="none", rx=0) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')

    def polyline(self, pts: list[tuple[float, float]], stroke, width=2.5, dash="") -> None:
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts if math.isfinite(x) and math.isfinite(y))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')

    def circle(self, x, y, r, fill, stroke="white", width=1.2) -> None:
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, text, size=13, fill=DARK, anchor="start", weight="400", cls="", rotate=None) -> None:
        klass = f' class="{cls}"' if cls else ""
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{klass}{transform}>{esc(text)}</text>')

    def finish(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    curves: list[dict],
    y_label: str = "Loss",
    max_y: float | None = None,
    min_y: float | None = None,
) -> None:
    width, height = 1250, 760
    left, right, top, bottom = 88, 48, 126, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height, title, subtitle)

    all_epochs = [e for c in curves for e in c["epochs"]]
    all_values = [v for c in curves for v in c["values"] if math.isfinite(v)]
    min_epoch, max_epoch = min(all_epochs), max(all_epochs)
    min_v = min_y if min_y is not None else max(0.0, min(all_values) - 0.05 * (max(all_values) - min(all_values)))
    max_v = max_y if max_y is not None else max(all_values) + 0.08 * (max(all_values) - min(all_values))
    if max_v == min_v:
        max_v += 1
        min_v -= 1
    for i in range(6):
        frac = i / 5
        y = top + plot_h - frac * plot_h
        value = min_v + frac * (max_v - min_v)
        svg.line(left, y, left + plot_w, y, stroke=GRID)
        svg.text(left - 10, y + 4, f"{value:.2f}", size=12, fill=MUTED, anchor="end")
    max_ticks = max_epoch - min_epoch
    for epoch in range(min_epoch, max_epoch + 1):
        if max_ticks > 12 and epoch % 2 != 0:
            continue
        x = left + ((epoch - min_epoch) / max(max_epoch - min_epoch, 1)) * plot_w
        svg.line(x, top, x, top + plot_h, stroke=GRID)
        svg.text(x, top + plot_h + 26, str(epoch), size=12, fill=MUTED, anchor="middle")
    svg.line(left, top, left, top + plot_h, stroke=DARK)
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, stroke=DARK)
    svg.text(left + plot_w / 2, height - 28, "Epoch", size=13, fill=MUTED, anchor="middle")
    svg.text(24, top + plot_h / 2, y_label, size=13, fill=MUTED, rotate=-90, anchor="middle")

    # Legend. Train curves are solid, dev curves are dashed.
    lx, ly = left, 95
    for c in curves:
        if c.get("legend", True):
            svg.line(lx, ly - 4, lx + 24, ly - 4, stroke=c["color"], width=3, dash=c.get("dash", ""))
            svg.text(lx + 32, ly, c["label"], size=12, fill=DARK)
            lx += 38 + len(c["label"]) * 7.2
            if lx > width - 240:
                lx = left
                ly += 22

    span_epoch = max(max_epoch - min_epoch, 1)
    span_v = max_v - min_v
    for c in curves:
        pts = []
        for epoch, value in zip(c["epochs"], c["values"]):
            x = left + ((epoch - min_epoch) / span_epoch) * plot_w
            y = top + plot_h - ((value - min_v) / span_v) * plot_h
            pts.append((x, y))
        svg.polyline(pts, c["color"], width=c.get("width", 2.5), dash=c.get("dash", ""))
        if len(pts) <= 5:
            for x, y in pts:
                svg.circle(x, y, 4, c["color"])
    svg.finish(path)


def curves_for_experiments(part: str, run_marker: str, experiments: list[str], include_dev: bool = True) -> list[dict]:
    curves = []
    for i, exp in enumerate(experiments):
        rows = read_epoch_log(find_epoch_log(part, run_marker, exp))
        epochs = [int(float(r["epoch"])) for r in rows]
        color = COLORS[i % len(COLORS)]
        curves.append({
            "label": f"{nice_label(exp)} train",
            "epochs": epochs,
            "values": [val(r, "train_loss") for r in rows],
            "color": color,
        })
        if include_dev and "dev_loss" in rows[0]:
            curves.append({
                "label": f"{nice_label(exp)} dev",
                "epochs": epochs,
                "values": [val(r, "dev_loss") for r in rows],
                "color": color,
                "dash": "7 5",
            })
    return curves


def component_curves(part: str, run_marker: str, experiment: str, label_prefix: str, color_offset: int = 0) -> list[dict]:
    rows = read_epoch_log(find_epoch_log(part, run_marker, experiment))
    epochs = [int(float(r["epoch"])) for r in rows]
    keys = [("slot_loss", "slot loss"), ("intent_loss", "intent loss")]
    curves = []
    for i, (key, label) in enumerate(keys):
        if key in rows[0]:
            curves.append({
                "label": f"{label_prefix} {label}",
                "epochs": epochs,
                "values": [val(r, key) for r in rows],
                "color": COLORS[(color_offset + i) % len(COLORS)],
            })
    return curves


def generate() -> list[tuple[str, str, str]]:
    captions = []

    lm_a_selected = ["baseline_lr_5e-4", "lr_sweep_1e-3", "ablation_d_model_192", "dropout_0_2_weight_tying"]
    line_chart(
        FIG_DIR / "loss_lm_parta_selected_train_dev.svg",
        "LM/partA loss curves: selected scratch GPT2 runs",
        "Solid lines are train loss; dashed lines are dev loss. Lower is better.",
        curves_for_experiments("LM/partA", "core", lm_a_selected),
    )
    captions.append(("loss_lm_parta_selected_train_dev.svg", "LM training diagnostics", "Train/dev loss curves for selected scratch GPT2 runs. The wider model reaches the lowest dev loss, while heavier dropout learns more slowly."))

    lm_a_all = [
        "baseline_lr_5e-4",
        "lr_sweep_1e-3",
        "lr_sweep_3e-4",
        "ablation_d_model_192",
        "ablation_n_heads_8",
        "ablation_num_layers_3",
        "ablation_ff_dim_768",
        "dropout_0_0",
        "dropout_0_2_weight_tying",
    ]
    train_only = []
    for i, exp in enumerate(lm_a_all):
        rows = read_epoch_log(find_epoch_log("LM/partA", "core", exp))
        train_only.append({
            "label": nice_label(exp),
            "epochs": [int(float(r["epoch"])) for r in rows],
            "values": [val(r, "train_loss") for r in rows],
            "color": COLORS[i % len(COLORS)],
            "width": 2.1,
        })
    line_chart(
        FIG_DIR / "loss_lm_parta_all_train.svg",
        "LM/partA train loss across all mandatory experiments",
        "All core scratch GPT2 runs. This dense view is useful for diagnostics rather than the final one-page report.",
        train_only,
    )
    captions.append(("loss_lm_parta_all_train.svg", "Supplemental LM diagnostics", "Train loss for all mandatory scratch GPT2 experiments."))

    lm_b_ranks = ["rank1_alpha2_qkv", "rank2_alpha4_qkv", "rank4_alpha8_qkv", "rank8_alpha16_qkv"]
    line_chart(
        FIG_DIR / "loss_lm_partb_lora_rank_train_dev.svg",
        "LM/partB loss curves: LoRA rank sweep",
        "Solid lines are train loss; dashed lines are dev loss. Higher rank consistently lowers loss.",
        curves_for_experiments("LM/partB", "core", lm_b_ranks),
    )
    captions.append(("loss_lm_partb_lora_rank_train_dev.svg", "Recommended LM training diagnostic", "Train/dev loss curves for the LoRA rank sweep."))

    lm_b_extra = ["extra_q_only_r4_alpha8", "extra_k_only_r4_alpha8", "extra_v_only_r4_alpha8", "extra_qkv_dropout_r4_alpha8", "extra_rank16_alpha32_qkv"]
    line_chart(
        FIG_DIR / "loss_lm_partb_lora_extra_train_dev.svg",
        "LM/partB optional LoRA loss curves",
        "Target ablations and rank16 optional run. Solid=train, dashed=dev.",
        curves_for_experiments("LM/partB", "extras", lm_b_extra),
    )
    captions.append(("loss_lm_partb_lora_extra_train_dev.svg", "Supplemental LM diagnostics", "Optional LoRA target/rank loss curves."))

    nlu_a_selected = ["baseline_lr_5e-4", "ablation_d_model_192", "ablation_n_heads_8", "dropout_before_heads_0_2"]
    line_chart(
        FIG_DIR / "loss_nlu_parta_selected_train_dev.svg",
        "NLU/partA loss curves: selected scratch GPT2 ATIS runs",
        "Total multitask loss. Solid=train, dashed=dev. The models keep improving over 20 epochs.",
        curves_for_experiments("NLU/partA", "core", nlu_a_selected),
    )
    captions.append(("loss_nlu_parta_selected_train_dev.svg", "Recommended NLU training diagnostic", "Train/dev multitask loss for selected scratch GPT2 ATIS experiments."))

    nlu_a_components = (
        component_curves("NLU/partA", "core", "ablation_d_model_192", "d=192", 0)
        + component_curves("NLU/partA", "core", "ablation_n_heads_8", "heads=8", 2)
    )
    line_chart(
        FIG_DIR / "loss_nlu_parta_slot_intent_components.svg",
        "NLU/partA loss components",
        "Slot loss and intent loss for the two strongest scratch GPT2 configurations.",
        nlu_a_components,
    )
    captions.append(("loss_nlu_parta_slot_intent_components.svg", "Supplemental NLU diagnostics", "Slot and intent loss components show how the multitask objective evolves."))

    nlu_b_models = ["bert", "gpt2"]
    line_chart(
        FIG_DIR / "loss_nlu_partb_core_train_dev.svg",
        "NLU/partB loss curves: BERT vs GPT2 core",
        "Total multitask loss. Solid=train, dashed=dev. BERT converges to a much lower dev loss.",
        curves_for_experiments("NLU/partB", "core", nlu_b_models),
    )
    captions.append(("loss_nlu_partb_core_train_dev.svg", "Recommended NLU training diagnostic", "Train/dev loss curves for BERT and GPT2 core multitask fine-tuning."))

    nlu_b_extra = ["bert_ontology_report", "gpt2_mean_pool"]
    line_chart(
        FIG_DIR / "loss_nlu_partb_extra_train_dev.svg",
        "NLU/partB optional loss curves",
        "BERT ontology report and GPT2 mean-pooling extension. Solid=train, dashed=dev.",
        curves_for_experiments("NLU/partB", "extras", nlu_b_extra),
    )
    captions.append(("loss_nlu_partb_extra_train_dev.svg", "Supplemental NLU diagnostics", "Loss curves for optional pretrained NLU analyses."))

    return captions


def append_index(captions: list[tuple[str, str, str]]) -> None:
    index = FIG_DIR / "figure_index.md"
    text = index.read_text(encoding="utf-8") if index.exists() else "# Report Figure Index\n"
    marker = "## Loss Curve Figures"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    lines = [text.rstrip(), "", marker, "", "These figures are generated from per-experiment `epoch_log.csv` files saved during the VM runs.", ""]
    for filename, use, caption in captions:
        lines.extend([f"### `{filename}`", "", f"- Use: {use}", f"- Caption: {caption}", ""])
    index.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    captions = generate()
    append_index(captions)
    final_dir = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results" / "reports" / "figures"
    if final_dir.exists():
        for path in FIG_DIR.glob("loss_*.svg"):
            shutil.copy2(path, final_dir / path.name)
        shutil.copy2(FIG_DIR / "figure_index.md", final_dir / "figure_index.md")
    print(f"Generated {len(captions)} loss-curve SVG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
