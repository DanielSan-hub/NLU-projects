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


COLORS = {
    "blue": "#2f6fbb",
    "orange": "#f28e2b",
    "green": "#59a14f",
    "red": "#e15759",
    "purple": "#7b5ea7",
    "teal": "#4eaaa6",
    "gray": "#7f7f7f",
    "light_gray": "#d9dee7",
    "dark": "#20242a",
    "muted": "#59616f",
    "bg": "#ffffff",
}


def read_csv(rel: str) -> list[dict]:
    path = RESULT_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except Exception:
        return default


def by_name(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in rows}


def short_num(value: float) -> str:
    if math.isnan(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) < 1 and value != 0:
        return f"{value:.3f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str = "") -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
            f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>',
            '<style>',
            'text{font-family:Inter,Segoe UI,Arial,sans-serif;}',
            '.title{font-weight:700;fill:#20242a;}',
            '.subtitle{fill:#59616f;}',
            '.axis{stroke:#4b5563;stroke-width:1;}',
            '.grid{stroke:#d9dee7;stroke-width:1;}',
            '.tick{fill:#59616f;font-size:12px;}',
            '.label{fill:#20242a;font-size:13px;}',
            '.legend{fill:#20242a;font-size:13px;}',
            '</style>',
        ]

    def rect(self, x, y, w, h, fill, stroke="none", rx=0, opacity=1.0) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=COLORS["dark"], width=1, dash="") -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
        )

    def polyline(self, pts: list[tuple[float, float]], stroke, width=2.5, fill="none") -> None:
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    def circle(self, x, y, r, fill, stroke="white", width=1.5, opacity=1.0) -> None:
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def text(self, x, y, text, size=13, fill=COLORS["dark"], anchor="start", weight="400", cls="", rotate=None) -> None:
        klass = f' class="{cls}"' if cls else ""
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{klass}{transform}>{esc(text)}</text>'
        )

    def legend(self, items: list[tuple[str, str]], x: float, y: float) -> None:
        cx = x
        for label, color in items:
            self.rect(cx, y - 10, 14, 14, color, rx=2)
            self.text(cx + 20, y + 2, label, size=13, fill=COLORS["dark"], cls="legend")
            cx += 20 + len(label) * 7.4 + 26

    def finish(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def title_block(svg: SVG, title: str, subtitle: str) -> None:
    svg.text(60, 42, title, size=24, weight="700", cls="title")
    if subtitle:
        svg.text(60, 68, subtitle, size=14, cls="subtitle")


def y_ticks(max_value: float, n: int = 5) -> list[float]:
    if max_value <= 0 or math.isnan(max_value):
        return [0]
    step = max_value / n
    return [i * step for i in range(n + 1)]


def grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    max_value: float | None = None,
    value_format=short_num,
    rotate_labels: bool = True,
) -> None:
    width, height = 1250, 760
    left, right, top, bottom = 88, 42, 112, 168
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height, title)
    title_block(svg, title, subtitle)
    svg.legend([(name, color) for name, _, color in series], left, 92)
    max_v = max_value if max_value is not None else max(max(values) for _, values, _ in series) * 1.15
    max_v = max(max_v, 1e-9)
    for tick in y_ticks(max_v):
        y = top + plot_h - (tick / max_v) * plot_h
        svg.line(left, y, left + plot_w, y, stroke=COLORS["light_gray"])
        svg.text(left - 10, y + 4, value_format(tick), size=12, fill=COLORS["muted"], anchor="end")
    svg.line(left, top, left, top + plot_h, stroke=COLORS["dark"])
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, stroke=COLORS["dark"])
    svg.text(20, top + plot_h / 2, y_label, size=13, fill=COLORS["muted"], rotate=-90, anchor="middle")

    n = len(categories)
    group_w = plot_w / n
    gap = group_w * 0.18
    bar_w = (group_w - gap) / max(len(series), 1)
    for i, cat in enumerate(categories):
        gx = left + i * group_w + gap / 2
        for j, (_name, values, color) in enumerate(series):
            v = values[i]
            bh = (v / max_v) * plot_h
            x = gx + j * bar_w
            y = top + plot_h - bh
            svg.rect(x, y, max(bar_w - 3, 1), bh, color, rx=2)
            if n <= 8:
                svg.text(x + (bar_w - 3) / 2, y - 5, value_format(v), size=11, fill=COLORS["dark"], anchor="middle")
        label_x = left + i * group_w + group_w / 2
        if rotate_labels:
            svg.text(label_x - 3, top + plot_h + 18, cat, size=12, fill=COLORS["dark"], anchor="end", rotate=-35)
        else:
            svg.text(label_x, top + plot_h + 24, cat, size=12, fill=COLORS["dark"], anchor="middle")
    svg.finish(path)


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    x_labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    max_value: float | None = None,
    min_value: float = 0.0,
    value_format=short_num,
) -> None:
    width, height = 1100, 680
    left, right, top, bottom = 86, 42, 112, 106
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height, title)
    title_block(svg, title, subtitle)
    svg.legend([(name, color) for name, _, color in series], left, 92)
    all_values = [v for _, values, _ in series for v in values]
    max_v = max_value if max_value is not None else max(all_values) * 1.12
    min_v = min_value
    span = max(max_v - min_v, 1e-9)
    for tick in y_ticks(max_v - min_v):
        val = min_v + tick
        y = top + plot_h - ((val - min_v) / span) * plot_h
        svg.line(left, y, left + plot_w, y, stroke=COLORS["light_gray"])
        svg.text(left - 10, y + 4, value_format(val), size=12, fill=COLORS["muted"], anchor="end")
    svg.line(left, top, left, top + plot_h, stroke=COLORS["dark"])
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, stroke=COLORS["dark"])
    svg.text(20, top + plot_h / 2, y_label, size=13, fill=COLORS["muted"], rotate=-90, anchor="middle")
    n = len(x_labels)
    xs = [left + (plot_w * i / max(n - 1, 1)) for i in range(n)]
    for i, label in enumerate(x_labels):
        svg.text(xs[i], top + plot_h + 28, label, size=12, fill=COLORS["dark"], anchor="middle")
    for name, values, color in series:
        pts = []
        for x, v in zip(xs, values):
            y = top + plot_h - ((v - min_v) / span) * plot_h
            pts.append((x, y))
        svg.polyline(pts, color, width=3)
        for x, y in pts:
            svg.circle(x, y, 5, color)
    svg.finish(path)


def combo_bar_line(
    path: Path,
    title: str,
    subtitle: str,
    categories: list[str],
    bars: list[float],
    line_values: list[float],
    bar_label: str,
    line_label: str,
    bar_color: str,
    line_color: str,
) -> None:
    width, height = 1300, 760
    left, right, top, bottom = 86, 92, 112, 178
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height, title)
    title_block(svg, title, subtitle)
    svg.legend([(bar_label, bar_color), (line_label, line_color)], left, 92)
    max_bar = max(bars) * 1.15
    max_line = max(line_values) * 1.2
    for tick in y_ticks(max_bar):
        y = top + plot_h - (tick / max_bar) * plot_h
        svg.line(left, y, left + plot_w, y, stroke=COLORS["light_gray"])
        svg.text(left - 10, y + 4, short_num(tick), size=12, fill=COLORS["muted"], anchor="end")
    for tick in y_ticks(max_line):
        y = top + plot_h - (tick / max_line) * plot_h
        svg.text(left + plot_w + 10, y + 4, short_num(tick), size=12, fill=COLORS["muted"])
    svg.line(left, top, left, top + plot_h, stroke=COLORS["dark"])
    svg.line(left + plot_w, top, left + plot_w, top + plot_h, stroke=COLORS["dark"])
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, stroke=COLORS["dark"])
    svg.text(20, top + plot_h / 2, "Dev PPL", size=13, fill=COLORS["muted"], rotate=-90, anchor="middle")
    svg.text(width - 25, top + plot_h / 2, "Train-dev gap", size=13, fill=COLORS["muted"], rotate=90, anchor="middle")
    n = len(categories)
    group_w = plot_w / n
    pts = []
    for i, cat in enumerate(categories):
        x = left + i * group_w + group_w * 0.2
        bar_w = group_w * 0.6
        bh = (bars[i] / max_bar) * plot_h
        y = top + plot_h - bh
        svg.rect(x, y, bar_w, bh, bar_color, rx=2)
        lx = left + i * group_w + group_w / 2
        ly = top + plot_h - (line_values[i] / max_line) * plot_h
        pts.append((lx, ly))
        svg.text(lx - 3, top + plot_h + 18, cat, size=12, fill=COLORS["dark"], anchor="end", rotate=-35)
    svg.polyline(pts, line_color, width=3)
    for x, y in pts:
        svg.circle(x, y, 5, line_color)
    svg.finish(path)


def scatter_chart(
    path: Path,
    title: str,
    subtitle: str,
    points: list[dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    invert_y: bool = False,
) -> None:
    width, height = 1100, 760
    left, right, top, bottom = 96, 48, 112, 108
    plot_w = width - left - right
    plot_h = height - top - bottom
    svg = SVG(width, height, title)
    title_block(svg, title, subtitle)
    xs_raw = [p[x_key] for p in points]
    ys = [p[y_key] for p in points]
    xs = [math.log10(x) if log_x else x for x in xs_raw]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * 0.08 or 1.0
    pad_y = (max_y - min_y) * 0.08 or 1.0
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    for i in range(6):
        frac = i / 5
        x = left + frac * plot_w
        y = top + plot_h - frac * plot_h
        svg.line(x, top, x, top + plot_h, stroke=COLORS["light_gray"])
        svg.line(left, y, left + plot_w, y, stroke=COLORS["light_gray"])
        xv = min_x + frac * (max_x - min_x)
        yv = min_y + frac * (max_y - min_y)
        if log_x:
            svg.text(x, top + plot_h + 24, short_num(10**xv), size=12, fill=COLORS["muted"], anchor="middle")
        else:
            svg.text(x, top + plot_h + 24, short_num(xv), size=12, fill=COLORS["muted"], anchor="middle")
        svg.text(left - 10, y + 4, short_num(yv), size=12, fill=COLORS["muted"], anchor="end")
    svg.line(left, top, left, top + plot_h, stroke=COLORS["dark"])
    svg.line(left, top + plot_h, left + plot_w, top + plot_h, stroke=COLORS["dark"])
    svg.text(left + plot_w / 2, height - 28, x_label, size=13, fill=COLORS["muted"], anchor="middle")
    svg.text(24, top + plot_h / 2, y_label, size=13, fill=COLORS["muted"], rotate=-90, anchor="middle")
    for p in points:
        xv = math.log10(p[x_key]) if log_x else p[x_key]
        yv = p[y_key]
        x = left + ((xv - min_x) / (max_x - min_x)) * plot_w
        y_frac = (yv - min_y) / (max_y - min_y)
        y = top + (y_frac * plot_h if invert_y else plot_h - y_frac * plot_h)
        svg.circle(x, y, p.get("r", 7), p["color"], stroke="white", width=1.8, opacity=0.9)
        if p.get("label"):
            svg.text(x + 9, y - 7, p["label"], size=12, fill=COLORS["dark"])
    legend_items = []
    seen = set()
    for p in points:
        group = p.get("group")
        if group and group not in seen:
            seen.add(group)
            legend_items.append((group, p["color"]))
    if legend_items:
        svg.legend(legend_items, left, 92)
    svg.finish(path)


def make_project_summary(path: Path, stats: dict[str, tuple[str, str, str]]) -> None:
    width, height = 1200, 680
    svg = SVG(width, height, "Project summary")
    title_block(svg, "Project summary: best outcomes by mini-project", "Higher is better for NLU metrics; lower is better for LM perplexity.")
    panels = [
        ("LM/partA", stats["lm_a"], COLORS["blue"]),
        ("LM/partB", stats["lm_b"], COLORS["orange"]),
        ("NLU/partA", stats["nlu_a"], COLORS["green"]),
        ("NLU/partB", stats["nlu_b"], COLORS["purple"]),
    ]
    card_w, card_h = 510, 210
    xs = [70, 620]
    ys = [120, 380]
    for idx, (part, (name, metric, note), color) in enumerate(panels):
        x = xs[idx % 2]
        y = ys[idx // 2]
        svg.rect(x, y, card_w, card_h, "#f7f9fc", stroke="#d9dee7", rx=8)
        svg.rect(x, y, 8, card_h, color, rx=4)
        svg.text(x + 28, y + 42, part, size=18, weight="700", fill=color)
        svg.text(x + 28, y + 82, name, size=18, weight="700")
        svg.text(x + 28, y + 126, metric, size=31, weight="700", fill=COLORS["dark"])
        svg.text(x + 28, y + 162, note, size=14, fill=COLORS["muted"])
    svg.finish(path)


def generate() -> list[tuple[str, str, str]]:
    captions: list[tuple[str, str, str]] = []
    lm_a = [r for r in read_csv("LM/partA/results/results_partA.csv") if r.get("mode") == "core"]
    lm_a_extra = read_csv("LM/partA/results/results_partA_extra.csv")
    lm_b = [r for r in read_csv("LM/partB/results/results_partB.csv") if r.get("mode") == "core"]
    lm_b_extra = read_csv("LM/partB/results/results_partB_extra.csv")
    nlu_a = [r for r in read_csv("NLU/partA/results/results_partA.csv") if r.get("mode") == "core"]
    nlu_a_extra = read_csv("NLU/partA/results/results_partA_extra.csv")
    nlu_b_core = [r for r in read_csv("NLU/partB/results/results_partB.csv") if r.get("mode") == "core"]
    nlu_b_extra = read_csv("NLU/partB/results/results_partB_extra.csv")

    lm_a_n = by_name(lm_a, "experiment_name")
    lm_b_n = by_name(lm_b, "experiment_name")
    lm_b_e = by_name(lm_b_extra, "experiment_name")
    nlu_a_n = by_name(nlu_a, "experiment_name")

    # Figure 1: LM overview.
    lm_overview_rows = [
        ("Scratch base", lm_a_n["baseline_lr_5e-4"]),
        ("Scratch lr=1e-3", lm_a_n["lr_sweep_1e-3"]),
        ("Scratch d=192", lm_a_n["ablation_d_model_192"]),
        ("LoRA r1", lm_b_n["rank1_alpha2_qkv"]),
        ("LoRA r2", lm_b_n["rank2_alpha4_qkv"]),
        ("LoRA r4", lm_b_n["rank4_alpha8_qkv"]),
        ("LoRA r8", lm_b_n["rank8_alpha16_qkv"]),
        ("LoRA r16 extra", lm_b_e["extra_rank16_alpha32_qkv"]),
    ]
    grouped_bar_chart(
        FIG_DIR / "lm_ppl_overview.svg",
        "LM results on Penn Treebank",
        "Perplexity comparison. Lower is better. LoRA uses frozen pretrained GPT2 plus manual adapters.",
        [label for label, _ in lm_overview_rows],
        [
            ("Dev PPL", [f(r, "dev_ppl") for _, r in lm_overview_rows], COLORS["blue"]),
            ("Test PPL", [f(r, "test_ppl") for _, r in lm_overview_rows], COLORS["orange"]),
        ],
        "Perplexity",
        rotate_labels=True,
    )
    captions.append(("lm_ppl_overview.svg", "Recommended for LM report", "Main LM comparison: scratch GPT2 is below the PPL target, while manual LoRA on pretrained GPT2 roughly halves perplexity again."))

    # Figure 2: PartA ablation with overfitting gap.
    parta_order = [
        ("base", "baseline_lr_5e-4"),
        ("lr1e-3", "lr_sweep_1e-3"),
        ("lr3e-4", "lr_sweep_3e-4"),
        ("d192", "ablation_d_model_192"),
        ("h8", "ablation_n_heads_8"),
        ("L3", "ablation_num_layers_3"),
        ("ff768", "ablation_ff_dim_768"),
        ("drop0", "dropout_0_0"),
        ("drop.2", "dropout_0_2_weight_tying"),
    ]
    combo_bar_line(
        FIG_DIR / "lm_parta_ppl_gap.svg",
        "Scratch GPT2 ablations: dev PPL and train-dev gap",
        "Lower bars are better; a larger gap indicates stronger train/dev separation.",
        [a for a, _ in parta_order],
        [f(lm_a_n[k], "dev_ppl") for _, k in parta_order],
        [f(lm_a_n[k], "train_dev_gap") for _, k in parta_order],
        "Dev PPL",
        "Train-dev gap",
        COLORS["blue"],
        COLORS["red"],
    )
    captions.append(("lm_parta_ppl_gap.svg", "Supplemental LM figure", "Shows the mandatory one-at-a-time scratch GPT2 ablations and the overfitting-aware train-dev gap."))

    # Figure 3: LoRA rank sweep.
    rank_rows = [lm_b_n[f"rank{r}_alpha{2*r}_qkv"] for r in [1, 2, 4, 8]] + [lm_b_e["extra_rank16_alpha32_qkv"]]
    line_chart(
        FIG_DIR / "lm_partb_lora_rank_sweep.svg",
        "Manual LoRA rank sweep on GPT2 c_attn QKV",
        "Increasing rank increases trainable parameters and consistently improves PPL.",
        ["r1", "r2", "r4", "r8", "r16 extra"],
        [
            ("Dev PPL", [f(r, "dev_ppl") for r in rank_rows], COLORS["blue"]),
            ("Test PPL", [f(r, "test_ppl") for r in rank_rows], COLORS["orange"]),
        ],
        "Perplexity",
        max_value=40,
    )
    captions.append(("lm_partb_lora_rank_sweep.svg", "Recommended for LM report", "Rank/alpha sweep for manual LoRA. It is the cleanest visual evidence that larger low-rank adapters improve PTB adaptation."))

    # Figure 4: LoRA target ablation.
    target_rows = [
        ("Q only", lm_b_e["extra_q_only_r4_alpha8"]),
        ("K only", lm_b_e["extra_k_only_r4_alpha8"]),
        ("V only", lm_b_e["extra_v_only_r4_alpha8"]),
        ("QKV r4", lm_b_n["rank4_alpha8_qkv"]),
        ("QKV drop", lm_b_e["extra_qkv_dropout_r4_alpha8"]),
    ]
    grouped_bar_chart(
        FIG_DIR / "lm_lora_target_ablation.svg",
        "LoRA target ablation",
        "All target-only runs use r=4, alpha=8. QKV remains strongest; V-only is much better than Q/K-only.",
        [label for label, _ in target_rows],
        [
            ("Dev PPL", [f(r, "dev_ppl") for _, r in target_rows], COLORS["blue"]),
            ("Test PPL", [f(r, "test_ppl") for _, r in target_rows], COLORS["orange"]),
        ],
        "Perplexity",
        rotate_labels=False,
    )
    captions.append(("lm_lora_target_ablation.svg", "Supplemental LM figure", "Optional LoRA target analysis: adapting all QKV sections is clearly stronger than adapting only Q or K."))

    # Figure 5: LM efficiency scatter.
    points = []
    for r in lm_a:
        points.append({
            "tokens": f(r, "tokens_per_second"),
            "ppl": f(r, "dev_ppl"),
            "color": COLORS["blue"],
            "group": "Scratch GPT2",
            "r": 5 + min(f(r, "total_params") / 2_000_000, 8),
            "label": "d=192" if r["experiment_name"] == "ablation_d_model_192" else "",
        })
    for r in lm_b:
        points.append({
            "tokens": f(r, "tokens_per_second"),
            "ppl": f(r, "dev_ppl"),
            "color": COLORS["orange"],
            "group": "LoRA GPT2",
            "r": 5 + min(f(r, "trainable_params") / 80_000, 9),
            "label": "r8" if r["experiment_name"] == "rank8_alpha16_qkv" else "",
        })
    scatter_chart(
        FIG_DIR / "lm_efficiency_tradeoff.svg",
        "LM efficiency trade-off",
        "Dev PPL versus throughput. Lower PPL and higher tokens/s are preferable; point size follows parameter count.",
        points,
        "tokens",
        "ppl",
        "Tokens per second",
        "Dev PPL",
    )
    captions.append(("lm_efficiency_tradeoff.svg", "Supplemental LM figure", "Shows the practical trade-off: scratch models are faster, but LoRA reaches much lower PPL."))

    # Figure 6: NLU PartA core.
    nlu_a_order = [
        ("base", "baseline_lr_5e-4"),
        ("lr1e-3", "lr_sweep_1e-3"),
        ("lr3e-4", "lr_sweep_3e-4"),
        ("d192", "ablation_d_model_192"),
        ("h8", "ablation_n_heads_8"),
        ("L3", "ablation_num_layers_3"),
        ("ff768", "ablation_ff_dim_768"),
        ("drop.2", "dropout_before_heads_0_2"),
    ]
    grouped_bar_chart(
        FIG_DIR / "nlu_parta_core_metrics.svg",
        "Scratch GPT2 ATIS ablations",
        "Intent accuracy, slot F1 and semantic frame accuracy. Higher is better.",
        [label for label, _ in nlu_a_order],
        [
            ("Dev slot F1", [f(nlu_a_n[k], "slot_f1_dev") for _, k in nlu_a_order], COLORS["blue"]),
            ("Test slot F1", [f(nlu_a_n[k], "slot_f1_test") for _, k in nlu_a_order], COLORS["orange"]),
            ("Test frame", [f(nlu_a_n[k], "semantic_frame_acc_test") for _, k in nlu_a_order], COLORS["green"]),
        ],
        "Score",
        max_value=1.02,
        value_format=lambda v: f"{v:.2f}",
        rotate_labels=False,
    )
    captions.append(("nlu_parta_core_metrics.svg", "Recommended or supplemental NLU figure", "Mandatory scratch GPT2 NLU ablations. It highlights that d_model=192 is strongest for slot/frame, while heads=8 is strongest by combined dev score."))

    # Figure 7: NLU PartB model comparison.
    nlu_b_rows = [
        ("BERT core", nlu_b_core[0] if nlu_b_core[0]["model_name"] == "bert" else nlu_b_core[1]),
        ("GPT2 core", nlu_b_core[1] if nlu_b_core[1]["model_name"] == "gpt2" else nlu_b_core[0]),
    ]
    nlu_b_extra_by_model = by_name(nlu_b_extra, "model_name")
    nlu_b_rows += [
        ("GPT2 mean", nlu_b_extra_by_model["gpt2_mean_pool"]),
        ("BERT ontology", nlu_b_extra_by_model["bert_ontology_report"]),
    ]
    grouped_bar_chart(
        FIG_DIR / "nlu_partb_model_comparison.svg",
        "Pretrained NLU comparison on ATIS",
        "BERT's bidirectional context gives the best slot F1 and semantic frame accuracy.",
        [label for label, _ in nlu_b_rows],
        [
            ("Test intent acc", [f(r, "intent_acc_test") for _, r in nlu_b_rows], COLORS["blue"]),
            ("Test slot F1", [f(r, "slot_f1_test") for _, r in nlu_b_rows], COLORS["orange"]),
            ("Test frame", [f(r, "semantic_frame_acc_test") for _, r in nlu_b_rows], COLORS["green"]),
        ],
        "Score",
        max_value=1.02,
        value_format=lambda v: f"{v:.2f}",
        rotate_labels=False,
    )
    captions.append(("nlu_partb_model_comparison.svg", "Recommended for NLU report", "Core BERT/GPT2 comparison plus optional analyses. BERT wins clearly; GPT2 mean pooling recovers much of the gap."))

    # Figure 8: NLU overall.
    nlu_overall_rows = [
        ("Scratch base", nlu_a_n["baseline_lr_5e-4"]),
        ("Scratch h8", nlu_a_n["ablation_n_heads_8"]),
        ("Scratch d192", nlu_a_n["ablation_d_model_192"]),
        ("BERT", nlu_b_rows[0][1]),
        ("GPT2", nlu_b_rows[1][1]),
        ("GPT2 mean", nlu_b_rows[2][1]),
        ("BERT extra", nlu_b_rows[3][1]),
    ]
    grouped_bar_chart(
        FIG_DIR / "nlu_overall_test_metrics.svg",
        "Overall ATIS test comparison",
        "Slot F1 and full semantic frame accuracy across scratch and pretrained systems.",
        [label for label, _ in nlu_overall_rows],
        [
            ("Test slot F1", [f(r, "slot_f1_test") for _, r in nlu_overall_rows], COLORS["orange"]),
            ("Test frame", [f(r, "semantic_frame_acc_test") for _, r in nlu_overall_rows], COLORS["green"]),
        ],
        "Score",
        max_value=1.02,
        value_format=lambda v: f"{v:.2f}",
        rotate_labels=True,
    )
    captions.append(("nlu_overall_test_metrics.svg", "Recommended for NLU report", "Compact cross-part NLU comparison: BERT is best, while scratch GPT2 remains competitive and GPT2 mean pooling improves over last-token pooling."))

    # Figure 9: NLU scatter slot vs frame.
    scatter_points = []
    for label, r in nlu_overall_rows:
        if label.startswith("Scratch"):
            color = COLORS["green"]
            group = "Scratch GPT2"
        elif label.startswith("BERT"):
            color = COLORS["purple"]
            group = "BERT"
        else:
            color = COLORS["orange"]
            group = "GPT2"
        scatter_points.append({
            "slot": f(r, "slot_f1_test"),
            "frame": f(r, "semantic_frame_acc_test"),
            "color": color,
            "group": group,
            "r": 7,
            "label": label.replace("Scratch ", "S-"),
        })
    scatter_chart(
        FIG_DIR / "nlu_slot_frame_scatter.svg",
        "Slot F1 vs semantic frame accuracy",
        "Frame accuracy is stricter: an example is correct only if intent and all valid slots are correct.",
        scatter_points,
        "slot",
        "frame",
        "Test slot F1",
        "Test semantic frame accuracy",
    )
    captions.append(("nlu_slot_frame_scatter.svg", "Supplemental NLU figure", "Shows why semantic frame accuracy is stricter than slot F1 and separates BERT from GPT2 last-token pooling."))

    # Figure 10: project summary.
    best_lm_a = min(lm_a, key=lambda r: f(r, "dev_ppl"))
    best_lm_b_core = min(lm_b, key=lambda r: f(r, "dev_ppl"))
    best_nlu_a = max(nlu_a, key=lambda r: (f(r, "intent_acc_dev") + f(r, "slot_f1_dev")) / 2)
    best_nlu_b = max(nlu_b_core + nlu_b_extra, key=lambda r: (f(r, "intent_acc_dev") + f(r, "slot_f1_dev")) / 2)
    make_project_summary(
        FIG_DIR / "project_best_summary.svg",
        {
            "lm_a": (
                best_lm_a["experiment_name"],
                f"dev PPL {f(best_lm_a, 'dev_ppl'):.2f} / test {f(best_lm_a, 'test_ppl'):.2f}",
                "Scratch GPT2 target PPL < 250 met.",
            ),
            "lm_b": (
                best_lm_b_core["experiment_name"],
                f"dev PPL {f(best_lm_b_core, 'dev_ppl'):.2f} / test {f(best_lm_b_core, 'test_ppl'):.2f}",
                "Manual LoRA beats scratch with <0.4% trainable params.",
            ),
            "nlu_a": (
                best_nlu_a["experiment_name"],
                f"dev intent {f(best_nlu_a, 'intent_acc_dev'):.3f}, slot F1 {f(best_nlu_a, 'slot_f1_dev'):.3f}",
                "Best scratch GPT2 by combined dev score.",
            ),
            "nlu_b": (
                best_nlu_b["model_name"],
                f"test intent {f(best_nlu_b, 'intent_acc_test'):.3f}, slot F1 {f(best_nlu_b, 'slot_f1_test'):.3f}",
                "Best pretrained NLU model.",
            ),
        },
    )
    captions.append(("project_best_summary.svg", "Presentation / oral exam figure", "One-slide overview of the best result from each mini-project."))

    return captions


def write_index(captions: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Report Figure Index",
        "",
        "Figures generated from the final VM CSV results. All files are SVG vector graphics, so they remain sharp when resized.",
        "",
        "Recommended minimal selection:",
        "",
        "- LM report: `lm_ppl_overview.svg` and `lm_partb_lora_rank_sweep.svg`.",
        "- NLU report: `nlu_partb_model_comparison.svg` and either `nlu_overall_test_metrics.svg` or `nlu_parta_core_metrics.svg`.",
        "- Oral presentation or appendix: `project_best_summary.svg`, `lm_efficiency_tradeoff.svg`, `nlu_slot_frame_scatter.svg`.",
        "",
        "## Captions",
        "",
    ]
    for filename, use, caption in captions:
        lines.extend([f"### `{filename}`", "", f"- Use: {use}", f"- Caption: {caption}", ""])
    (FIG_DIR / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    captions = generate()
    write_index(captions)
    final_dir = WORKSPACE_ROOT / "NLU_results" / "extracted_final_results" / "reports" / "figures"
    if final_dir.parent.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        for path in FIG_DIR.glob("*"):
            if path.is_file():
                shutil.copy2(path, final_dir / path.name)
    print(f"Generated {len(list(FIG_DIR.glob('*.svg')))} SVG figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
