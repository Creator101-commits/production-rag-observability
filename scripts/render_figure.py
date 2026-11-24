#!/usr/bin/env python3
"""Render image.png from results/ in the shared portfolio style.

Reads results/metrics.json, results/manifest.json and
results/traces.jsonl only; contains no metric values of its own.
Run: uv run --with matplotlib python scripts/render_figure.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150

def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color")
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)



ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "image.png"

RETRIEVAL_ORDER = [
    "bm25",
    "dense",
    "hybrid-rrf-k10-w1-1",
    "hybrid-rrf-k10-w1-2",
    "hybrid-rrf-k10-w2-1",
    "hybrid-rrf-k30-w1-1",
    "hybrid-rrf-k30-w1-2",
    "hybrid-rrf-k30-w2-1",
    "hybrid-rrf-k60-w1-1",
    "hybrid-rrf-k60-w1-2",
    "hybrid-rrf-k60-w2-1",
    "hybrid-rrf-k10-w1-1+rerank-none",
    "hybrid-rrf-k10-w1-1+rerank-cross_encoder",
]

STAGE_ORDER = [
    "retrieval", "fusion", "rerank", "request", "generation",
    "nli_scoring",
]

E2E_ORDER = [
    ("required_point_recall", "required-point recall"),
    ("all_points_covered", "all points covered"),
    ("abstain_f1", "abstain F1"),
    ("citation_label_validity", "citation validity"),
    ("citation_page_precision", "citation page prec."),
    ("citation_hit_rate", "citation hit rate"),
    ("nli_claim_support_rate", "NLI support (est.)"),
]

GEN_LABELS = {
    "qwen2.5-3b-fp16__k3": "3B, k=3",
    "qwen2.5-3b-fp16__k5": "3B, k=5",
    "qwen2.5-7b-nf4__k3": "7B, k=3",
    "qwen2.5-7b-nf4__k5": "7B, k=5",
}
GEN_ORDER = list(GEN_LABELS)


def _short_system(name: str) -> str:
    s = name.replace("hybrid-rrf-", "")
    s = s.replace("+rerank-cross_encoder", "+CE")
    s = s.replace("+rerank-none", "+no-rerank")
    return s


def _load_stage_durations() -> dict[str, list[float]]:
    durs: dict[str, list[float]] = defaultdict(list)
    with open(RESULTS / "traces.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            durs[rec["name"]].append(float(rec["duration_ms"]))
    return durs


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    frozen_ret = manifest["frozen_config"]["retrieval"]["name"]
    frozen_rerank = manifest["frozen_config"]["rerank"]
    frozen_gen = "qwen2.5-3b-fp16__k5"
    retrieval = metrics["retrieval_ablations_test"]
    e2e = metrics["e2e_test"]
    gen = metrics["generator_ablations_test"]
    stage_durs = _load_stage_durations()

    fig, axes = plt.subplots(
        2, 2, figsize=(18.0, 10.5), dpi=DPI, layout="constrained"
    )
    (a1, a2), (a3, a4) = axes

    # (a) retrieval ablation: nDCG@10 and Hit@5 for every test system.
    names = [n for n in RETRIEVAL_ORDER if n in retrieval]
    ys = np.arange(len(names))[::-1]
    h = 0.36
    for y, name in zip(ys, names, strict=True):
        entry = retrieval[name]
        is_frozen = (
            name == f"{frozen_ret}+rerank-{frozen_rerank}"
        )
        ndcg_color = AMBER if is_frozen else BLUE
        hit_color = AMBER if is_frozen else GRAY
        a1.barh(
            y + h / 2, entry["ndcg_at_10"], h,
            color=ndcg_color, alpha=0.9 if is_frozen else 0.85,
        )
        a1.barh(
            y - h / 2, entry["hit_at_5"], h,
            color=hit_color, alpha=0.9 if is_frozen else 0.55,
        )
        if is_frozen:
            a1.annotate(
                "frozen", (entry["hit_at_5"] + 0.015, y),
                va="center", fontsize=8, color=AMBER,
            )
    style(
        a1, "a", "Retrieval ablation, test (blue nDCG@10, gray Hit@5)",
        xlabel="score", grid=False,
    )
    a1.set_xlim(0.0, 1.12)
    cat(
        a1, "y", ys,
        [_short_system(n) for n in names], fontsize=7,
    )
    frozen_idx = names.index(
        f"{frozen_ret}+rerank-{frozen_rerank}"
    )
    a1.get_yticklabels()[len(names) - 1 - frozen_idx].set_color(INK)

    # (b) per-stage latency distributions from exported spans, log ms.
    stages = [s for s in STAGE_ORDER if s in stage_durs]
    xpos = np.arange(len(stages))
    boxes = a2.boxplot(
        [stage_durs[s] for s in stages],
        positions=xpos,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.5,
                    "markerfacecolor": GRAY,
                    "markeredgecolor": GRAY},
        medianprops={"color": INK, "linewidth": 1.4},
        whiskerprops={"color": TICK_LABEL, "linewidth": 0.9},
        capprops={"color": TICK_LABEL, "linewidth": 0.9},
        boxprops={"linewidth": 0.9},
    )
    for patch, stage in zip(boxes["boxes"], stages, strict=True):
        patch.set_facecolor(BLUE if stage != "request" else TEAL)
        patch.set_alpha(0.55)
        patch.set_edgecolor(TICK_LABEL)
    a2.set_yscale("log")
    a2.set_ylim(0.05, 120000.0)
    style(
        a2, "b",
        "Stage latency, test spans (log ms; generation pooled x4)",
        xlabel="stage", ylabel="duration (ms)",
    )
    a2.yaxis.set_major_formatter(
        mticker.LogFormatterMathtext(labelOnlyBase=True)
    )
    cat(a2, "x", xpos, stages)

    # (c) end-to-end quality rates with bootstrap CIs.
    labels = []
    means, lows, highs = [], [], []
    for key, label in E2E_ORDER:
        val = e2e.get(key)
        labels.append(label)
        if isinstance(val, dict):
            means.append(float(val["mean"]))
            lows.append(float(val["mean"]) - float(val["ci_low"]))
            highs.append(float(val["ci_high"]) - float(val["mean"]))
        else:
            means.append(float(val))
            lows.append(0.0)
            highs.append(0.0)
    y3 = np.arange(len(labels))[::-1]
    a3.barh(
        y3, means, height=0.62, color=BLUE, alpha=0.85,
    )
    a3.errorbar(
        means, y3, xerr=[lows, highs], fmt="none",
        ecolor=INK, elinewidth=1.1, capsize=3,
    )
    style(
        a3, "c", "End-to-end quality, frozen system (test)",
        xlabel="rate", grid=False,
    )
    a3.set_xlim(0.0, 1.12)
    cat(a3, "y", y3, labels, fontsize=7.5)

    # (d) generator x context ablation: recall vs false-abstain.
    gx = np.arange(len(GEN_ORDER))
    w = 0.36
    recalls, fas = [], []
    for name in GEN_ORDER:
        entry = gen[name]
        recalls.append(float(entry["required_point_recall"]["mean"]))
        fas.append(float(entry["false_abstain_rate"]))
    recalls_a = np.array(recalls)
    fas_a = np.array(fas)
    a4.bar(gx - w / 2, recalls_a, w, color=BLUE, alpha=0.85)
    a4.bar(gx + w / 2, fas_a, w, color=GRAY, alpha=0.7)
    frozen_x = GEN_ORDER.index(frozen_gen)
    a4.annotate(
        "frozen", (frozen_x, max(recalls_a[frozen_x], fas_a[frozen_x]) + 0.04),
        ha="center", fontsize=8, color=AMBER,
    )
    style(
        a4, "d",
        "Generator x context, test (blue recall, gray false-abstain)",
        xlabel="candidate", ylabel="rate",
    )
    a4.set_ylim(0.0, 1.15)
    cat(a4, "x", gx, [GEN_LABELS[n] for n in GEN_ORDER])
    for t in a4.get_xticklabels():
        if t.get_text() == GEN_LABELS[frozen_gen]:
            t.set_color(INK)

    finish(fig, OUT)


if __name__ == "__main__":
    main()
