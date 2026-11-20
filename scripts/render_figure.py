"""Render image.png from scripts/figure-data.json.

The data file is generated from curated benchmark results; this script
contains no metric values of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "scripts" / "figure-data.json").read_text())

COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b", "#e8a13c"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

# Panel 1: retrieval ablation nDCG@10 (test, retrieval-only).
abl = DATA["retrieval_ablations"]
names = list(abl.keys())
ndcg = [abl[n]["ndcg_at_10"] for n in names]
hit5 = [abl[n]["hit_at_5"] for n in names]
y = np.arange(len(names))
axes[0].barh(y, ndcg, color=COLORS[0], label="nDCG@10")
axes[0].barh(y, hit5, color=COLORS[4], alpha=0.55, label="Hit@5")
axes[0].set_yticks(y, names, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("score")
axes[0].set_title("Retrieval ablations (test)")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_xlim(0, 1.0)

# Panel 2: per-stage latency P50 and P95 from exported spans.
lat = DATA["latency_by_stage"]
stage_order = (
    "retrieval", "fusion", "rerank", "threshold", "generation",
    "citation_validation", "serialization", "nli_scoring",
)
stages = [s for s in stage_order if s in lat]
x = np.arange(len(stages))
width = 0.38
p50 = [lat[s]["p50_ms"] for s in stages]
p95 = [lat[s]["p95_ms"] for s in stages]
axes[1].bar(x - width / 2, p50, width, color=COLORS[0], label="P50")
axes[1].bar(x + width / 2, p95, width, color=COLORS[2], label="P95")
axes[1].set_xticks(x, stages, rotation=35, ha="right", fontsize=8)
axes[1].set_ylabel("ms")
axes[1].set_title("Per-stage latency (from spans)")
axes[1].legend(fontsize=8)

# Panel 3: headline e2e rates with CI whiskers where available.
e2e = DATA["e2e"]
rate_keys = [
    ("required_point_recall", "Req-point recall"),
    ("all_points_covered", "All points covered"),
    ("abstain_f1", "Abstain F1"),
    ("citation_label_validity", "Citation validity"),
    ("citation_page_precision", "Citation page prec."),
    ("citation_hit_rate", "Citation hit rate"),
]
means, lows, highs, labels = [], [], [], []
for key, label in rate_keys:
    val = e2e.get(key)
    if isinstance(val, dict):
        means.append(val["mean"])
        lows.append(val["mean"] - val["ci_low"])
        highs.append(val["ci_high"] - val["mean"])
    elif isinstance(val, (int, float)):
        means.append(float(val))
        lows.append(0.0)
        highs.append(0.0)
    else:
        continue
    labels.append(label)
nli = e2e.get("nli_claim_support_rate")
if nli is not None:
    means.append(float(nli))
    lows.append(0.0)
    highs.append(0.0)
    labels.append("NLI support (est.)")
y2 = np.arange(len(labels))
axes[2].barh(y2, means, color=COLORS[1])
axes[2].errorbar(
    means,
    y2,
    xerr=[lows, highs],
    fmt="none",
    ecolor="#1f2937",
    capsize=3,
    lw=1.2,
)
axes[2].set_yticks(y2, labels, fontsize=8)
axes[2].invert_yaxis()
axes[2].set_xlim(0, 1.05)
axes[2].set_xlabel("rate")
axes[2].set_title("End-to-end quality (test)")

fig.suptitle(
    f"NIST RAG benchmark - {DATA.get('source', '')}",
    fontsize=12,
    fontweight="bold",
)
fig.savefig(ROOT / "image.png", dpi=180, bbox_inches="tight")
print(f"wrote {ROOT / 'image.png'}")
