"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rag_system.evaluation.dataset import load_evaluation_dataset
from rag_system.evaluation.runner import BenchmarkRunner
from rag_system.reranking.cross_encoder import CrossEncoderReranker

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


questions = [q for q in load_evaluation_dataset() if q.split == "test"]
runner = BenchmarkRunner()
metrics = runner.run_evaluation(questions)
labels = ["Doc recall@5", "Doc MRR", "Citation ID validity", "False answer rate"]
values = [
    metrics.retrieval_recall_at_5,
    metrics.retrieval_mrr,
    metrics.citation_validity_rate,
    metrics.false_answer_rate,
]
axes[0].bar(range(4), values, color=COLORS[:4])
axes[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 1.1), title="Measured fixture evaluation")
axes[0].tick_params(axis="x", rotation=28)
stages = ["retrieval", "reranking", "generation", "citation_validation"]
timings = [
    float(
        np.mean(
            [
                s.duration_ms
                for t in runner.tracer.completed_traces
                for s in t.spans
                if s.name == name
            ]
        )
    )
    for name in stages
]
axes[1].barh(stages, timings, color=COLORS[:4])
axes[1].invert_yaxis()
axes[1].set(xlabel="Mean duration (ms), this CPU run", title="In-memory stage timing")
query = "What is the primary role of the GOVERN core function?"
candidates = runner.retriever.retrieve(query, top_k=5)
reranked, _ = CrossEncoderReranker().rerank(query, candidates, top_k=5)
axes[2].bar(range(len(reranked)), [e.score for e in reranked], color=COLORS[0])
axes[2].axhline(0.3, ls="--", color=COLORS[2], label="Evidence threshold 0.30")
axes[2].set(
    xticks=range(len(reranked)),
    xticklabels=[f"Rank {e.rank}" for e in reranked],
    ylim=(0, 1),
    ylabel="Lexical coverage score",
    title="GOVERN query: evidence gate",
)
axes[2].legend(fontsize=9)
fig.suptitle("RAG observability | deterministic fixture service", fontsize=16)
fig.supxlabel(
    "Seven synthetic test questions; teaching paraphrases; hashed retrieval and "
    "lexical reranking. Citation IDs do not prove claim support.",
    fontsize=10,
)
payload = {
    "source": "teaching fixtures, test split only; deterministic fake generator",
    "metrics": dict(zip(labels, values, strict=True)),
    "mean_stage_ms": dict(zip(stages, timings, strict=True)),
    "evidence_scores": [e.score for e in reranked],
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
