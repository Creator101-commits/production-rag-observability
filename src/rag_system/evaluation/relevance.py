"""Chunk relevance judgments and retrieval metrics for the QA benchmark.

A chunk is relevant to a question when it shares a document with an
evidence entry and its page range overlaps that entry's pages. Page
numbers are 1-based physical PDF page indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rag_system.evaluation.dataset import EvalQuestion
from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.schema import Evidence


@dataclass(frozen=True)
class RetrievalScores:
    """Per-question retrieval metrics at standard cutoffs."""

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    ndcg_at_10: float


def chunk_is_relevant(chunk: Chunk, question: EvalQuestion) -> bool:
    """True when the chunk's doc+page range overlaps a judged evidence entry."""
    for ref in question.evidence:
        if ref.document_id != chunk.document_id:
            continue
        if any(chunk.page_start <= p <= chunk.page_end for p in ref.pages):
            return True
    return False


def evidence_is_relevant(evidence: Evidence, question: EvalQuestion) -> bool:
    """Same relevance rule applied to retrieved Evidence items."""
    for ref in question.evidence:
        if ref.document_id != evidence.document_id:
            continue
        if any(evidence.page_start <= p <= evidence.page_end for p in ref.pages):
            return True
    return False


def relevant_chunk_ids(
    chunks: list[Chunk], question: EvalQuestion
) -> set[str]:
    return {c.chunk_id for c in chunks if chunk_is_relevant(c, question)}


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def retrieval_scores(
    ranked_chunk_ids: list[str], relevant: set[str]
) -> RetrievalScores:
    """Compute Hit@k, MRR@10, and binary nDCG@10 for one ranking."""
    top = ranked_chunk_ids[:10]

    def hit(k: int) -> float:
        return 1.0 if any(cid in relevant for cid in top[:k]) else 0.0

    mrr = 0.0
    for rank, cid in enumerate(top, start=1):
        if cid in relevant:
            mrr = 1.0 / rank
            break

    gains = [1.0 if cid in relevant else 0.0 for cid in top]
    ideal_count = min(len(relevant), 10)
    ideal = _dcg([1.0] * ideal_count)
    ndcg = _dcg(gains) / ideal if ideal > 0 else 0.0

    return RetrievalScores(
        hit_at_1=hit(1),
        hit_at_3=hit(3),
        hit_at_5=hit(5),
        hit_at_10=hit(10),
        mrr_at_10=mrr,
        ndcg_at_10=ndcg,
    )


def mean_retrieval_scores(rows: list[RetrievalScores]) -> dict[str, float]:
    """Average per-question retrieval scores into a metrics dict."""
    if not rows:
        return {}
    keys = [
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]
    return {
        key: float(np.mean([getattr(r, key) for r in rows])) for key in keys
    }


def bootstrap_ci(
    values: list[float], n_resamples: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)
