"""Dev-tuned, frozen-config QA benchmark over the NIST eval set.

Pipeline: ingest pinned PDFs, chunk at candidate token sizes, tune chunk
size, retrieval system, reranker, and abstention threshold on dev only,
freeze the configuration, then evaluate test once. All model-touching
components are injected so tests run without downloads.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag_system.citations.validator import CitationValidator
from rag_system.evaluation.answers import (
    abstention_metrics,
    required_point_recall,
)
from rag_system.evaluation.dataset import EvalQuestion
from rag_system.evaluation.nli import (
    NLIScorer,
    claim_supported,
    sentence_windows,
)
from rag_system.evaluation.relevance import (
    RetrievalScores,
    bootstrap_ci,
    chunk_is_relevant,
    mean_retrieval_scores,
    retrieval_scores,
)
from rag_system.evaluation.schema_guard import count_schema_bound_violations
from rag_system.generation.base import GenerationRequest
from rag_system.generation.resolve import resolve_generation_response
from rag_system.ingestion.chunk_tokens import chunk_document_tokens
from rag_system.ingestion.schema import Chunk, Document
from rag_system.ingestion.tokens import Tokenizer
from rag_system.observability.tracing import (
    JsonlSpanExporter,
    OTelTracer,
    fingerprint_query,
)
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.reranking.scorer import PairScorer
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.hybrid import HybridRetriever
from rag_system.retrieval.schema import Evidence
from rag_system.retrieval.sentence_transformer import (
    SentenceTransformerRetriever,
    TextEncoder,
)

CHUNK_SIZE_CANDIDATES = (256, 512, 768)
OVERLAP_RATIO = 0.15
RETRIEVAL_TOP_K = 10
CANDIDATE_DEPTH = 20
EVIDENCE_TOP_K = 5
RERANK_DEPTH = 20
SEED = 42

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class RetrievalSpec:
    """One retrieval system configuration."""

    name: str
    kind: str  # "bm25" | "dense" | "hybrid"
    rrf_k: float = 60.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


@dataclass(frozen=True)
class FrozenConfig:
    """The single configuration evaluated once on test."""

    chunk_tokens: int
    overlap_ratio: float
    retrieval: RetrievalSpec
    rerank: str  # "none" | "cross_encoder"
    abstain_threshold: float | None
    candidate_depth: int = CANDIDATE_DEPTH
    retrieval_top_k: int = RETRIEVAL_TOP_K
    evidence_top_k: int = EVIDENCE_TOP_K
    rerank_depth: int = RERANK_DEPTH
    seed: int = SEED


@dataclass
class Components:
    """Injected model/tokenizer/provider dependencies for one run."""

    tokenizer: Tokenizer
    encoder: TextEncoder | None = None
    query_prefix: str = ""
    pair_scorer: PairScorer | None = None
    nli_scorer: NLIScorer | None = None
    provider: Any = None  # GenerationProvider
    provider_name: str = "fake"
    provider_model_id: str = "deterministic-rag-generator-v1"
    dense_model_id: str = ""
    dense_revision: str = ""
    reranker_model_id: str = ""
    reranker_revision: str = ""
    nli_model_id: str = ""
    nli_revision: str = ""
    # Generator candidates as JSON-serializable provider specs (name ->
    # spec dict). Non-empty selects the staged subprocess pipeline so each
    # candidate runs in its own process; empty uses the single ``provider``.
    provider_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    k_ctx_candidates: tuple[int, ...] = ()
    # Factories to rebuild retrieval models for the stage-C ablation pass
    # after they are unloaded for the generator subprocesses.
    encoder_factory: Callable[[], Any] | None = None
    pair_scorer_factory: Callable[[], Any] | None = None
    max_context_tokens: int = 2048
    temperature: float = 0.0
    seed: int = SEED
    extra: dict[str, Any] = field(default_factory=dict)


def retrieval_grid() -> list[RetrievalSpec]:
    """Candidate retrieval systems tuned on dev."""
    specs = [
        RetrievalSpec(name="bm25", kind="bm25"),
        RetrievalSpec(name="dense", kind="dense"),
    ]
    for k in (10.0, 30.0, 60.0):
        for bw, dw in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0)):
            specs.append(
                RetrievalSpec(
                    name=f"hybrid-rrf-k{int(k)}-w{bw:g}-{dw:g}",
                    kind="hybrid",
                    rrf_k=k,
                    bm25_weight=bw,
                    dense_weight=dw,
                )
            )
    return specs


def build_retriever(
    spec: RetrievalSpec,
    chunks: list[Chunk],
    comps: Components,
    *,
    preindexed: dict[str, Any] | None = None,
) -> Any:
    """Instantiate and index one retrieval system."""
    if spec.kind == "bm25":
        return BM25Retriever().index(chunks)
    if spec.kind == "dense":
        if comps.encoder is None:
            raise ValueError("dense retrieval requires an encoder")
        retriever = SentenceTransformerRetriever(
            comps.encoder,
            query_prefix=comps.query_prefix,
            retriever_name=spec.name,
        )
        return retriever.index(chunks)
    # hybrid: reuse pre-indexed arms when supplied to avoid re-encoding.
    if preindexed is not None:
        bm25 = preindexed["bm25"]
        dense = preindexed["dense"]
        hybrid = HybridRetriever(
            rrf_k=spec.rrf_k,
            bm25_weight=spec.bm25_weight,
            dense_weight=spec.dense_weight,
            bm25=bm25,
            dense=dense,
        )
        hybrid.chunk_map = {c.chunk_id: c for c in chunks}
        hybrid.is_indexed = True
        return hybrid
    if comps.encoder is None:
        raise ValueError("hybrid retrieval requires an encoder")
    return HybridRetriever(
        rrf_k=spec.rrf_k,
        bm25_weight=spec.bm25_weight,
        dense_weight=spec.dense_weight,
        bm25=BM25Retriever(),
        dense=SentenceTransformerRetriever(
            comps.encoder,
            query_prefix=comps.query_prefix,
            retriever_name="dense",
        ),
    ).index(chunks)


def eval_retrieval(
    retriever: Any,
    questions: list[EvalQuestion],
    chunks: list[Chunk],
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[list[RetrievalScores], list[dict[str, Any]]]:
    """Score retrieval quality over answerable questions."""
    scores: list[RetrievalScores] = []
    records: list[dict[str, Any]] = []
    for q in questions:
        if not q.answerable:
            continue
        ranked = retriever.retrieve(q.question, top_k=top_k)
        relevant = {c.chunk_id for c in chunks if chunk_is_relevant(c, q)}
        row = retrieval_scores([e.chunk_id for e in ranked], relevant)
        scores.append(row)
        records.append(
            {
                "question_id": q.question_id,
                "hit_at_1": row.hit_at_1,
                "hit_at_3": row.hit_at_3,
                "hit_at_5": row.hit_at_5,
                "hit_at_10": row.hit_at_10,
                "mrr_at_10": row.mrr_at_10,
                "ndcg_at_10": row.ndcg_at_10,
            }
        )
    return scores, records


def mean_context_tokens(chunks: list[Chunk]) -> float:
    return float(np.mean([c.token_count for c in chunks])) if chunks else 0.0


def chunk_all_docs(
    docs: list[Document], comps: Components, target_tokens: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document_tokens(
                doc,
                comps.tokenizer,
                target_tokens=target_tokens,
                overlap_ratio=OVERLAP_RATIO,
            )
        )
    return chunks


def select_chunk_size(
    docs: list[Document],
    dev_questions: list[EvalQuestion],
    comps: Components,
) -> tuple[int, list[dict[str, Any]], dict[int, list[Chunk]]]:
    """Pick chunk size on dev by Hit@5, tie-broken by mean context tokens."""
    table: list[dict[str, Any]] = []
    chunks_by_size: dict[int, list[Chunk]] = {}
    for size in CHUNK_SIZE_CANDIDATES:
        chunks = chunk_all_docs(docs, comps, size)
        chunks_by_size[size] = chunks
        retriever = build_retriever(
            RetrievalSpec(name="hybrid-default", kind="hybrid", rrf_k=60.0),
            chunks,
            comps,
        )
        scores, _ = eval_retrieval(retriever, dev_questions, chunks)
        means = mean_retrieval_scores(scores)
        table.append(
            {
                "chunk_tokens": size,
                "hit_at_5": means.get("hit_at_5", 0.0),
                "ndcg_at_10": means.get("ndcg_at_10", 0.0),
                "mean_context_tokens": mean_context_tokens(chunks),
                "num_chunks": len(chunks),
            }
        )
    best = max(
        table,
        key=lambda row: (row["hit_at_5"], -row["mean_context_tokens"]),
    )
    return int(best["chunk_tokens"]), table, chunks_by_size


def tune_retrieval(
    chunks: list[Chunk],
    dev_questions: list[EvalQuestion],
    comps: Components,
) -> tuple[RetrievalSpec, list[dict[str, Any]]]:
    """Tune retrieval system on dev by nDCG@10."""
    if comps.encoder is not None:
        preindexed = {
            "bm25": BM25Retriever().index(chunks),
            "dense": SentenceTransformerRetriever(
                comps.encoder,
                query_prefix=comps.query_prefix,
                retriever_name="dense",
            ).index(chunks),
        }
    else:
        preindexed = {"bm25": BM25Retriever().index(chunks)}

    table: list[dict[str, Any]] = []
    best_spec: RetrievalSpec | None = None
    best_ndcg = -1.0
    for spec in retrieval_grid():
        if spec.kind != "bm25" and comps.encoder is None:
            continue
        retriever = build_retriever(spec, chunks, comps, preindexed=preindexed)
        scores, _ = eval_retrieval(retriever, dev_questions, chunks)
        means = mean_retrieval_scores(scores)
        ndcg = means.get("ndcg_at_10", 0.0)
        table.append(
            {
                "system": spec.name,
                "kind": spec.kind,
                "rrf_k": spec.rrf_k,
                "bm25_weight": spec.bm25_weight,
                "dense_weight": spec.dense_weight,
                **means,
            }
        )
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_spec = spec
    if best_spec is None:
        raise RuntimeError("no retrieval system could be evaluated")
    return best_spec, table


def rerank_candidates(
    evidence: list[Evidence],
    query: str,
    reranker: CrossEncoderReranker | None,
    top_k: int,
) -> tuple[list[Evidence], float]:
    """Apply optional cross-encoder rerank; return (ranked, rerank_ms)."""
    if reranker is None:
        return evidence[:top_k], 0.0
    return reranker.rerank(query, evidence, top_k=top_k)


def tune_rerank(
    chunks: list[Chunk],
    dev_questions: list[EvalQuestion],
    comps: Components,
    spec: RetrievalSpec,
) -> tuple[str, list[dict[str, Any]]]:
    """Choose {none, cross_encoder@20} on dev by nDCG@10."""
    retriever = build_retriever(spec, chunks, comps)
    table: list[dict[str, Any]] = []
    best_name = "none"
    best_ndcg = -1.0
    for option in ("none", "cross_encoder"):
        reranker = (
            CrossEncoderReranker(comps.pair_scorer, max_candidates=RERANK_DEPTH)
            if option == "cross_encoder" and comps.pair_scorer is not None
            else None
        )
        if option == "cross_encoder" and reranker is None:
            continue
        rows: list[RetrievalScores] = []
        for q in dev_questions:
            if not q.answerable:
                continue
            cands = retriever.retrieve(q.question, top_k=CANDIDATE_DEPTH)
            ranked, _ = rerank_candidates(cands, q.question, reranker, RETRIEVAL_TOP_K)
            relevant = {c.chunk_id for c in chunks if chunk_is_relevant(c, q)}
            rows.append(retrieval_scores([e.chunk_id for e in ranked], relevant))
        means = mean_retrieval_scores(rows)
        ndcg = means.get("ndcg_at_10", 0.0)
        table.append({"rerank": option, "depth": RERANK_DEPTH if option != "none" else 0, **means})
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_name = option
    return best_name, table


def _retrieve_ranked(
    q: EvalQuestion,
    retriever: Any,
    reranker: CrossEncoderReranker | None,
    cfg: FrozenConfig,
    tracer: OTelTracer,
) -> tuple[list[Any], float]:
    """Run retrieval, fusion and rerank spans; return (ranked, rerank_ms)."""
    # Hybrid retrievers expose separable arms so fusion gets its own span.
    if isinstance(retriever, HybridRetriever):
        with tracer.span("retrieval") as ret_span:
            bm25_cands, dense_cands = retriever.candidate_lists(
                q.question, candidate_depth=cfg.candidate_depth
            )
            ret_span.set_attribute(
                "candidate_count", len(bm25_cands) + len(dense_cands)
            )
        with tracer.span("fusion") as fus_span:
            candidates = retriever.fuse(
                bm25_cands, dense_cands, cfg.candidate_depth
            )
            fus_span.set_attribute("fused_count", len(candidates))
    else:
        with tracer.span("retrieval") as ret_span:
            candidates = retriever.retrieve(q.question, top_k=cfg.candidate_depth)
            ret_span.set_attribute("candidate_count", len(candidates))

    with tracer.span("rerank") as rr_span:
        ranked, rerank_ms = rerank_candidates(
            candidates, q.question, reranker, cfg.retrieval_top_k
        )
        rr_span.set_attribute("rerank_ms", round(rerank_ms, 3))
        rr_span.set_attribute("enabled", reranker is not None)
    return ranked, rerank_ms


def run_pipeline_question(
    q: EvalQuestion,
    retriever: Any,
    reranker: CrossEncoderReranker | None,
    provider: Any,
    comps: Components,
    cfg: FrozenConfig,
    chunk_map: dict[str, Chunk],
    tracer: OTelTracer,
    *,
    abstain_threshold: float | None,
    record_nli: bool = False,
) -> dict[str, Any]:
    """Run one question through the full pipeline with OTel spans."""
    query_sha, query_len = fingerprint_query(q.question)
    record: dict[str, Any] = {
        "question_id": q.question_id,
        "split": q.split,
        "category": q.category,
        "difficulty": q.difficulty,
        "answerable": q.answerable,
    }

    with tracer.span(
        "request",
        {"query_sha256": query_sha, "query_length": query_len},
    ):
        ranked, rerank_ms = _retrieve_ranked(q, retriever, reranker, cfg, tracer)

        top_score = ranked[0].score if ranked else 0.0
        evidence = ranked[: cfg.evidence_top_k]
        record["top_score"] = float(top_score)
        record["retrieved_chunk_ids"] = [e.chunk_id for e in ranked]

        threshold_abstain = (
            abstain_threshold is not None
            and (not ranked or top_score < abstain_threshold)
        )

        with tracer.span(
            "threshold",
            {
                "top_score": round(float(top_score), 6),
                "threshold": abstain_threshold,
                "abstained": threshold_abstain,
            },
        ):
            pass

        if threshold_abstain:
            record.update(
                status="abstained",
                abstain_source="threshold",
                abstain_reason="evidence_score_threshold",
                answer="",
                claims=[],
                citations=[],
                prompt_tokens=0,
                completion_tokens=0,
                hit_token_cap=False,
                parse_failure=False,
                latency_ms=0.0,
                rerank_ms=rerank_ms,
            )
        else:
            gen_req = GenerationRequest(
                query=q.question,
                evidence_list=evidence,
                temperature=comps.temperature,
                # Pipeline threshold is authoritative; disable provider-side
                # score filtering so only threshold + model abstain apply.
                evidence_score_threshold=-1e9,
                max_context_tokens=comps.max_context_tokens,
            )
            with tracer.span("generation") as gen_span:
                raw_resp = provider.generate(gen_req)
                gen_span.set_attribute("status", raw_resp.status)
                gen_resp, resolve_warnings = resolve_generation_response(
                    raw_resp, evidence
                )
                raw_label_total = sum(
                    len(c.citation_labels) for c in raw_resp.claims
                )
                resolved_label_total = sum(
                    len(c.chunk_ids) for c in gen_resp.claims
                )
                gen_resp = CitationValidator.enforce(
                    gen_resp, evidence, minimum_citations=1
                )
                gen_span.set_attribute("prompt_tokens", gen_resp.prompt_tokens)
                gen_span.set_attribute(
                    "completion_tokens", gen_resp.completion_tokens
                )
                record["resolve_warnings"] = list(resolve_warnings)
                record["citation_label_validity"] = (
                    resolved_label_total / raw_label_total
                    if raw_label_total
                    else None
                )

            with tracer.span("citation_validation") as val_span:
                val_result = CitationValidator.validate(gen_resp.answer, evidence)
                val_span.set_attribute("citation_count", len(val_result.resolved_citations))
                val_span.set_attribute("valid", val_result.is_valid)

            max_new_tokens = getattr(provider, "max_new_tokens", None)
            hit_token_cap = bool(
                max_new_tokens is not None
                and gen_resp.completion_tokens >= max_new_tokens
            )
            reason = (gen_resp.abstain_reason or "").lower()
            if gen_resp.status == "abstained":
                if gen_resp.parse_failure:
                    abstain_source = (
                        "truncated" if hit_token_cap else "parse_failure"
                    )
                elif "citation" in reason or "valid" in reason:
                    abstain_source = "citation_validation"
                else:
                    abstain_source = "model_abstain"
            else:
                abstain_source = ""
            record.update(
                status=gen_resp.status,
                abstain_source=abstain_source,
                abstain_reason=gen_resp.abstain_reason,
                answer=gen_resp.answer,
                raw_output=raw_resp.raw_output,
                claims=[
                    {"text": c.text, "chunk_ids": list(c.chunk_ids)}
                    for c in gen_resp.claims
                ],
                citations=list(gen_resp.citations),
                prompt_tokens=gen_resp.prompt_tokens,
                completion_tokens=gen_resp.completion_tokens,
                hit_token_cap=hit_token_cap,
                parse_failure=gen_resp.parse_failure,
                latency_ms=gen_resp.latency_ms,
                rerank_ms=rerank_ms,
            )

            # Citation page precision / hit rate against judged evidence.
            with tracer.span("serialization"):
                cited_ids = list(dict.fromkeys(gen_resp.citations))
                cited_chunks = [
                    chunk_map[cid] for cid in cited_ids if cid in chunk_map
                ]
                relevant_cited = [
                    c for c in cited_chunks if chunk_is_relevant(c, q)
                ]
                record["citation_page_precision"] = (
                    len(relevant_cited) / len(cited_chunks) if cited_chunks else None
                )
                record["citation_hit"] = 1.0 if relevant_cited else 0.0

                # NLI claim-support estimate (model-based). Premises are
                # ~3-sentence windows so long chunks are not truncated;
                # a claim is supported if any window entails. Whole-chunk
                # scoring is kept as a secondary field.
                if record_nli and comps.nli_scorer is not None and gen_resp.claims:
                    window_pairs: list[tuple[str, str]] = []
                    whole_pairs: list[tuple[str, str]] = []
                    pair_claim_idx: list[int] = []
                    pair_is_window: list[bool] = []
                    for ci, claim in enumerate(gen_resp.claims):
                        for cid in claim.chunk_ids:
                            chunk = chunk_map.get(cid)
                            if chunk is None:
                                continue
                            for window in sentence_windows(chunk.text):
                                window_pairs.append((window, claim.text))
                                pair_claim_idx.append(ci)
                                pair_is_window.append(True)
                            whole_pairs.append((chunk.text, claim.text))
                    all_pairs = window_pairs + whole_pairs
                    pair_claim_idx_all = pair_claim_idx + [
                        ci
                        for ci, claim in enumerate(gen_resp.claims)
                        for cid in claim.chunk_ids
                        if cid in chunk_map
                    ]
                    pair_is_window_all = pair_is_window + [False] * len(
                        whole_pairs
                    )
                    with tracer.span("nli_scoring") as nli_span:
                        labels = comps.nli_scorer.predict_labels(all_pairs)
                        nli_span.set_attribute("pair_count", len(all_pairs))
                    supported = [False] * len(gen_resp.claims)
                    whole_supported = [False] * len(gen_resp.claims)
                    for (_premise, _hyp), label, ci, is_win in zip(
                        all_pairs,
                        labels,
                        pair_claim_idx_all,
                        pair_is_window_all,
                        strict=True,
                    ):
                        if claim_supported(label):
                            if is_win:
                                supported[ci] = True
                            else:
                                whole_supported[ci] = True
                    record["nli_supported_claims"] = sum(supported)
                    record["nli_total_claims"] = len(gen_resp.claims)
                    record["nli_whole_chunk_supported_claims"] = sum(
                        whole_supported
                    )

    # Answer-quality metrics: match required points against the answer
    # text plus every claim text.
    if q.answerable:
        claim_texts = " ".join(c.get("text", "") for c in record.get("claims", []))
        haystack = f"{record.get('answer', '')} {claim_texts}"
        recall, covered, total, all_cov = required_point_recall(
            haystack, [list(p) for p in q.required_points]
        )
        record["required_point_recall"] = recall if record["status"] == "answered" else 0.0
        record["points_covered"] = covered
        record["points_total"] = total
        record["all_points_covered"] = all_cov if record["status"] == "answered" else False
    return record


def _candidate_thresholds(records: list[dict[str, Any]]) -> list[float]:
    """Midpoints between sorted unique top scores, plus extremes."""
    scores = sorted({r["top_score"] for r in records if "top_score" in r})
    if not scores:
        return [-1e9]
    cands = [scores[0] - 1e-6]
    for a, b in zip(scores, scores[1:], strict=False):
        cands.append((a + b) / 2.0)
    cands.append(scores[-1] + 1e-6)
    return cands


def tune_abstention(
    dev_records: list[dict[str, Any]],
) -> tuple[float | None, str, list[dict[str, Any]]]:
    """Tune the code-level threshold on dev records.

    Objective: maximize answerable-answer rate subject to abstaining on
    all dev unanswerables; fall back to max abstention F1 and say so.
    """
    table: list[dict[str, Any]] = []
    best: tuple[float, float, float | None] | None = None  # (rate, f1, t)
    best_f1_row: tuple[float, float, float | None] | None = None

    for t in _candidate_thresholds(dev_records):
        rows = [
            (
                bool(r["answerable"]),
                bool(r["top_score"] < t or r["status"] != "answered"),
            )
            for r in dev_records
        ]
        m = abstention_metrics(rows)
        table.append({"threshold": t, **m})
        unans_abstain = 1.0 - m["false_answer_rate"]
        feasible = unans_abstain >= 1.0 - 1e-12
        if feasible and (
            best is None
            or m["answerable_answer_rate"] > best[0]
            or (
                m["answerable_answer_rate"] == best[0]
                and m["abstain_f1"] > best[1]
            )
        ):
            best = (m["answerable_answer_rate"], m["abstain_f1"], t)
        if best_f1_row is None or m["abstain_f1"] > best_f1_row[0]:
            best_f1_row = (m["abstain_f1"], m["answerable_answer_rate"], t)

    if best is not None:
        return best[2], "max_answerable_rate_s.t._all_dev_unanswerable_abstain", table
    assert best_f1_row is not None
    return best_f1_row[2], "max_abstain_f1_infeasible", table


def aggregate_e2e_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute headline e2e metrics with bootstrap CIs from records."""
    answerable = [r for r in records if r["answerable"]]
    abst_rows = [(bool(r["answerable"]), r["status"] != "answered") for r in records]
    abs_m = abstention_metrics(abst_rows)

    recalls = [r["required_point_recall"] for r in answerable]
    all_cov = [1.0 if r.get("all_points_covered") else 0.0 for r in answerable]
    rec_mean, rec_lo, rec_hi = bootstrap_ci(recalls)
    cov_mean, cov_lo, cov_hi = bootstrap_ci(all_cov)

    answered = [r for r in records if r["status"] == "answered"]
    validity_vals = [
        r["citation_label_validity"]
        for r in answered
        if r.get("citation_label_validity") is not None
    ]
    page_prec_vals = [
        r["citation_page_precision"]
        for r in answered
        if r.get("citation_page_precision") is not None
    ]
    hit_vals = [r.get("citation_hit", 0.0) for r in answered]

    def _ci(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        mean, lo, hi = bootstrap_ci(vals)
        return {"mean": mean, "ci_low": lo, "ci_high": hi}

    nli_supported = sum(r.get("nli_supported_claims", 0) for r in answered)
    nli_whole = sum(
        r.get("nli_whole_chunk_supported_claims", 0) for r in answered
    )
    nli_total = sum(r.get("nli_total_claims", 0) for r in answered)

    abstain_sources: dict[str, int] = {}
    for r in records:
        src = r.get("abstain_source")
        if src:
            abstain_sources[src] = abstain_sources.get(src, 0) + 1

    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in records)

    return {
        "num_questions": len(records),
        "num_answerable": len(answerable),
        "num_unanswerable": len(records) - len(answerable),
        "required_point_recall": {"mean": rec_mean, "ci_low": rec_lo, "ci_high": rec_hi},
        "all_points_covered": {"mean": cov_mean, "ci_low": cov_lo, "ci_high": cov_hi},
        **abs_m,
        "citation_label_validity": _ci(validity_vals),
        "citation_page_precision": _ci(page_prec_vals),
        "citation_hit_rate": _ci(hit_vals),
        "nli_claim_support_rate": (
            nli_supported / nli_total if nli_total else None
        ),
        "nli_supported_claims": nli_supported,
        "nli_total_claims": nli_total,
        "nli_whole_chunk_support_rate": (
            nli_whole / nli_total if nli_total else None
        ),
        "abstention_sources": abstain_sources,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "mean_prompt_tokens": prompt_tokens / len(records) if records else 0.0,
        "mean_completion_tokens": (
            completion_tokens / len(records) if records else 0.0
        ),
        "parse_failures": sum(1 for r in records if r.get("parse_failure")),
        "token_cap_hits": sum(
            1 for r in records if r.get("hit_token_cap")
        ),
        "truncation_rate": (
            sum(1 for r in records if r.get("hit_token_cap"))
            / len(records)
            if records
            else 0.0
        ),
    }


def stage_latency_table(traces_path: Path) -> dict[str, dict[str, float]]:
    """Per-stage P50/P95 computed from the exported JSONL spans."""
    by_stage: dict[str, list[float]] = {}
    if not traces_path.is_file():
        return {}
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_stage.setdefault(rec["name"], []).append(float(rec["duration_ms"]))
    table: dict[str, dict[str, float]] = {}
    for name, durations in by_stage.items():
        arr = np.asarray(durations)
        table[name] = {
            "count": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
        }
    return table


def gpu_seconds_from_spans(traces_path: Path) -> float:
    """Sum durations of GPU-heavy stages (rerank, generation, nli)."""
    gpu_stages = {"rerank", "generation", "nli_scoring"}
    total = 0.0
    if not traces_path.is_file():
        return 0.0
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["name"] in gpu_stages:
            total += float(rec["duration_ms"])
    return total / 1000.0


def render_report(result: dict[str, Any]) -> str:
    """Render a markdown report from benchmark metrics and manifest."""
    metrics = result["metrics"]
    manifest = result["manifest"]
    e2e = metrics["e2e_test"]
    frozen = manifest["frozen_config"]
    dev = manifest["dev_selection"]

    def _pct(metric: dict[str, float], scale: float = 100.0) -> str:
        lo = metric["ci_low"] * scale
        hi = metric["ci_high"] * scale
        return f"{metric['mean'] * scale:.1f}% [{lo:.1f}, {hi:.1f}]"

    lines = [
        "# NIST RAG Benchmark Report",
        "",
        f"- git SHA: `{manifest['git_sha']}`",
        f"- GPU: {manifest['gpu']}",
        f"- Frozen system: {frozen['retrieval']['name']} + "
        f"rerank={frozen['rerank']}@{frozen['rerank_depth']}, "
        f"chunk={frozen['chunk_tokens']} tokens, "
        f"provider={frozen['provider']}, k_ctx={frozen['evidence_top_k']}, "
        f"abstain_threshold={frozen['abstain_threshold']}",
        f"- Threshold score scale: {frozen['threshold_score_scale']}; "
        f"effectively inactive: {frozen['threshold_effectively_inactive']}",
        f"- Generator selection objective: {dev['generator_objective']}",
        "",
        "## End-to-end test metrics (frozen config, run once)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Required-point recall | {_pct(e2e['required_point_recall'])} |",
        f"| All points covered | {_pct(e2e['all_points_covered'])} |",
        f"| Abstain precision | {e2e['abstain_precision']:.3f} |",
        f"| Abstain recall | {e2e['abstain_recall']:.3f} |",
        f"| Abstain F1 | {e2e['abstain_f1']:.3f} |",
        f"| False-answer rate | {e2e['false_answer_rate']:.3f} |",
        f"| False-abstain rate | {e2e['false_abstain_rate']:.3f} |",
        f"| Answerable answer rate | {e2e['answerable_answer_rate']:.3f} |",
        f"| Citation label validity | {_pct(e2e['citation_label_validity'])} |",
        f"| Citation page precision | {_pct(e2e['citation_page_precision'])} |",
        f"| Citation hit rate | {_pct(e2e['citation_hit_rate'])} |",
        f"| NLI claim support, windowed (estimate) | "
        f"{(e2e['nli_claim_support_rate'] or 0.0) * 100:.1f}% "
        f"({e2e['nli_supported_claims']}/{e2e['nli_total_claims']}) |",
        f"| NLI claim support, whole-chunk (secondary) | "
        f"{(e2e.get('nli_whole_chunk_support_rate') or 0.0) * 100:.1f}% |",
        f"| Abstentions by source | {e2e.get('abstention_sources', {})} |",
        f"| Parse failures | {e2e['parse_failures']} |",
        f"| Schema bound violations | {e2e.get('schema_bound_violations', 0)} |",
        f"| Truncation rate | {e2e['truncation_rate'] * 100:.1f}% "
        f"({e2e['token_cap_hits']}/{e2e['num_questions']} hit cap) |",
        f"| Prompt tokens total | {e2e['prompt_tokens_total']} |",
        f"| Completion tokens total | {e2e['completion_tokens_total']} |",
        f"| GPU-seconds | {metrics['gpu_seconds']:.2f} |",
        "",
        "## Retrieval ablations (test, retrieval-only)",
        "",
        "| System | Hit@5 | MRR@10 | nDCG@10 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for name, scores in sorted(metrics["retrieval_ablations_test"].items()):
        lines.append(
            f"| {name} | {scores.get('hit_at_5', 0.0):.3f} | "
            f"{scores.get('mrr_at_10', 0.0):.3f} | "
            f"{scores.get('ndcg_at_10', 0.0):.3f} |"
        )
    lines += [
        "",
        "## Generator x context ablation (test)",
        "",
        "| Candidate | Req-point recall | Abstain F1 | False-abstain | "
        "Parse failures | Truncation | Schema violations |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    frozen_key = f"{frozen['provider']}__k{frozen['evidence_top_k']}"
    for name, ev in sorted(metrics.get("generator_ablations_test", {}).items()):
        tag = " (frozen)" if name == frozen_key else ""
        lines.append(
            f"| {name}{tag} | "
            f"{(ev['required_point_recall']['mean']):.3f} | "
            f"{ev['abstain_f1']:.3f} | {ev['false_abstain_rate']:.3f} | "
            f"{ev['parse_failures']} | {ev.get('truncation_rate', 0.0):.3f} | "
            f"{ev.get('schema_bound_violations', 0)} |"
        )
    lines += [
        "",
        "## Per-stage latency (from exported spans)",
        "",
        "| Stage | Count | P50 ms | P95 ms | Mean ms |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for stage, row in sorted(metrics["latency_by_stage"].items()):
        lines.append(
            f"| {stage} | {row['count']} | {row['p50_ms']:.2f} | "
            f"{row['p95_ms']:.2f} | {row['mean_ms']:.2f} |"
        )
    lines += [
        "",
        "## Dev selection",
        "",
        "### Chunk size",
        "",
        "| Tokens | Hit@5 | nDCG@10 | Mean ctx tokens | Chunks |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dev["chunk_size_table"]:
        lines.append(
            f"| {row['chunk_tokens']} | {row['hit_at_5']:.3f} | "
            f"{row['ndcg_at_10']:.3f} | {row['mean_context_tokens']:.1f} | "
            f"{row['num_chunks']} |"
        )
    lines += [
        "",
        "### Retrieval system",
        "",
        "| System | nDCG@10 | Hit@5 | MRR@10 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for row in dev["retrieval_table"]:
        lines.append(
            f"| {row['system']} | {row.get('ndcg_at_10', 0.0):.3f} | "
            f"{row.get('hit_at_5', 0.0):.3f} | {row.get('mrr_at_10', 0.0):.3f} |"
        )
    lines += [
        "",
        "### Rerank",
        "",
        "| Option | nDCG@10 | Hit@5 |",
        "| :--- | ---: | ---: |",
    ]
    for row in dev["rerank_table"]:
        lines.append(
            f"| {row['rerank']} | {row.get('ndcg_at_10', 0.0):.3f} | "
            f"{row.get('hit_at_5', 0.0):.3f} |"
        )
    lines += [
        "",
        "### Generator x context",
        "",
        "| Provider | k_ctx | Dev req recall | Dev abstain F1 | "
        "All dev unans abstained |",
        "| :--- | ---: | ---: | ---: | :--- |",
    ]
    for row in dev["generator_table"]:
        lines.append(
            f"| {row['provider']} | {row['k_ctx']} | "
            f"{row['dev_required_point_recall']:.3f} | "
            f"{row['dev_abstain_f1']:.3f} | "
            f"{row['dev_all_unanswerables_abstained']} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- 70-question set authored by a single person; treat as a smoke",
        "  benchmark, not a broad capability measure.",
        "- Required-point scoring is lexical after normalization; paraphrases",
        "  outside the acceptable-alternative lists count as misses.",
        "- NLI claim support is a model-based estimate, not verified entailment.",
        "",
    ]
    return "\n".join(lines)


def _apply_threshold_to_records(
    records: list[dict[str, Any]], threshold: float | None
) -> list[dict[str, Any]]:
    """Copy records with the evidence threshold applied on stored scores.

    Model-side abstentions are already final; the threshold only converts
    would-be answers whose top score falls below it.
    """
    out: list[dict[str, Any]] = []
    for r in records:
        rr = dict(r)
        if (
            threshold is not None
            and rr.get("status") == "answered"
            and float(rr.get("top_score") or 0.0) < threshold
        ):
            rr["status"] = "abstained"
            rr["abstain_source"] = "threshold"
            rr["abstain_reason"] = "evidence_score_threshold"
            rr["required_point_recall"] = 0.0
            rr["all_points_covered"] = False
        out.append(rr)
    return out


def _empty_cuda_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _score_scale_desc(frozen: FrozenConfig) -> str:
    """Describe the score the abstention threshold applies to."""
    if frozen.rerank == "cross_encoder":
        return "cross-encoder relevance score (post-rerank top score)"
    if frozen.retrieval.kind == "hybrid":
        return "weighted RRF fused score"
    if frozen.retrieval.kind == "dense":
        return "dense cosine similarity"
    return "BM25 score"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "rag-system",
        "numpy",
        "scipy",
        "pandas",
        "pypdf",
        "fastapi",
        "uvicorn",
        "httpx",
        "opentelemetry-sdk",
        "sentence-transformers",
        "torch",
        "transformers",
        "faiss-cpu",
        "lm-format-enforcer",
        "bitsandbytes",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return versions


def _gpu_name() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _log_resources(tag: str) -> None:
    """Print RAM/disk diagnostics at stage boundaries (Kaggle debugging)."""
    try:
        mem_available_mb = -1
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    mem_available_mb = int(line.split()[1]) // 1024
                    break
        disk_free_gb = shutil.disk_usage("/").free / 1e9
        work_free_gb = (
            shutil.disk_usage("/kaggle/working").free / 1e9
            if Path("/kaggle/working").is_dir()
            else -1.0
        )
        hf_cache = os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        print(
            f"[resources] {tag}: mem_available_mb={mem_available_mb} "
            f"disk_free_gb={disk_free_gb:.1f} work_free_gb={work_free_gb:.1f} "
            f"hf_cache={hf_cache}",
            flush=True,
        )
    except Exception as exc:  # diagnostics must never kill the run
        print(f"[resources] {tag}: unavailable ({type(exc).__name__})", flush=True)


def _free_model_cache(model_id: str) -> None:
    """Drop a model's HF disk cache; large checkpoints fill the worker disk."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
    except ImportError:
        HF_HUB_CACHE = str(Path.home() / ".cache" / "huggingface" / "hub")
    folder = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
        print(f"freed HF cache for {model_id}", flush=True)


def _evidence_to_dict(e: Evidence) -> dict[str, Any]:
    return {
        "chunk_id": e.chunk_id,
        "document_id": e.document_id,
        "document_title": e.document_title,
        "page_start": e.page_start,
        "page_end": e.page_end,
        "text": e.text,
        "score": e.score,
        "rank": e.rank,
        "retriever_name": e.retriever_name,
    }


def _chunk_meta_dict(c: Chunk) -> dict[str, Any]:
    return {
        "document_id": c.document_id,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "text": c.text,
    }


def _meta_chunk_is_relevant(
    meta: dict[str, Any], question: EvalQuestion
) -> bool:
    """chunk_is_relevant over serialized chunk metadata."""
    for ref in question.evidence:
        if ref.document_id != meta["document_id"]:
            continue
        if any(meta["page_start"] <= p <= meta["page_end"] for p in ref.pages):
            return True
    return False


def _collect_contexts(
    dev_qs: list[EvalQuestion],
    test_qs: list[EvalQuestion],
    chunks: list[Chunk],
    best_spec: RetrievalSpec,
    best_rerank: str,
    comps: Components,
    best_size: int,
    k_ctxs: tuple[int, ...],
    tracer: OTelTracer,
) -> dict[str, Any]:
    """Stage A: serialize approved top-k evidence per question."""
    retriever = build_retriever(best_spec, chunks, comps)
    reranker = (
        CrossEncoderReranker(comps.pair_scorer, max_candidates=RERANK_DEPTH)
        if best_rerank == "cross_encoder" and comps.pair_scorer is not None
        else None
    )
    top_k = max(k_ctxs)
    cfg = FrozenConfig(
        chunk_tokens=best_size,
        overlap_ratio=OVERLAP_RATIO,
        retrieval=best_spec,
        rerank=best_rerank,
        abstain_threshold=None,
        evidence_top_k=top_k,
    )
    questions: list[dict[str, Any]] = []
    for q in [*dev_qs, *test_qs]:
        query_sha, query_len = fingerprint_query(q.question)
        with tracer.span(
            "request",
            {"query_sha256": query_sha, "query_length": query_len},
        ):
            ranked, _rerank_ms = _retrieve_ranked(
                q, retriever, reranker, cfg, tracer
            )
        evidence = ranked[:top_k]
        questions.append(
            {
                "question_id": q.question_id,
                "split": q.split,
                "category": q.category,
                "difficulty": q.difficulty,
                "answerable": q.answerable,
                "question": q.question,
                "required_points": [list(p) for p in q.required_points],
                "top_score": float(ranked[0].score) if ranked else 0.0,
                "retrieved_chunk_ids": [e.chunk_id for e in ranked],
                "evidence": [_evidence_to_dict(e) for e in evidence],
            }
        )
    return {
        "k_ctxs": list(k_ctxs),
        "temperature": comps.temperature,
        "max_context_tokens": comps.max_context_tokens,
        "chunk_meta": {c.chunk_id: _chunk_meta_dict(c) for c in chunks},
        "questions": questions,
    }


def _finalize_record_fields(
    record: dict[str, Any],
    q: EvalQuestion,
    chunk_meta: dict[str, dict[str, Any]],
) -> None:
    """Add answer-quality and citation fields stage B cannot compute."""
    if q.answerable:
        claim_texts = " ".join(
            c.get("text", "") for c in record.get("claims", [])
        )
        haystack = f"{record.get('answer', '')} {claim_texts}"
        recall, covered, total, all_cov = required_point_recall(
            haystack, [list(p) for p in q.required_points]
        )
        answered = record.get("status") == "answered"
        record["required_point_recall"] = recall if answered else 0.0
        record["points_covered"] = covered
        record["points_total"] = total
        record["all_points_covered"] = all_cov if answered else False
    cited_ids = list(dict.fromkeys(record.get("citations", [])))
    cited_meta = [
        chunk_meta[cid] for cid in cited_ids if cid in chunk_meta
    ]
    relevant = [m for m in cited_meta if _meta_chunk_is_relevant(m, q)]
    record["citation_page_precision"] = (
        len(relevant) / len(cited_meta) if cited_meta else None
    )
    record["citation_hit"] = 1.0 if relevant else 0.0


def _score_nli_records(
    records: list[dict[str, Any]],
    qmap: dict[str, EvalQuestion],
    chunk_meta: dict[str, dict[str, Any]],
    nli_scorer: NLIScorer,
    tracer: OTelTracer,
) -> None:
    """Windowed NLI claim support for answered records (stage C)."""
    for record in records:
        if record.get("status") != "answered" or not record.get("claims"):
            continue
        q = qmap.get(record["question_id"])
        if q is None:
            continue
        window_pairs: list[tuple[str, str]] = []
        whole_pairs: list[tuple[str, str]] = []
        pair_claim_idx: list[int] = []
        pair_is_window: list[bool] = []
        for ci, claim in enumerate(record["claims"]):
            for cid in claim.get("chunk_ids", []):
                meta = chunk_meta.get(cid)
                if meta is None:
                    continue
                for window in sentence_windows(meta["text"]):
                    window_pairs.append((window, claim["text"]))
                    pair_claim_idx.append(ci)
                    pair_is_window.append(True)
                whole_pairs.append((meta["text"], claim["text"]))
        all_pairs = window_pairs + whole_pairs
        if not all_pairs:
            continue
        pair_claim_idx_all = pair_claim_idx + [
            ci
            for ci, claim in enumerate(record["claims"])
            for cid in claim.get("chunk_ids", [])
            if cid in chunk_meta
        ]
        pair_is_window_all = pair_is_window + [False] * len(whole_pairs)
        with tracer.span("nli_scoring") as nli_span:
            labels = nli_scorer.predict_labels(all_pairs)
            nli_span.set_attribute("pair_count", len(all_pairs))
        supported = [False] * len(record["claims"])
        whole_supported = [False] * len(record["claims"])
        for (_premise, _hyp), label, ci, is_win in zip(
            all_pairs,
            labels,
            pair_claim_idx_all,
            pair_is_window_all,
            strict=True,
        ):
            if claim_supported(label):
                if is_win:
                    supported[ci] = True
                else:
                    whole_supported[ci] = True
        record["nli_supported_claims"] = sum(supported)
        record["nli_total_claims"] = len(record["claims"])
        record["nli_whole_chunk_supported_claims"] = sum(whole_supported)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _merge_candidate_traces(work_dir: Path, traces_path: Path) -> None:
    """Append each candidate's span file into the shared traces.jsonl."""
    chunks_out: list[str] = []
    if traces_path.is_file():
        chunks_out.append(traces_path.read_text(encoding="utf-8"))
    for trace_file in sorted(work_dir.glob("*/*/traces_*.jsonl")):
        chunks_out.append(trace_file.read_text(encoding="utf-8"))
    traces_path.write_text("".join(chunks_out), encoding="utf-8")


def _staged_generate_candidates(
    dev_qs: list[EvalQuestion],
    test_qs: list[EvalQuestion],
    chunks: list[Chunk],
    best_spec: RetrievalSpec,
    best_rerank: str,
    best_size: int,
    comps: Components,
    k_ctxs: tuple[int, ...],
    out_dir: Path,
    tracer: OTelTracer,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Stages A/B/C: contexts, per-candidate subprocesses, metrics."""
    _log_resources("stage-a-start")
    contexts = _collect_contexts(
        dev_qs, test_qs, chunks, best_spec, best_rerank, comps,
        best_size, k_ctxs, tracer,
    )
    ctx_path = out_dir / "contexts.json"
    ctx_path.write_text(json.dumps(contexts), encoding="utf-8")
    _log_resources("stage-a-end")

    # Unload retrieval models so generator subprocesses get the worker.
    comps.encoder = None
    comps.pair_scorer = None
    gc.collect()
    _empty_cuda_cache()

    work = out_dir / "candidates"
    spec_dir = work / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    qmap = {q.question_id: q for q in [*dev_qs, *test_qs]}
    chunk_meta = contexts["chunk_meta"]
    provider_meta: dict[str, dict[str, str]] = {}
    raw_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

    # Stage B: one subprocess per (provider spec, k_ctx) so each process
    # handles at most one split-pass of generations; a failure aborts.
    expected_counts = {
        split: sum(1 for q in contexts["questions"] if q["split"] == split)
        for split in ("dev", "test")
    }
    for pname, spec in comps.provider_specs.items():
        spec = dict(spec)
        spec["name"] = pname
        spec_path = spec_dir / f"{pname}.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        provider_meta[pname] = {
            "model_id": str(spec.get("model_id", "")),
            "revision": str(spec.get("revision", "") or ""),
            "quantization": str(spec.get("quantization") or ""),
        }
        for k_ctx in k_ctxs:
            cand_dir = work / pname / f"k{k_ctx}"
            _log_resources(f"stage-b-{pname}-k{k_ctx}-start")
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "rag_system.evaluation.generate_candidate",
                    "--contexts",
                    str(ctx_path),
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    str(cand_dir),
                    "--k-ctx",
                    str(k_ctx),
                ],
                check=True,
            )
            _log_resources(f"stage-b-{pname}-k{k_ctx}-end")
            for split, expected in expected_counts.items():
                record_path = (
                    cand_dir / f"{pname}__k{k_ctx}__{split}.jsonl"
                )
                lines = (
                    sum(
                        1
                        for line in record_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    )
                    if record_path.is_file()
                    else 0
                )
                if lines != expected:
                    raise RuntimeError(
                        f"candidate {pname} k_ctx={k_ctx} {split} wrote "
                        f"{lines} records, expected {expected}"
                    )
        model_id = spec.get("model_id")
        if model_id:
            _free_model_cache(str(model_id))
        gc.collect()
        _empty_cuda_cache()

    # Stage C: finalize fields, tune thresholds on dev, score NLI on test.
    _log_resources("stage-c-start")
    gen_candidates: list[dict[str, Any]] = []
    for pname in comps.provider_specs:
        for k_ctx in k_ctxs:
            dev_raw = _read_records(
                work / pname / f"k{k_ctx}" / f"{pname}__k{k_ctx}__dev.jsonl"
            )
            test_raw = _read_records(
                work / pname / f"k{k_ctx}" / f"{pname}__k{k_ctx}__test.jsonl"
            )
            for rec in [*dev_raw, *test_raw]:
                q = qmap.get(rec["question_id"])
                if q is not None:
                    _finalize_record_fields(rec, q, chunk_meta)
            # Dev: tune threshold on model outcomes, then apply.
            threshold_k, obj_k, _table_k = tune_abstention(dev_raw)
            dev_applied = _apply_threshold_to_records(dev_raw, threshold_k)
            dev_eval = aggregate_e2e_metrics(dev_applied)
            dev_unans = [r for r in dev_applied if not r["answerable"]]
            all_unans_abstained = bool(dev_unans) and all(
                r["status"] != "answered" for r in dev_unans
            )
            threshold_binding = sum(
                1
                for r in dev_raw
                if r.get("status") == "answered"
                and threshold_k is not None
                and float(r.get("top_score") or 0.0) < threshold_k
            )
            # Test: apply the frozen dev threshold, then NLI on survivors.
            test_applied = _apply_threshold_to_records(test_raw, threshold_k)
            if comps.nli_scorer is not None:
                _score_nli_records(
                    test_applied, qmap, chunk_meta,
                    comps.nli_scorer, tracer,
                )
            test_eval = aggregate_e2e_metrics(test_applied)
            gen_candidates.append(
                {
                    "provider": pname,
                    "k_ctx": k_ctx,
                    "threshold": threshold_k,
                    "threshold_objective": obj_k,
                    "threshold_dev_binding": threshold_binding,
                    "dev_eval": dev_eval,
                    "dev_all_unanswerables_abstained": all_unans_abstained,
                    "test_eval": test_eval,
                    "test_records": test_applied,
                    "dev_records": dev_applied,
                    "schema_bound_violations_dev": (
                        count_schema_bound_violations(dev_raw)
                    ),
                    "schema_bound_violations_test": (
                        count_schema_bound_violations(test_raw)
                    ),
                }
            )
            raw_by_key[(pname, k_ctx, "dev")] = dev_raw
            raw_by_key[(pname, k_ctx, "test")] = test_raw

    # Reload retrieval models for the stage-C/D ablation pass.
    if comps.encoder is None and comps.encoder_factory is not None:
        comps.encoder = comps.encoder_factory()
    if comps.pair_scorer is None and comps.pair_scorer_factory is not None:
        comps.pair_scorer = comps.pair_scorer_factory()
    _log_resources("stage-c-end")

    tracer.force_flush()
    _merge_candidate_traces(work, out_dir / "traces.jsonl")
    return gen_candidates, provider_meta


def run_full_benchmark(
    docs: list[Document],
    corpus_manifest: dict[str, Any],
    eval_questions: list[EvalQuestion],
    comps: Components,
    out_dir: Path,
    *,
    eval_path: Path | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Run the complete dev-tune/freeze/test pipeline and write artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_qs = [q for q in eval_questions if q.split == "dev"]
    test_qs = [q for q in eval_questions if q.split == "test"]

    tracer = OTelTracer(
        service_name="rag_system-benchmark",
        span_exporters=[JsonlSpanExporter(out_dir / "traces.jsonl")],
        enable_otlp=False,
    )

    # 1. Chunk size selection on dev.
    best_size, chunk_table, chunks_by_size = select_chunk_size(docs, dev_qs, comps)
    chunks = chunks_by_size[best_size]
    chunk_map = {c.chunk_id: c for c in chunks}

    # 2. Retrieval tuning on dev.
    best_spec, retrieval_table = tune_retrieval(chunks, dev_qs, comps)

    # 3. Rerank selection on dev.
    best_rerank, rerank_table = tune_rerank(chunks, dev_qs, comps, best_spec)
    retriever = build_retriever(best_spec, chunks, comps)

    # 4. Test retrieval-only ablations for every retrieval system. These run
    #    before generation because the staged path unloads the encoder and
    #    pair scorer for the provider subprocesses.
    retrieval_ablations: dict[str, Any] = {}
    for spec in retrieval_grid():
        if spec.kind != "bm25" and comps.encoder is None:
            continue
        ret = build_retriever(spec, chunks, comps)
        scores, _ = eval_retrieval(ret, test_qs, chunks)
        retrieval_ablations[spec.name] = mean_retrieval_scores(scores)
    # Best retrieval system with each rerank option.
    for option in ("none", "cross_encoder"):
        rr = (
            CrossEncoderReranker(comps.pair_scorer, max_candidates=RERANK_DEPTH)
            if option == "cross_encoder" and comps.pair_scorer is not None
            else None
        )
        rows: list[RetrievalScores] = []
        for q in test_qs:
            if not q.answerable:
                continue
            cands = retriever.retrieve(q.question, top_k=CANDIDATE_DEPTH)
            ranked, _ = rerank_candidates(cands, q.question, rr, RETRIEVAL_TOP_K)
            relevant = {c.chunk_id for c in chunks if chunk_is_relevant(c, q)}
            rows.append(retrieval_scores([e.chunk_id for e in ranked], relevant))
        retrieval_ablations[f"{best_spec.name}+rerank-{option}"] = mean_retrieval_scores(rows)

    # 5. Generator x context-size candidates. Serializable provider specs
    #    select the staged pipeline (one subprocess per provider); without
    #    them the single provider runs in-process as before.
    k_ctxs = comps.k_ctx_candidates or (EVIDENCE_TOP_K,)
    if comps.provider_specs:
        gen_candidates, provider_meta = _staged_generate_candidates(
            dev_qs, test_qs, chunks, best_spec, best_rerank, best_size,
            comps, k_ctxs, out_dir, tracer,
        )
    else:
        reranker = (
            CrossEncoderReranker(comps.pair_scorer, max_candidates=RERANK_DEPTH)
            if best_rerank == "cross_encoder" and comps.pair_scorer is not None
            else None
        )
        provider_specs: dict[str, Callable[[], Any]] = {
            comps.provider_name: (lambda: comps.provider)
        }
        gen_candidates = []
        provider_meta = {}
        for pname, factory in provider_specs.items():
            provider = factory()
            provider_meta[pname] = {
                "model_id": str(
                    getattr(provider, "model_id", comps.provider_model_id)
                ),
                "revision": str(getattr(provider, "revision", "") or ""),
            }
            for k_ctx in k_ctxs:
                cfg_dev = FrozenConfig(
                    chunk_tokens=best_size,
                    overlap_ratio=OVERLAP_RATIO,
                    retrieval=best_spec,
                    rerank=best_rerank,
                    abstain_threshold=None,
                    evidence_top_k=k_ctx,
                )
                dev_recs = [
                    run_pipeline_question(
                        q, retriever, reranker, provider, comps, cfg_dev,
                        chunk_map, tracer,
                        abstain_threshold=None,
                        record_nli=False,
                    )
                    for q in dev_qs
                ]
                # Abstention threshold on dev only, per candidate.
                threshold_k, obj_k, _table_k = tune_abstention(dev_recs)
                dev_applied = _apply_threshold_to_records(dev_recs, threshold_k)
                dev_eval = aggregate_e2e_metrics(dev_applied)
                dev_unans = [r for r in dev_applied if not r["answerable"]]
                all_unans_abstained = bool(dev_unans) and all(
                    r["status"] != "answered" for r in dev_unans
                )
                threshold_binding = sum(
                    1
                    for r in dev_recs
                    if r.get("status") == "answered"
                    and threshold_k is not None
                    and float(r.get("top_score") or 0.0) < threshold_k
                )
                cfg_test = FrozenConfig(
                    chunk_tokens=best_size,
                    overlap_ratio=OVERLAP_RATIO,
                    retrieval=best_spec,
                    rerank=best_rerank,
                    abstain_threshold=threshold_k,
                    evidence_top_k=k_ctx,
                )
                test_recs = [
                    run_pipeline_question(
                        q, retriever, reranker, provider, comps, cfg_test,
                        chunk_map, tracer,
                        abstain_threshold=threshold_k,
                        record_nli=True,
                    )
                    for q in test_qs
                ]
                gen_candidates.append(
                    {
                        "provider": pname,
                        "k_ctx": k_ctx,
                        "threshold": threshold_k,
                        "threshold_objective": obj_k,
                        "threshold_dev_binding": threshold_binding,
                        "dev_eval": dev_eval,
                        "dev_all_unanswerables_abstained": all_unans_abstained,
                        "test_eval": aggregate_e2e_metrics(test_recs),
                        "test_records": test_recs,
                        "dev_records": dev_recs,
                        "schema_bound_violations_dev": (
                            count_schema_bound_violations(dev_recs)
                        ),
                        "schema_bound_violations_test": (
                            count_schema_bound_violations(test_recs)
                        ),
                    }
                )
            del provider
            gc.collect()
            _empty_cuda_cache()

    # Select on dev: required-point recall subject to abstaining on all dev
    # unanswerables; if none qualifies, fall back to best abstention F1.
    feasible = [
        g for g in gen_candidates if g["dev_all_unanswerables_abstained"]
    ]
    if feasible:
        best_gen = max(
            feasible,
            key=lambda g: (
                g["dev_eval"]["required_point_recall"]["mean"],
                g["dev_eval"]["abstain_f1"],
            ),
        )
        gen_objective = (
            "required_point_recall_s.t._all_dev_unanswerables_abstain"
        )
    else:
        best_gen = max(
            gen_candidates, key=lambda g: g["dev_eval"]["abstain_f1"]
        )
        gen_objective = "abstain_f1_fallback_no_feasible_candidate"

    frozen = FrozenConfig(
        chunk_tokens=best_size,
        overlap_ratio=OVERLAP_RATIO,
        retrieval=best_spec,
        rerank=best_rerank,
        abstain_threshold=best_gen["threshold"],
        evidence_top_k=best_gen["k_ctx"],
    )
    dev_records = best_gen["dev_records"]
    test_records = best_gen["test_records"]
    e2e_metrics = dict(best_gen["test_eval"])
    e2e_metrics["schema_bound_violations"] = best_gen[
        "schema_bound_violations_test"
    ]

    # 6. Metrics, latency table, GPU seconds from exported spans.
    traces_path = out_dir / "traces.jsonl"
    latency_table = stage_latency_table(traces_path)
    gpu_sec = gpu_seconds_from_spans(traces_path)

    metrics = {
        "e2e_test": e2e_metrics,
        "generator_ablations_test": {
            f"{g['provider']}__k{g['k_ctx']}": {
                **g["test_eval"],
                "schema_bound_violations": g[
                    "schema_bound_violations_test"
                ],
            }
            for g in gen_candidates
        },
        "schema_bound_violations": {
            "dev": sum(g["schema_bound_violations_dev"] for g in gen_candidates),
            "test": sum(
                g["schema_bound_violations_test"] for g in gen_candidates
            ),
        },
        "retrieval_ablations_test": retrieval_ablations,
        "latency_by_stage": latency_table,
        "gpu_seconds": gpu_sec,
        "nli_note": (
            "model-based estimate; entailment argmax over ~3-sentence "
            "windows of cited chunks (whole-chunk kept as secondary)"
        ),
    }
    tracer.force_flush()
    tracer.shutdown()

    manifest = {
        "git_sha": git_sha or _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eval_file": str(eval_path) if eval_path else "data/eval/nist_qa_v1.jsonl",
        "eval_sha256": (
            hashlib.sha256(eval_path.read_bytes()).hexdigest()
            if eval_path and eval_path.is_file()
            else None
        ),
        "corpus": corpus_manifest,
        "frozen_config": {
            "chunk_tokens": frozen.chunk_tokens,
            "overlap_ratio": frozen.overlap_ratio,
            "retrieval": {
                "name": frozen.retrieval.name,
                "kind": frozen.retrieval.kind,
                "rrf_k": frozen.retrieval.rrf_k,
                "bm25_weight": frozen.retrieval.bm25_weight,
                "dense_weight": frozen.retrieval.dense_weight,
            },
            "rerank": frozen.rerank,
            "rerank_depth": frozen.rerank_depth,
            "provider": best_gen["provider"],
            "abstain_threshold": frozen.abstain_threshold,
            "threshold_score_scale": _score_scale_desc(frozen),
            "threshold_effectively_inactive": (
                best_gen["threshold_dev_binding"] == 0
            ),
            "candidate_depth": frozen.candidate_depth,
            "retrieval_top_k": frozen.retrieval_top_k,
            "evidence_top_k": frozen.evidence_top_k,
            "seed": frozen.seed,
        },
        "dev_selection": {
            "chunk_size_table": chunk_table,
            "retrieval_table": retrieval_table,
            "rerank_table": rerank_table,
            "generator_objective": gen_objective,
            "generator_table": [
                {
                    "provider": g["provider"],
                    "k_ctx": g["k_ctx"],
                    "threshold": g["threshold"],
                    "threshold_objective": g["threshold_objective"],
                    "dev_all_unanswerables_abstained": g[
                        "dev_all_unanswerables_abstained"
                    ],
                    "dev_required_point_recall": g["dev_eval"][
                        "required_point_recall"
                    ]["mean"],
                    "dev_abstain_f1": g["dev_eval"]["abstain_f1"],
                    "dev_false_abstain_rate": g["dev_eval"][
                        "false_abstain_rate"
                    ],
                    "dev_abstention_sources": g["dev_eval"][
                        "abstention_sources"
                    ],
                    "test_required_point_recall": g["test_eval"][
                        "required_point_recall"
                    ]["mean"],
                    "test_abstain_f1": g["test_eval"]["abstain_f1"],
                    "test_false_abstain_rate": g["test_eval"][
                        "false_abstain_rate"
                    ],
                    "test_parse_failures": g["test_eval"]["parse_failures"],
                    "dev_truncation_rate": g["dev_eval"]["truncation_rate"],
                    "test_truncation_rate": g["test_eval"]["truncation_rate"],
                    "schema_bound_violations_dev": g[
                        "schema_bound_violations_dev"
                    ],
                    "schema_bound_violations_test": g[
                        "schema_bound_violations_test"
                    ],
                }
                for g in gen_candidates
            ],
        },
        "models": {
            "dense": {"model_id": comps.dense_model_id, "revision": comps.dense_revision},
            "reranker": {"model_id": comps.reranker_model_id, "revision": comps.reranker_revision},
            "generator": provider_meta.get(best_gen["provider"], {}),
            "generator_candidates": provider_meta,
            "nli": {"model_id": comps.nli_model_id, "revision": comps.nli_revision},
            "tokenizer": comps.tokenizer.name,
        },
        "gpu": _gpu_name(),
        "package_versions": _package_versions(),
        "seeds": {"benchmark": SEED, "bootstrap": 42, "n_resamples": 1000},
    }

    write_jsonl(test_records, out_dir / "test_questions.jsonl")
    write_jsonl(dev_records, out_dir / "dev_questions.jsonl")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        render_report({"metrics": metrics, "manifest": manifest}),
        encoding="utf-8",
    )
    return {"metrics": metrics, "manifest": manifest, "records": test_records}
