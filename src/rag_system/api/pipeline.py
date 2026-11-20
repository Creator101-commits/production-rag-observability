"""Shared query pipeline: retrieval, fusion, rerank, threshold, generation.

Emits OpenTelemetry spans for every stage and applies the code-level
evidence threshold on the top post-retrieval score before generation.
Raw query text never leaves the process; spans carry only fingerprints
and aggregate attributes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

from rag_system.api.schemas import CitationItem, QueryResponse
from rag_system.citations.validator import CitationValidator
from rag_system.config import RAGConfig
from rag_system.generation.base import GenerationRequest, GenerationResponse
from rag_system.generation.provider import ABSTAIN_ANSWER
from rag_system.generation.resolve import resolve_generation_response
from rag_system.observability.tracing import OTelTracer, fingerprint_query
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.retrieval.hybrid import HybridRetriever
from rag_system.retrieval.schema import Evidence


@dataclass
class RAGPipeline:
    """Configured retrieve-rerank-generate pipeline with tracing."""

    retriever: Any
    reranker: CrossEncoderReranker | Any
    provider: Any
    tracer: OTelTracer
    config: RAGConfig

    def _retrieval_span(self, tracer: OTelTracer, query: str, depth: int) -> list[Evidence]:
        """Retrieve candidates; hybrid retrievers get a fusion sub-span."""
        if isinstance(self.retriever, HybridRetriever):
            with tracer.span("retrieval") as ret_span:
                bm25_cands, dense_cands = self.retriever.candidate_lists(
                    query, candidate_depth=depth
                )
                ret_span.set_attribute(
                    "candidate_count", len(bm25_cands) + len(dense_cands)
                )
            with tracer.span("fusion") as fus_span:
                fused = self.retriever.fuse(bm25_cands, dense_cands, depth)
                fus_span.set_attribute("fused_count", len(fused))
            return fused
        with tracer.span("retrieval") as ret_span:
            cands = cast(
                list[Evidence],
                self.retriever.retrieve(query, top_k=depth),
            )
            ret_span.set_attribute("candidate_count", len(cands))
        return cands

    def query(self, query_text: str, request_id: str | None = None) -> QueryResponse:
        """Answer one query end to end with full stage instrumentation."""
        cfg = self.config
        rid = request_id or str(uuid.uuid4())
        q_sha, q_len = fingerprint_query(query_text)
        started = time.perf_counter()

        with self.tracer.span(
            "request",
            {"request_id": rid, "query_sha256": q_sha, "query_length": q_len},
        ) as request_span:
            trace_id = format(request_span.get_span_context().trace_id, "032x")

            candidates = self._retrieval_span(
                self.tracer, query_text, cfg.retrieval_candidate_depth
            )

            with self.tracer.span("rerank") as rr_span:
                if self.reranker is not None:
                    top_evidence, rerank_ms = self.reranker.rerank(
                        query_text, candidates, top_k=cfg.rerank_top_k
                    )
                else:
                    top_evidence = candidates[: cfg.rerank_top_k]
                    rerank_ms = 0.0
                rr_span.set_attribute("rerank_ms", round(rerank_ms, 3))
                rr_span.set_attribute("enabled", self.reranker is not None)
                rr_span.set_attribute("evidence_count", len(top_evidence))

            top_score = top_evidence[0].score if top_evidence else 0.0
            threshold_hit = (
                top_evidence
                and cfg.evidence_score_threshold is not None
                and top_score < cfg.evidence_score_threshold
            )

            with self.tracer.span(
                "threshold",
                {
                    "top_score": round(float(top_score), 6),
                    "threshold": cfg.evidence_score_threshold,
                    "abstained": bool(threshold_hit),
                },
            ):
                pass

            resolve_warnings: tuple[str, ...] = ()
            if threshold_hit:
                gen_resp = GenerationResponse(
                    answer=ABSTAIN_ANSWER,
                    status="abstained",
                    abstain=True,
                    abstain_reason="insufficient_evidence_score",
                )
            else:
                gen_req = GenerationRequest(
                    query=query_text,
                    evidence_list=top_evidence,
                    temperature=cfg.temperature,
                    evidence_score_threshold=cfg.evidence_score_threshold,
                    max_context_tokens=cfg.max_context_tokens,
                )
                with self.tracer.span("generation") as gen_span:
                    raw = self.provider.generate(gen_req)
                    gen_resp, resolve_warnings = resolve_generation_response(
                        raw, top_evidence
                    )
                    gen_resp = CitationValidator.enforce(
                        gen_resp, top_evidence, cfg.min_required_citations
                    )
                    gen_span.set_attribute("status", gen_resp.status)
                    gen_span.set_attribute("prompt_tokens", gen_resp.prompt_tokens)
                    gen_span.set_attribute(
                        "completion_tokens", gen_resp.completion_tokens
                    )

            with self.tracer.span("citation_validation") as val_span:
                val_result = CitationValidator.validate(gen_resp.answer, top_evidence)
                val_span.set_attribute(
                    "citation_count", len(val_result.resolved_citations)
                )
                val_span.set_attribute("valid", val_result.is_valid)

            with self.tracer.span("serialization"):
                citation_items = [
                    CitationItem(
                        citation_id=c.citation_id,
                        document_id=c.document_id,
                        document_title=c.document_title,
                        page_start=c.page_start,
                        page_end=c.page_end,
                        text_snippet=c.text_snippet,
                    )
                    for c in val_result.resolved_citations
                ]
                response = QueryResponse(
                    request_id=rid,
                    trace_id=trace_id,
                    status=gen_resp.status,
                    answer=gen_resp.answer,
                    citations=citation_items,
                    claims=[
                        claim.text + "".join(f" [{cid}]" for cid in claim.chunk_ids)
                        for claim in gen_resp.claims
                    ],
                    warnings=[*resolve_warnings, *val_result.warnings],
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

        return response
