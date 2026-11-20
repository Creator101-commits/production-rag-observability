"""Evaluation pipeline runner across benchmark dataset."""

from __future__ import annotations

import time

import numpy as np

from rag_system.citations.validator import CitationValidator
from rag_system.config import RAGConfig
from rag_system.evaluation.dataset import EvalQuestion
from rag_system.evaluation.metrics import (
    OverallRAGMetrics,
    compute_abstention_metrics,
    compute_latency_percentiles,
)
from rag_system.generation.base import GenerationRequest
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.generation.resolve import resolve_generation_response
from rag_system.ingestion.chunk import chunk_document
from rag_system.ingestion.corpus import load_offline_fixture_corpus
from rag_system.observability.tracing import OTelTracer, fingerprint_query
from rag_system.reranking.lexical import LexicalOverlapReranker
from rag_system.retrieval.hybrid import HybridRetriever


class BenchmarkRunner:
    """End-to-end evaluation runner over versioned question suites."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.tracer = OTelTracer(service_name="rag_system-benchmark")
        self.retriever = HybridRetriever(rrf_k=self.config.rrf_k)
        self.reranker = LexicalOverlapReranker()
        self.generator = DeterministicFakeProvider(
            default_abstain_threshold=self.config.evidence_score_threshold
        )

        # Index corpus
        docs = load_offline_fixture_corpus()
        chunks = []
        for doc in docs:
            chunks.extend(
                chunk_document(
                    doc,
                    chunk_size_tokens=self.config.chunk_size_tokens,
                    chunk_overlap_tokens=self.config.chunk_overlap_tokens,
                )
            )
        self.retriever.index(chunks)

    def _cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost only when both per-1k prices are explicitly configured."""
        pin = self.config.input_token_cost_per_1k
        pout = self.config.output_token_cost_per_1k
        if pin is None or pout is None:
            return 0.0
        return (prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout

    def run_evaluation(self, questions: list[EvalQuestion]) -> OverallRAGMetrics:
        """Run full evaluation suite and return consolidated metrics."""
        y_true_answerable: list[bool] = []
        y_pred_answered: list[bool] = []
        recalls_at_5: list[float] = []
        mrrs: list[float] = []
        valid_citations_count = 0
        total_citations_count = 0
        latencies_ms: list[float] = []
        total_cost = 0.0

        for q in questions:
            query_sha, query_len = fingerprint_query(q.question)
            q_started = time.perf_counter()

            with self.tracer.span(
                "request",
                {"query_sha256": query_sha, "query_length": query_len},
            ):
                with self.tracer.span("retrieval"):
                    cands = self.retriever.retrieve(
                        q.question,
                        top_k=self.config.retrieval_candidate_depth,
                    )

                with self.tracer.span("reranking"):
                    top_evidence, _ = self.reranker.rerank(
                        q.question,
                        cands,
                        top_k=self.config.rerank_top_k,
                    )

                # Retrieval recall & MRR on expected documents
                if q.answerable and q.expected_doc_ids:
                    retrieved_docs_top5 = [ev.document_id for ev in top_evidence[:5]]
                    hits = sum(
                        1 for exp in q.expected_doc_ids if exp in retrieved_docs_top5
                    )
                    recalls_at_5.append(hits / len(q.expected_doc_ids))

                    mrr = 0.0
                    for rank, ev in enumerate(top_evidence[:5], 1):
                        if ev.document_id in q.expected_doc_ids:
                            mrr = 1.0 / rank
                            break
                    mrrs.append(mrr)

                # Generation
                gen_req = GenerationRequest(
                    query=q.question,
                    evidence_list=top_evidence,
                    temperature=self.config.temperature,
                    evidence_score_threshold=self.config.evidence_score_threshold,
                    max_context_tokens=self.config.max_context_tokens,
                )
                with self.tracer.span("generation") as gen_span:
                    gen_resp, _resolve_warnings = resolve_generation_response(
                        self.generator.generate(gen_req), top_evidence
                    )
                    gen_resp = CitationValidator.enforce(
                        gen_resp,
                        top_evidence,
                        self.config.min_required_citations,
                    )
                    gen_span.set_attribute("prompt_tokens", gen_resp.prompt_tokens)
                    gen_span.set_attribute(
                        "completion_tokens", gen_resp.completion_tokens
                    )
                    gen_span.set_attribute("status", gen_resp.status)

                with self.tracer.span("citation_validation"):
                    val_result = CitationValidator.validate(
                        gen_resp.answer, top_evidence
                    )

            latencies_ms.append((time.perf_counter() - q_started) * 1000.0)
            total_cost += self._cost_usd(
                gen_resp.prompt_tokens, gen_resp.completion_tokens
            )

            # Abstention tracking
            is_answered = gen_resp.status == "answered"
            y_true_answerable.append(q.answerable)
            y_pred_answered.append(is_answered)

            if is_answered:
                total_citations_count += len(gen_resp.citations)
                valid_citations_count += len(val_result.resolved_citations)

        mean_lat, p95_lat = compute_latency_percentiles(latencies_ms)
        abs_prec, abs_rec, false_ans, false_abs = compute_abstention_metrics(
            y_true_answerable, y_pred_answered
        )

        cit_validity_rate = (
            (valid_citations_count / total_citations_count)
            if total_citations_count > 0
            else 1.0
        )

        return OverallRAGMetrics(
            retrieval_recall_at_5=float(np.mean(recalls_at_5)) if recalls_at_5 else 0.0,
            retrieval_mrr=float(np.mean(mrrs)) if mrrs else 0.0,
            citation_validity_rate=float(cit_validity_rate),
            citation_precision=float(cit_validity_rate),
            abstention_precision=float(abs_prec),
            abstention_recall=float(abs_rec),
            false_answer_rate=float(false_ans),
            false_abstention_rate=float(false_abs),
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            total_cost_usd=total_cost,
        )
