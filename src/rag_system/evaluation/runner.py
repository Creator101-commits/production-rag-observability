"""Evaluation pipeline runner across benchmark dataset."""

from __future__ import annotations

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
from rag_system.ingestion.chunk import chunk_document
from rag_system.ingestion.corpus import load_nist_ai_rmf_corpus
from rag_system.observability.tracing import Tracer
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.retrieval.hybrid import HybridRetriever


class BenchmarkRunner:
    """End-to-end evaluation runner over versioned question suites."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.tracer = Tracer(
            input_token_cost_per_1k=self.config.input_token_cost_per_1k,
            output_token_cost_per_1k=self.config.output_token_cost_per_1k,
        )
        self.retriever = HybridRetriever(rrf_k=self.config.rrf_k)
        self.reranker = CrossEncoderReranker()
        self.generator = DeterministicFakeProvider(
            default_abstain_threshold=self.config.evidence_score_threshold
        )

        # Index corpus
        docs = load_nist_ai_rmf_corpus()
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
            trace = self.tracer.create_trace()

            with self.tracer.span(trace, "retrieval"):
                cands = self.retriever.retrieve(
                    q.question,
                    top_k=self.config.retrieval_candidate_depth,
                )

            with self.tracer.span(trace, "reranking"):
                top_evidence, _ = self.reranker.rerank(
                    q.question,
                    cands,
                    top_k=self.config.rerank_top_k,
                )

            # Retrieval recall & MRR on expected documents
            if q.answerable and q.expected_doc_ids:
                retrieved_docs_top5 = [ev.document_id for ev in top_evidence[:5]]
                hits = sum(1 for exp in q.expected_doc_ids if exp in retrieved_docs_top5)
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
            )
            with self.tracer.span(trace, "generation"):
                gen_resp = CitationValidator.enforce(
                    self.generator.generate(gen_req),
                    top_evidence,
                    self.config.min_required_citations,
                )

            self.tracer.record_token_usage(
                trace,
                gen_resp.prompt_tokens,
                gen_resp.completion_tokens,
            )

            # Citation validation
            with self.tracer.span(trace, "citation_validation"):
                val_result = CitationValidator.validate(gen_resp.answer, top_evidence)

            self.tracer.finish_trace(trace)
            latencies_ms.append(trace.total_latency_ms)
            total_cost += trace.estimated_cost_usd

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
            (valid_citations_count / total_citations_count) if total_citations_count > 0 else 1.0
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
