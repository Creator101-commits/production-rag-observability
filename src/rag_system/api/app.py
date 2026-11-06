"""FastAPI service endpoints for query answering, health, readiness, and feedback."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from rag_system import __version__
from rag_system.api.schemas import (
    CitationItem,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from rag_system.citations.validator import CitationValidator
from rag_system.config import RAGConfig
from rag_system.generation.base import GenerationRequest
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.ingestion.chunk import chunk_document
from rag_system.ingestion.corpus import load_nist_ai_rmf_corpus
from rag_system.observability.tracing import Tracer
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.retrieval.hybrid import HybridRetriever


def create_app(config: RAGConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI RAG service application."""
    cfg = config or RAGConfig()
    app = FastAPI(
        title="Production RAG Service",
        description="Citation-grounded NIST AI RMF Q&A service with end-to-end observability.",
        version=__version__,
    )

    # Initialize components
    tracer = Tracer(
        input_token_cost_per_1k=cfg.input_token_cost_per_1k,
        output_token_cost_per_1k=cfg.output_token_cost_per_1k,
    )
    retriever = HybridRetriever(rrf_k=cfg.rrf_k)
    reranker = CrossEncoderReranker()
    generator = DeterministicFakeProvider(default_abstain_threshold=cfg.evidence_score_threshold)

    # Ingest and index corpus
    docs = load_nist_ai_rmf_corpus()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                doc,
                chunk_size_tokens=cfg.chunk_size_tokens,
                chunk_overlap_tokens=cfg.chunk_overlap_tokens,
            )
        )
    retriever.index(all_chunks)

    # In-memory feedback store
    feedback_log: list[FeedbackRequest] = []

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness probe."""
        return HealthResponse(status="healthy", version=__version__)

    @app.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        """Readiness probe checking index readiness."""
        if not retriever.is_indexed:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Index is not ready",
            )
        return ReadinessResponse(status="ready", chunks_indexed=len(all_chunks))

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(req: QueryRequest) -> QueryResponse:
        """Retrieve evidence, rerank, generate citation-grounded response, and validate."""
        trace = tracer.create_trace(request_id=req.request_id)

        with tracer.span(trace, "request_validation", {"query": req.query}):
            if not req.query.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query text cannot be empty.",
                )

        with tracer.span(trace, "retrieval"):
            candidates = retriever.retrieve(req.query, top_k=cfg.retrieval_candidate_depth)

        with tracer.span(trace, "reranking"):
            top_evidence, _ = reranker.rerank(req.query, candidates, top_k=cfg.rerank_top_k)

        with tracer.span(trace, "generation"):
            gen_req = GenerationRequest(
                query=req.query,
                evidence_list=top_evidence,
                temperature=cfg.temperature,
                evidence_score_threshold=cfg.evidence_score_threshold,
            )
            gen_resp = generator.generate(gen_req)

        tracer.record_token_usage(
            trace,
            gen_resp.prompt_tokens,
            gen_resp.completion_tokens,
        )

        with tracer.span(trace, "citation_validation"):
            val_result = CitationValidator.validate(gen_resp.answer, top_evidence)

        tracer.finish_trace(trace)

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

        return QueryResponse(
            request_id=trace.request_id,
            trace_id=trace.trace_id,
            status=gen_resp.status,
            answer=gen_resp.answer,
            citations=citation_items,
            claims=gen_resp.claims,
            warnings=val_result.warnings,
            latency_ms=trace.total_latency_ms,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
        """Capture user satisfaction rating and feedback."""
        feedback_log.append(req)
        return FeedbackResponse(status="recorded", request_id=req.request_id)

    return app
