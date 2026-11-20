"""FastAPI service endpoints for query answering, health, readiness, and feedback."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from rag_system import __version__
from rag_system.api.pipeline import RAGPipeline
from rag_system.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from rag_system.config import RAGConfig
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.ingestion.chunk import chunk_document
from rag_system.ingestion.corpus import load_offline_fixture_corpus
from rag_system.observability.tracing import OTelTracer
from rag_system.reranking.lexical import LexicalOverlapReranker
from rag_system.retrieval.hybrid import HybridRetriever


def create_app(config: RAGConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI RAG service application."""
    cfg = config or RAGConfig()
    app = FastAPI(
        title="Production RAG Service",
        description="Citation-grounded NIST AI RMF Q&A service with end-to-end observability.",
        version=__version__,
    )

    retriever = HybridRetriever(rrf_k=cfg.rrf_k)
    reranker = LexicalOverlapReranker()
    generator = DeterministicFakeProvider(default_abstain_threshold=cfg.evidence_score_threshold)

    docs = load_offline_fixture_corpus()
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

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        provider=generator,
        tracer=OTelTracer(service_name="rag_system"),
        config=cfg,
    )

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
        """Retrieve evidence, rerank, generate citation-grounded response."""
        if not req.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text cannot be empty.",
            )
        return pipeline.query(req.query, req.request_id)

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
        """Capture user satisfaction rating and feedback."""
        feedback_log.append(req)
        return FeedbackResponse(status="recorded", request_id=req.request_id)

    return app
