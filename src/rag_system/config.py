"""System configuration settings for production RAG service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RAGConfig:
    """Configuration settings for retrieval, reranking, generation, and observability."""

    # Chunking
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32

    # Retrieval
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: float = 60.0
    retrieval_candidate_depth: int = 20

    # Reranking
    rerank_top_k: int = 5

    # Abstention policy
    evidence_score_threshold: float = 0.30
    min_required_citations: int = 1

    # Context window
    max_context_tokens: int = 2048

    # Generation provider
    provider_name: str = "fake"
    model_name: str = "deterministic-rag-generator-v1"
    temperature: float = 0.0

    # Observability
    enable_tracing: bool = True
    redact_pii: bool = True
    export_traces_locally: bool = True

    # Pricing per 1k tokens (USD)
    input_token_cost_per_1k: float = 0.0015
    output_token_cost_per_1k: float = 0.0020

    extra_metadata: dict[str, str] = field(default_factory=dict)
