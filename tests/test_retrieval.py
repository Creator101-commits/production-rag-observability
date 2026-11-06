"""Tests for BM25, Dense, Hybrid RRF, and Cross-Encoder retrievers."""

from __future__ import annotations

import pytest

from rag_system.ingestion.schema import Chunk
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.dense import DenseRetriever
from rag_system.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            document_id="doc1",
            document_title="Title 1",
            page_start=1,
            page_end=1,
            heading="Section 1",
            text="The GOVERN core function cultivates risk management culture and processes.",
            token_count=10,
        ),
        Chunk(
            chunk_id="c2",
            document_id="doc1",
            document_title="Title 1",
            page_start=2,
            page_end=2,
            heading="Section 2",
            text="The MAP core function identifies and categorizes context and societal impacts.",
            token_count=10,
        ),
        Chunk(
            chunk_id="c3",
            document_id="doc2",
            document_title="Title 2",
            page_start=5,
            page_end=5,
            heading="Section 3",
            text="Testing and verification (TEVV) workflows require independent audit teams.",
            token_count=10,
        ),
    ]


def test_bm25_retrieval(sample_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever().index(sample_chunks)
    results = bm25.retrieve("GOVERN culture", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"

    unindexed = BM25Retriever()
    with pytest.raises(RuntimeError, match="must be indexed"):
        unindexed.retrieve("test")


def test_dense_retrieval(sample_chunks: list[Chunk]) -> None:
    dense = DenseRetriever().index(sample_chunks)
    results = dense.retrieve("TEVV testing and verification", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c3"

    unindexed = DenseRetriever()
    with pytest.raises(RuntimeError, match="must be indexed"):
        unindexed.retrieve("test")


def test_hybrid_rrf_retrieval(sample_chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever(rrf_k=10.0).index(sample_chunks)
    results = hybrid.retrieve("MAP context societal", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"

    unindexed = HybridRetriever()
    with pytest.raises(RuntimeError, match="must be indexed"):
        unindexed.retrieve("test")


def test_cross_encoder_reranking(sample_chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever().index(sample_chunks)
    candidates = hybrid.retrieve("GOVERN culture and processes", top_k=3)

    reranker = CrossEncoderReranker()
    reranked, latency_ms = reranker.rerank("GOVERN culture and processes", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert latency_ms >= 0.0
