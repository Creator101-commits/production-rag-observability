"""Tests for BM25, hashed embedding, sentence-transformer, hybrid, reranking."""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from rag_system.ingestion.schema import Chunk
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.reranking.lexical import LexicalOverlapReranker
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.hashed import HashedEmbeddingRetriever
from rag_system.retrieval.hybrid import HybridRetriever
from rag_system.retrieval.sentence_transformer import SentenceTransformerRetriever


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


class FakeEncoder:
    """Deterministic bag-of-words encoder; no model downloads."""

    model_id = "fake-encoder-v1"
    revision = "test"
    dim = 32

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                idx = zlib.crc32(word.encode()) % self.dim
                mat[i, idx] += 1.0
        return mat


def test_bm25_retrieval(sample_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever().index(sample_chunks)
    results = bm25.retrieve("GOVERN culture", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"

    unindexed = BM25Retriever()
    with pytest.raises(RuntimeError, match="must be indexed"):
        unindexed.retrieve("test")


def test_hashed_embedding_retrieval(sample_chunks: list[Chunk]) -> None:
    retriever = HashedEmbeddingRetriever().index(sample_chunks)
    results = retriever.retrieve("TEVV testing and verification", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c3"
    assert results[0].retriever_name == "hashed_embedding"

    unindexed = HashedEmbeddingRetriever()
    with pytest.raises(RuntimeError, match="must be indexed"):
        unindexed.retrieve("test")


def test_sentence_transformer_retriever(sample_chunks: list[Chunk]) -> None:
    retriever = SentenceTransformerRetriever(
        FakeEncoder(), query_prefix="query: ", retriever_name="dense-test"
    ).index(sample_chunks)
    results = retriever.retrieve("MAP context societal", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "c2"
    assert results[0].retriever_name == "dense-test"
    # Normalized inner product scores live in [-1, 1].
    assert all(-1.0 <= r.score <= 1.0 for r in results)

    # Query prefix is actually applied to the query text.
    seen: list[str] = []

    class SpyEncoder(FakeEncoder):
        def encode(self, texts: list[str]) -> np.ndarray:
            seen.extend(texts)
            return super().encode(texts)

    spy = SentenceTransformerRetriever(SpyEncoder(), query_prefix="Q> ").index(
        sample_chunks
    )
    spy.retrieve("hello", top_k=1)
    assert seen[-1].startswith("Q> ")

    unindexed = SentenceTransformerRetriever(FakeEncoder())
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


def test_hybrid_weighted_rrf_changes_order() -> None:
    chunks = [
        Chunk(
            chunk_id="only_bm25",
            document_id="d",
            document_title="t",
            page_start=1,
            page_end=1,
            heading="",
            text="alpha beta gamma delta",
            token_count=4,
        ),
        Chunk(
            chunk_id="only_dense",
            document_id="d",
            document_title="t",
            page_start=1,
            page_end=1,
            heading="",
            text="zz zz zz zz",
            token_count=4,
        ),
    ]
    # With unit weights both chunks appear: only_bm25 via both lists,
    # only_dense via the dense list alone.
    hybrid = HybridRetriever(rrf_k=1.0).index(chunks)
    results = hybrid.retrieve("alpha beta", top_k=5)
    assert {r.chunk_id for r in results} == {"only_bm25", "only_dense"}

    # A zero weight disables that arm entirely.
    lexical_only = HybridRetriever(
        rrf_k=1.0, bm25_weight=1.0, dense_weight=0.0
    ).index(chunks)
    results = lexical_only.retrieve("alpha beta", top_k=5)
    assert [r.chunk_id for r in results] == ["only_bm25"]


class FakeScorer:
    """Deterministic pair scorer; scores by shared-word count."""

    model_id = "fake-scorer-v1"
    revision = "test"

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        q_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [
            float(len(q_words & set(re.findall(r"[a-z0-9]+", t.lower()))))
            for t in texts
        ]


def test_cross_encoder_reranking(sample_chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever().index(sample_chunks)
    candidates = hybrid.retrieve("GOVERN culture and processes", top_k=3)

    reranker = CrossEncoderReranker(FakeScorer(), max_candidates=20)
    reranked, latency_ms = reranker.rerank(
        "GOVERN culture and processes", candidates, top_k=2
    )

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].retriever_name == "cross_encoder"
    assert latency_ms >= 0.0


def test_lexical_overlap_reranking(sample_chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever().index(sample_chunks)
    candidates = hybrid.retrieve("GOVERN culture and processes", top_k=3)

    reranker = LexicalOverlapReranker()
    reranked, latency_ms = reranker.rerank(
        "GOVERN culture and processes", candidates, top_k=2
    )

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].retriever_name == "lexical_rerank"
    assert latency_ms >= 0.0
