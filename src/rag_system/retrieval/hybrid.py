"""Hybrid retrieval via weighted Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.hashed import HashedEmbeddingRetriever
from rag_system.retrieval.schema import Evidence


class RankedRetriever(Protocol):
    """Any retriever that indexes chunks and returns ranked Evidence."""

    def index(self, chunks: list[Chunk]) -> RankedRetriever:
        ...

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        ...


class HybridRetriever:
    """Combines lexical and embedding retrievers via weighted RRF.

    Score for a chunk = ``bm25_weight/(k+rank_bm25) + dense_weight/(k+rank_dense)``;
    a chunk absent from one list contributes only the other term.
    """

    def __init__(
        self,
        rrf_k: float = 60.0,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        bm25: BM25Retriever | None = None,
        dense: RankedRetriever | None = None,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or HashedEmbeddingRetriever()
        self.chunk_map: dict[str, Chunk] = {}
        self.is_indexed: bool = False

    def index(self, chunks: list[Chunk]) -> HybridRetriever:
        """Index chunks across both lexical and embedding engines."""
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self.is_indexed = True
        return self

    def candidate_lists(
        self, query: str, candidate_depth: int = 20
    ) -> tuple[list[Evidence], list[Evidence]]:
        """Return the BM25 and dense candidate lists for fusion."""
        if not self.is_indexed:
            raise RuntimeError("HybridRetriever must be indexed before calling retrieve.")
        return (
            self.bm25.retrieve(query, top_k=candidate_depth),
            self.dense.retrieve(query, top_k=candidate_depth),
        )

    def fuse(
        self,
        bm25_candidates: list[Evidence],
        dense_candidates: list[Evidence],
        top_k: int,
    ) -> list[Evidence]:
        """Weighted RRF fusion over two candidate lists."""
        if top_k <= 0:
            raise ValueError("Result count must be positive.")
        rrf_scores: dict[str, float] = defaultdict(float)

        if self.bm25_weight:
            for cand in bm25_candidates:
                rrf_scores[cand.chunk_id] += self.bm25_weight / (self.rrf_k + cand.rank)

        if self.dense_weight:
            for cand in dense_candidates:
                rrf_scores[cand.chunk_id] += self.dense_weight / (self.rrf_k + cand.rank)

        # Deterministic order: fused score descending, chunk_id ascending.
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
        top_ids = sorted_ids[:top_k]

        results: list[Evidence] = []
        for r, (chunk_id, score) in enumerate(top_ids):
            chunk = self.chunk_map[chunk_id]
            results.append(
                Evidence(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    score=float(score),
                    rank=r + 1,
                    retriever_name="hybrid_rrf",
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_depth: int = 20,
    ) -> list[Evidence]:
        """Perform weighted reciprocal rank fusion over candidate lists."""
        candidate_depth = max(candidate_depth, top_k)
        bm25_candidates, dense_candidates = self.candidate_lists(
            query, candidate_depth
        )
        return self.fuse(bm25_candidates, dense_candidates, top_k)
