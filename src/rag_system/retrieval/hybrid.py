"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.bm25 import BM25Retriever
from rag_system.retrieval.dense import DenseRetriever
from rag_system.retrieval.schema import Evidence


class HybridRetriever:
    """Combines BM25 and Dense retrieval ranks via Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: float = 60.0) -> None:
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.chunk_map: dict[str, Chunk] = {}
        self.is_indexed: bool = False

    def index(self, chunks: list[Chunk]) -> HybridRetriever:
        """Index chunks across both lexical and dense engines."""
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self.is_indexed = True
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_depth: int = 20,
    ) -> list[Evidence]:
        """Perform reciprocal rank fusion over BM25 and dense candidate lists."""
        if not self.is_indexed:
            raise RuntimeError("HybridRetriever must be indexed before calling retrieve.")

        bm25_candidates = self.bm25.retrieve(query, top_k=candidate_depth)
        dense_candidates = self.dense.retrieve(query, top_k=candidate_depth)

        rrf_scores: dict[str, float] = defaultdict(float)

        for cand in bm25_candidates:
            rrf_scores[cand.chunk_id] += 1.0 / (self.rrf_k + cand.rank)

        for cand in dense_candidates:
            rrf_scores[cand.chunk_id] += 1.0 / (self.rrf_k + cand.rank)

        sorted_cands = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

        results: list[Evidence] = []
        for r, (chunk_id, score) in enumerate(sorted_cands):
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
