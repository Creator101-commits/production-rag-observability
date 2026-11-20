"""Hashed lexical embedding retriever (offline baseline, not neural)."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.schema import Evidence


class HashedEmbeddingRetriever:
    """Exact inner-product retriever over deterministic hashed word vectors.

    This is a lexical baseline for offline tests and the CPU Docker image;
    it is not a neural embedding model.
    """

    retriever_name = "hashed_embedding"

    def __init__(self, embedding_dim: int = 128) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.dim = embedding_dim
        self.chunks: list[Chunk] = []
        self.doc_vectors: np.ndarray | None = None
        self.is_indexed: bool = False

    def _embed_text(self, text: str) -> np.ndarray:
        """Compute deterministic normalized hashed feature vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            weight = 1.0 / math.log2(len(word) + 2.0)
            vec[idx] += float(sign * weight)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def index(self, chunks: list[Chunk]) -> HashedEmbeddingRetriever:
        """Compute document embeddings and build vector matrix."""
        self.chunks = list(chunks)
        if not chunks:
            raise ValueError("Cannot build dense index on empty chunks collection.")

        embeddings = [self._embed_text(c.text) for c in chunks]
        self.doc_vectors = np.stack(embeddings, axis=0)
        self.is_indexed = True
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve top evidence chunks using exact inner product."""
        if top_k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed or self.doc_vectors is None:
            raise RuntimeError("HashedEmbeddingRetriever must be indexed before retrieve.")

        q_vec = self._embed_text(query)
        if float(np.linalg.norm(q_vec)) < 1e-8:
            return []

        scores = self.doc_vectors @ q_vec

        # Deterministic order: score descending, chunk_id ascending on ties.
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )[:top_k]

        return [
            Evidence(
                chunk_id=self.chunks[idx].chunk_id,
                document_id=self.chunks[idx].document_id,
                document_title=self.chunks[idx].document_title,
                page_start=self.chunks[idx].page_start,
                page_end=self.chunks[idx].page_end,
                text=self.chunks[idx].text,
                score=float(scores[idx]),
                rank=r + 1,
                retriever_name=self.retriever_name,
            )
            for r, idx in enumerate(order)
            if scores[idx] > -1.0
        ]
