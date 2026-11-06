"""Dense semantic retriever using normalized vector embeddings."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.schema import Evidence


class DenseRetriever:
    """Exact cosine dense semantic retriever with normalized feature vectors."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.dim = embedding_dim
        self.chunks: list[Chunk] = []
        self.doc_vectors: np.ndarray | None = None
        self.is_indexed: bool = False

    def _embed_text(self, text: str) -> np.ndarray:
        """Compute deterministic normalized dense semantic embedding vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return vec

        for word in words:
            # Hash word into dimension index and pseudo-random directional weight
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            weight = 1.0 / math.log2(len(word) + 2.0)
            vec[idx] += float(sign * weight)

        # L2 normalize
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def index(self, chunks: list[Chunk]) -> DenseRetriever:
        """Compute document embeddings and build vector matrix."""
        self.chunks = list(chunks)
        if not chunks:
            raise ValueError("Cannot build dense index on empty chunks collection.")

        embeddings = [self._embed_text(c.text) for c in chunks]
        self.doc_vectors = np.stack(embeddings, axis=0)
        self.is_indexed = True
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve top evidence chunks using exact cosine similarity."""
        if not self.is_indexed or self.doc_vectors is None:
            raise RuntimeError("DenseRetriever must be indexed before calling retrieve.")

        q_vec = self._embed_text(query)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm < 1e-8:
            return []

        # Cosine similarities
        scores = self.doc_vectors @ q_vec

        ranked_indices = np.argsort(-scores)[:top_k]

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
                retriever_name="dense",
            )
            for r, idx in enumerate(ranked_indices)
            if scores[idx] > -1.0
        ]
