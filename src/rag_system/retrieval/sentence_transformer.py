"""Sentence-transformer dense retriever with normalized inner product.

The encoder is injected behind ``TextEncoder`` so tests use a deterministic
fake and never download models; ``SentenceTransformerEncoder`` wraps the
real model and is only built on Kaggle or explicit neural installs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_system.ingestion.schema import Chunk
from rag_system.retrieval.schema import Evidence


class TextEncoder(Protocol):
    """Encodes texts into fixed-size embedding vectors."""

    model_id: str
    revision: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float array; rows need not be normalized."""
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (mat / norms).astype(np.float32)


class SentenceTransformerEncoder:
    """Real encoder backed by sentence-transformers (lazy import)."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_id = model_id
        self.revision = revision or "main"
        self._model = SentenceTransformer(model_id, revision=revision)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=False),
            dtype=np.float32,
        )


class SentenceTransformerRetriever:
    """Exact normalized inner-product search over encoder embeddings.

    ``query_prefix`` carries model-specific instructions (for example the
    BGE retrieval instruction); ``passage_prefix`` is usually empty.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        retriever_name: str = "dense",
    ) -> None:
        self.encoder = encoder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.retriever_name = retriever_name
        self.chunks: list[Chunk] = []
        self.doc_vectors: np.ndarray | None = None
        self.is_indexed = False

    @property
    def model_id(self) -> str:
        return self.encoder.model_id

    @property
    def revision(self) -> str:
        return self.encoder.revision

    def index(self, chunks: list[Chunk]) -> SentenceTransformerRetriever:
        """Encode all chunks and store the normalized embedding matrix."""
        self.chunks = list(chunks)
        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")
        texts = [self.passage_prefix + c.text for c in chunks]
        self.doc_vectors = _normalize(self.encoder.encode(texts))
        self.is_indexed = True
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve top chunks by normalized inner product."""
        if top_k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed or self.doc_vectors is None:
            raise RuntimeError("SentenceTransformerRetriever must be indexed first.")

        q_vec = _normalize(
            self.encoder.encode([self.query_prefix + query])
        )[0]
        scores = self.doc_vectors @ q_vec

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
        ]
