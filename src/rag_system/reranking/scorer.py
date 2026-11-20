"""Pairwise relevance scorers for reranking.

``PairScorer`` is injected so tests use deterministic fakes; the real
implementation wraps a sentence-transformers cross-encoder and is only
built on Kaggle or explicit neural installs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PairScorer(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    model_id: str
    revision: str

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, aligned by position."""
        ...


class SentenceTransformerPairScorer:
    """Real scorer backed by a sentence-transformers cross-encoder."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self.model_id = model_id
        self.revision = revision or "main"
        self._model = CrossEncoder(model_id, revision=revision)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]
