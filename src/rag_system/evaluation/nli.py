"""NLI-based claim-support estimation.

``NLIScorer`` is injected so tests use deterministic fakes; the real
implementation wraps ``cross-encoder/nli-deberta-v3-base`` and is only
built on Kaggle or explicit neural installs. Support is a model-based
estimate: a claim is supported when entailment is the argmax label.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_LABELS = ("contradiction", "entailment", "neutral")


def sentence_windows(text: str, size: int = 3, stride: int = 2) -> list[str]:
    """Split text into overlapping windows of ~``size`` sentences.

    Short texts (fewer sentences than ``size``) yield the whole text as a
    single window so nothing is dropped.
    """
    import re  # noqa: PLC0415

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= size:
        return [text] if text.strip() else []
    windows: list[str] = []
    for start in range(0, len(sentences) - size + 1, stride):
        windows.append(" ".join(sentences[start : start + size]))
    return windows


class NLIScorer(Protocol):
    """Predicts the NLI label for (premise, hypothesis) pairs."""

    model_id: str
    revision: str

    def predict_labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Return one NLI label per pair, aligned by position."""
        ...


class HFNLICrossEncoder:
    """Real NLI scorer via a sentence-transformers cross-encoder."""

    def __init__(self, model_id: str = DEFAULT_NLI_MODEL, revision: str | None = None) -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self.model_id = model_id
        self.revision = revision or "main"
        self._model = CrossEncoder(model_id, revision=revision)
        id2label = getattr(self._model.model.config, "id2label", {}) or {}
        self._index_to_label = {
            int(k): str(v).lower() for k, v in id2label.items()
        } or {i: label for i, label in enumerate(NLI_LABELS)}

    def predict_labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        if not pairs:
            return []
        scores = np.asarray(
            self._model.predict(pairs, show_progress_bar=False)
        ).reshape(len(pairs), -1)
        labels: list[str] = []
        for row in scores:
            idx = int(np.argmax(row))
            labels.append(self._index_to_label.get(idx, "neutral"))
        return labels


def claim_supported(label: str) -> bool:
    """Supported means entailment is the argmax NLI label."""
    return label == "entailment"
