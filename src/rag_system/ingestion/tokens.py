"""Tokenizer protocols for token-aware chunking.

Real runs use the embedding model's tokenizer via ``HFTokenizer``;
tests and the offline path use ``RegexTokenizer``, a deterministic
approximation that never downloads anything.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer(Protocol):
    """Counts tokens and returns their character offsets in a text."""

    name: str

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) char offsets of each token, in order."""
        ...

    def count(self, text: str) -> int:
        """Return the token count of ``text``."""
        ...


class RegexTokenizer:
    """Deterministic whitespace/punctuation tokenizer for offline use."""

    name = "regex-approx-v1"

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.token_spans(text))


class HFTokenizer:
    """Wraps a Hugging Face fast tokenizer with offset mapping.

    Requires the ``neural`` extra; only constructed on real runs.
    """

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415 - lazy neural import

        self.name = f"hf:{model_id}@{(revision or 'main')[:12]}"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        return [
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
            if end > start
        ]

    def count(self, text: str) -> int:
        return len(
            self._tokenizer(text, add_special_tokens=False)["input_ids"]
        )
