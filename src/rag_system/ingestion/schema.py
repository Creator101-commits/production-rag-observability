"""Data structures representing raw documents, pages, and token-aware chunks."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Single page extracted from a source technical document."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Document:
    """Source technical document with provenance metadata and pages."""

    document_id: str
    title: str
    source_url: str
    version: str
    pages: list[Page] = field(default_factory=list)
    # SHA-256 of the exact source bytes; empty for synthetic fixtures.
    source_sha256: str = ""

    @property
    def total_pages(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class Chunk:
    """Token-aware text chunk with source provenance and navigation links."""

    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    heading: str
    text: str
    token_count: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None

    @staticmethod
    def generate_chunk_id(document_id: str, page_start: int, chunk_idx: int, text: str) -> str:
        """Create a deterministic, hash-verified chunk identifier."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{document_id}_p{page_start}_c{chunk_idx}_{content_hash}"


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode conservatively."""
    norm = unicodedata.normalize("NFKC", text)
    lines = [line.strip() for line in norm.splitlines() if line.strip()]
    return " ".join(lines)
