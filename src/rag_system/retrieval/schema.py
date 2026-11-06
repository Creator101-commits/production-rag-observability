"""Retrieval evidence representations and search outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """Retrieved evidence item with full source provenance and score."""

    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int
    retriever_name: str
