"""Tests for citation extraction, resolution, and phantom detection."""

from __future__ import annotations

from rag_system.citations.validator import CitationValidator
from rag_system.retrieval.schema import Evidence


def test_citation_validator_valid() -> None:
    evidence = [
        Evidence(
            chunk_id="chunk_valid_1",
            document_id="doc1",
            document_title="NIST AI RMF",
            page_start=12,
            page_end=12,
            text="Seven characteristics define trustworthy AI systems.",
            score=0.85,
            rank=1,
            retriever_name="hybrid",
        )
    ]
    answer = "Trustworthy AI has seven characteristics [chunk_valid_1]."

    result = CitationValidator.validate(answer, evidence)
    assert result.is_valid is True
    assert len(result.resolved_citations) == 1
    assert result.resolved_citations[0].citation_id == "chunk_valid_1"
    assert result.resolved_citations[0].page_start == 12
    assert len(result.unsupported_citation_ids) == 0


def test_citation_validator_phantom_citation() -> None:
    evidence = [
        Evidence(
            chunk_id="chunk_real_1",
            document_id="doc1",
            document_title="NIST AI RMF",
            page_start=5,
            page_end=5,
            text="Real evidence passage.",
            score=0.9,
            rank=1,
            retriever_name="hybrid",
        )
    ]
    answer = "Claim grounded in real [chunk_real_1], but also citing invented [chunk_fake_99]."

    result = CitationValidator.validate(answer, evidence)
    assert result.is_valid is False
    assert len(result.resolved_citations) == 1
    assert "chunk_fake_99" in result.unsupported_citation_ids
    assert len(result.warnings) > 0
    assert "Phantom citation" in result.warnings[0]
