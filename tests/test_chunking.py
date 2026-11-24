"""Tests for document chunking, provenance, and navigation links."""

from __future__ import annotations

from rag_system.ingestion.chunk import approximate_token_count, chunk_document, chunk_page
from rag_system.ingestion.schema import Document, Page


def test_approximate_token_count() -> None:
    text = "NIST AI Risk Management Framework 1.0 (NIST AI 100-1)."
    count = approximate_token_count(text)
    assert count > 0


def test_chunk_page_fits_single_chunk() -> None:
    doc = Document(
        document_id="test_doc",
        title="Test Document",
        source_url="http://example.com",
        version="1.0",
        pages=[Page(page_number=1, text="Short paragraph explaining AI risk management.")],
    )
    chunks = chunk_page(doc, doc.pages[0], chunk_size_tokens=100, chunk_overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].document_id == "test_doc"
    assert "Short paragraph" in chunks[0].text


def test_chunk_document_links() -> None:
    words = [f"word{i}" for i in range(150)]
    doc = Document(
        document_id="doc_multi",
        title="Multi Page Document",
        source_url="http://example.com",
        version="1.0",
        pages=[
            Page(page_number=1, text=" ".join(words[:75])),
            Page(page_number=2, text=" ".join(words[75:])),
        ],
    )
    chunks = chunk_document(doc, chunk_size_tokens=40, chunk_overlap_tokens=10)
    assert len(chunks) >= 3

    assert chunks[0].previous_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[-1].next_chunk_id is None
