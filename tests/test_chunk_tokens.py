"""Tests for page-aware token chunking and tokenizer protocols."""

from __future__ import annotations

from rag_system.ingestion.chunk_tokens import chunk_document_tokens
from rag_system.ingestion.schema import Document, Page
from rag_system.ingestion.tokens import RegexTokenizer


def _doc(pages: list[str]) -> Document:
    return Document(
        document_id="doc_t",
        title="Token Doc",
        source_url="https://example.com",
        version="test",
        source_sha256="a" * 64,
        pages=[Page(page_number=i + 1, text=t) for i, t in enumerate(pages)],
    )


def test_chunk_document_tokens_single_chunk() -> None:
    doc = _doc(["short text here"])
    chunks = chunk_document_tokens(doc, RegexTokenizer(), target_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].token_count == 3
    assert chunks[0].previous_chunk_id is None
    assert chunks[0].next_chunk_id is None


def test_chunk_document_tokens_overlap_and_links() -> None:
    words = " ".join(f"w{i}" for i in range(60))
    doc = _doc([words])
    chunks = chunk_document_tokens(
        doc, RegexTokenizer(), target_tokens=20, overlap_ratio=0.25
    )
    # step = 15 -> chunks covering tokens 0-19, 15-34, 30-49, 45-59
    assert len(chunks) == 4
    assert chunks[-1].token_count == 15
    for i, c in enumerate(chunks):
        assert c.token_count <= 20
        if i > 0:
            assert c.previous_chunk_id == chunks[i - 1].chunk_id
        if i < len(chunks) - 1:
            assert c.next_chunk_id == chunks[i + 1].chunk_id


def test_chunk_document_tokens_page_ranges() -> None:
    page1 = " ".join(f"a{i}" for i in range(30))
    page2 = " ".join(f"b{i}" for i in range(30))
    doc = _doc([page1, page2])
    chunks = chunk_document_tokens(
        doc, RegexTokenizer(), target_tokens=25, overlap_ratio=0.0
    )
    assert len(chunks) >= 2
    # First chunk covers only page 1 tokens; later chunks span pages.
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[-1].page_end == 2
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning, "expected at least one chunk spanning the page boundary"


def test_chunk_document_tokens_skips_empty_pages() -> None:
    doc = _doc(["real text", "", "more real text"])
    chunks = chunk_document_tokens(doc, RegexTokenizer(), target_tokens=100)
    assert len(chunks) == 1
    # The single chunk spans pages 1 and 3 (page 2 has no text).
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 3


def test_chunk_ids_stable_and_config_sensitive() -> None:
    doc = _doc(["alpha beta gamma delta epsilon"])
    tok = RegexTokenizer()
    a = chunk_document_tokens(doc, tok, target_tokens=3, overlap_ratio=0.0)
    b = chunk_document_tokens(doc, tok, target_tokens=3, overlap_ratio=0.0)
    c = chunk_document_tokens(doc, tok, target_tokens=4, overlap_ratio=0.0)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert {x.chunk_id for x in a}.isdisjoint({x.chunk_id for x in c})


def test_chunk_document_tokens_empty_doc() -> None:
    doc = _doc(["", "   "])
    assert chunk_document_tokens(doc, RegexTokenizer(), target_tokens=10) == []
