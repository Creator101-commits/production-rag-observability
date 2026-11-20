"""Tests for PDF normalization, extraction, integrity checks, and manifests."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from rag_system.ingestion.pdf import (
    CorpusDocumentSpec,
    extract_pdf_pages,
    ingest_corpus,
    ingest_document,
    normalize_pdf_text,
    sha256_bytes,
)


def _make_pdf(pages: list[str]) -> bytes:
    """Build a minimal text PDF; empty strings produce textless pages."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        content = DecodedStreamObject()
        if text:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            content.set_data(
                f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
            )
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)
        resources = DictionaryObject()
        resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font_ref})
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(content)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_normalize_pdf_text_dehyphenates_and_nfkc() -> None:
    raw = "hy-\nphenation   of   words\n\nﬁle ligature"
    out = normalize_pdf_text(raw)
    assert "hyphenation" in out
    assert "  " not in out
    assert "\n" not in out
    assert "ﬁ" not in out  # NFKC expands the fi ligature
    assert "fi" in out


def test_normalize_pdf_text_keeps_real_hyphens() -> None:
    out = normalize_pdf_text("state-of-the-art\nresults")
    assert "state-of-the-art" in out


def test_extract_pdf_pages_flags_empty_pages() -> None:
    pdf = _make_pdf(["First page text here", "", "Third page"])
    pages = extract_pdf_pages(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert pages[0].text == "First page text here"
    assert pages[1].extraction_failed is True
    assert pages[2].extraction_failed is False


def test_ingest_document_verifies_sha_and_manifest(tmp_path: Path) -> None:
    pdf = _make_pdf(["Alpha content", "Beta content"])
    spec = CorpusDocumentSpec(
        document_id="doc_a",
        title="Doc A",
        url="https://example.com/a.pdf",
        expected_sha256=sha256_bytes(pdf),
        expected_pages=2,
    )
    path = tmp_path / "doc_a.pdf"
    path.write_bytes(pdf)

    doc, manifest = ingest_document(spec, path)
    assert doc.document_id == "doc_a"
    assert doc.source_sha256 == spec.expected_sha256
    assert doc.total_pages == 2
    assert doc.pages[0].page_number == 1
    assert manifest["page_count"] == 2
    assert manifest["failed_pages"] == []
    assert manifest["byte_size"] == len(pdf)
    assert manifest["parser_version"]


def test_ingest_document_rejects_sha_mismatch(tmp_path: Path) -> None:
    pdf = _make_pdf(["content"])
    spec = CorpusDocumentSpec(
        document_id="doc_b",
        title="Doc B",
        url="https://example.com/b.pdf",
        expected_sha256="0" * 64,
    )
    path = tmp_path / "doc_b.pdf"
    path.write_bytes(pdf)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ingest_document(spec, path)


def test_ingest_document_rejects_page_count_mismatch(tmp_path: Path) -> None:
    pdf = _make_pdf(["one", "two"])
    spec = CorpusDocumentSpec(
        document_id="doc_c",
        title="Doc C",
        url="https://example.com/c.pdf",
        expected_sha256=sha256_bytes(pdf),
        expected_pages=5,
    )
    path = tmp_path / "doc_c.pdf"
    path.write_bytes(pdf)
    with pytest.raises(ValueError, match="expected 5 pages"):
        ingest_document(spec, path)


def test_ingest_corpus_no_download(tmp_path: Path) -> None:
    pdf_a = _make_pdf(["Doc A page one"])
    pdf_b = _make_pdf(["Doc B page one", "Doc B page two"])
    for doc_id, pdf in (("doc_a", pdf_a), ("doc_b", pdf_b)):
        d = tmp_path / doc_id
        d.mkdir()
        (d / f"{doc_id}.pdf").write_bytes(pdf)

    specs = (
        CorpusDocumentSpec(
            document_id="doc_a",
            title="Doc A",
            url="https://example.com/a.pdf",
            expected_sha256=sha256_bytes(pdf_a),
            expected_pages=1,
        ),
        CorpusDocumentSpec(
            document_id="doc_b",
            title="Doc B",
            url="https://example.com/b.pdf",
            expected_sha256=sha256_bytes(pdf_b),
            expected_pages=2,
        ),
    )
    docs, manifest = ingest_corpus(specs, tmp_path, download=False)
    assert [d.document_id for d in docs] == ["doc_a", "doc_b"]
    assert len(manifest["documents"]) == 2
    assert manifest["documents"][1]["page_count"] == 2
