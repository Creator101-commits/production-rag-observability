"""Token-aware text chunking with page boundary and provenance preservation."""

from __future__ import annotations

import re

from rag_system.ingestion.schema import Chunk, Document, Page, normalize_text


def approximate_token_count(text: str) -> int:
    """Approximate word/token count using whitespace and punctuation tokens."""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


def chunk_page(
    doc: Document,
    page: Page,
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    heading: str = "",
) -> list[Chunk]:
    """Chunk a single page into overlapping segments with stable provenance."""
    clean_text = normalize_text(page.text)
    words = clean_text.split()
    if not words:
        return []

    # If page fits in one chunk
    if len(words) <= chunk_size_tokens:
        chunk_id = Chunk.generate_chunk_id(doc.document_id, page.page_number, 0, clean_text)
        return [
            Chunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                document_title=doc.title,
                page_start=page.page_number,
                page_end=page.page_number,
                heading=heading,
                text=clean_text,
                token_count=approximate_token_count(clean_text),
            )
        ]

    chunks: list[Chunk] = []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    idx = 0
    start = 0

    while start < len(words):
        end = min(len(words), start + chunk_size_tokens)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunk_id = Chunk.generate_chunk_id(doc.document_id, page.page_number, idx, chunk_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                document_title=doc.title,
                page_start=page.page_number,
                page_end=page.page_number,
                heading=heading,
                text=chunk_text,
                token_count=approximate_token_count(chunk_text),
            )
        )
        idx += 1
        start += step

    return chunks


def chunk_document(
    doc: Document,
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
) -> list[Chunk]:
    """Chunk all pages of a document and wire previous/next pointers."""
    raw_chunks: list[Chunk] = []
    for page in doc.pages:
        page_chunks = chunk_page(
            doc,
            page,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        raw_chunks.extend(page_chunks)

    # Wire previous and next chunk links
    linked_chunks: list[Chunk] = []
    n = len(raw_chunks)
    for i, c in enumerate(raw_chunks):
        prev_id = raw_chunks[i - 1].chunk_id if i > 0 else None
        next_id = raw_chunks[i + 1].chunk_id if i < n - 1 else None

        linked_chunks.append(
            Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                page_start=c.page_start,
                page_end=c.page_end,
                heading=c.heading,
                text=c.text,
                token_count=c.token_count,
                previous_chunk_id=prev_id,
                next_chunk_id=next_id,
            )
        )

    return linked_chunks
