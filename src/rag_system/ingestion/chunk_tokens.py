"""Page-aware token chunking over a whole document token stream.

Chunks may span page boundaries but always record the exact
``page_start``/``page_end`` they cover. Chunk IDs are stable hashes of
the source SHA-256, character offsets, and chunking configuration.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right

from rag_system.ingestion.schema import Chunk, Document, normalize_text
from rag_system.ingestion.tokens import Tokenizer


def _config_hash(target_tokens: int, overlap_ratio: float, tokenizer_name: str) -> str:
    payload = json.dumps(
        {
            "target_tokens": target_tokens,
            "overlap_ratio": overlap_ratio,
            "tokenizer": tokenizer_name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_page_segments(doc: Document) -> tuple[str, list[tuple[int, int, int]]]:
    """Join normalized page texts; return full text and (start, end, page) spans."""
    full_parts: list[str] = []
    segments: list[tuple[int, int, int]] = []
    cursor = 0
    for page in doc.pages:
        text = normalize_text(page.text)
        if not text:
            continue
        if full_parts:
            full_parts.append("\n\n")
            cursor += 2
        start = cursor
        full_parts.append(text)
        cursor += len(text)
        segments.append((start, cursor, page.page_number))
    return "".join(full_parts), segments


def _page_for_offset(segments: list[tuple[int, int, int]], offset: int) -> int:
    """Return the page number whose segment contains ``offset``."""
    starts = [s[0] for s in segments]
    idx = bisect_right(starts, offset) - 1
    if idx < 0:
        idx = 0
    return segments[idx][2]


def chunk_document_tokens(
    doc: Document,
    tokenizer: Tokenizer,
    target_tokens: int,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Chunk a document by token count with page provenance preserved."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1).")

    full_text, segments = _build_page_segments(doc)
    if not segments:
        return []

    spans = tokenizer.token_spans(full_text)
    if not spans:
        return []

    overlap_tokens = int(round(target_tokens * overlap_ratio))
    step = max(1, target_tokens - overlap_tokens)
    cfg_hash = _config_hash(target_tokens, overlap_ratio, tokenizer.name)
    source_sha = doc.source_sha256 or doc.document_id

    raw: list[Chunk] = []
    start = 0
    while start < len(spans):
        end = min(len(spans), start + target_tokens)
        char_start = spans[start][0]
        char_end = spans[end - 1][1]
        chunk_text = full_text[char_start:char_end]
        page_start = _page_for_offset(segments, char_start)
        page_end = _page_for_offset(segments, spans[end - 1][0])
        digest = hashlib.sha256(
            f"{source_sha}|{char_start}|{char_end}|{cfg_hash}".encode()
        ).hexdigest()[:16]
        raw.append(
            Chunk(
                chunk_id=f"{doc.document_id}_{digest}",
                document_id=doc.document_id,
                document_title=doc.title,
                page_start=page_start,
                page_end=page_end,
                heading="",
                text=chunk_text,
                token_count=end - start,
            )
        )
        if end == len(spans):
            break
        start += step

    linked: list[Chunk] = []
    n = len(raw)
    for i, c in enumerate(raw):
        linked.append(
            Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                page_start=c.page_start,
                page_end=c.page_end,
                heading=c.heading,
                text=c.text,
                token_count=c.token_count,
                previous_chunk_id=raw[i - 1].chunk_id if i > 0 else None,
                next_chunk_id=raw[i + 1].chunk_id if i < n - 1 else None,
            )
        )
    return linked
