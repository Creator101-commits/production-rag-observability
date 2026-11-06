"""Prompt templates and context builders for grounded RAG generation."""

from __future__ import annotations

from rag_system.retrieval.schema import Evidence

RAG_SYSTEM_PROMPT = """You are a technical question-answering assistant
for the NIST AI Risk Management Framework.
Answer questions strictly based on the provided numbered evidence passages.
Every factual statement must cite the source chunk identifier in brackets, e.g. [chunk_id].
If the evidence does not contain sufficient factual support to answer the question, state:
"I cannot answer this question based on the provided technical documents." and abstain.
Never invent external facts, citations, or references not present in the context.
"""


def build_context_block(evidence_list: list[Evidence], max_tokens: int = 2048) -> str:
    """Format evidence passages with chunk IDs into a prompt context block."""
    blocks: list[str] = []
    total_words = 0

    for ev in evidence_list:
        chunk_word_count = len(ev.text.split())
        if total_words + chunk_word_count > max_tokens:
            break

        header = f"[{ev.chunk_id}] (Source: {ev.document_title}, Page {ev.page_start}):"
        block = f"{header}\n{ev.text}"
        blocks.append(block)
        total_words += chunk_word_count

    return "\n\n".join(blocks)


def build_rag_prompt(query: str, evidence_list: list[Evidence]) -> str:
    """Build full user prompt with context block and query."""
    context = build_context_block(evidence_list)
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT PASSAGES ---\n{context}\n\n"
        f"--- USER QUESTION ---\n{query}\n\n"
        "--- STRUCTURED ANSWER ---"
    )
