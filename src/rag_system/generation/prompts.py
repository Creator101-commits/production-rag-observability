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

_STRUCTURED_EXAMPLE = (
    '{"answer": "Water boils at 100 C.", "claims": [{"text": "Water boils '
    'at 100 C.", "citations": ["c1"]}], "abstain": false, "reason": null}'
)

STRUCTURED_OUTPUT_INSTRUCTION = """Respond with a single JSON object and nothing else:
{"answer": string, "claims": [{"text": string, "citations": [string]}],
 "abstain": boolean, "reason": string|null}
Rules:
- "answer" is a one-sentence summary; detailed statements go in "claims".
- Every claim must cite at least one evidence label like "c1".
- If the evidence cannot support an answer, set "abstain" to true, give a
  short "reason", and leave "claims" empty. Never write a plain-text refusal.
- Never cite labels that are not shown in the context, and never invent facts.
Example:
""" + _STRUCTURED_EXAMPLE + "\n"

SYSTEM_PROMPT = (
    "You are a precise JSON-only answering engine. You answer questions "
    "using only the provided evidence passages and always reply with one "
    "JSON object matching the required schema.\n\n"
    + STRUCTURED_OUTPUT_INSTRUCTION
)


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


def build_rag_prompt(
    query: str, evidence_list: list[Evidence], max_tokens: int = 2048
) -> str:
    """Build full user prompt with context block and query."""
    context = build_context_block(evidence_list, max_tokens=max_tokens)
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT PASSAGES ---\n{context}\n\n"
        f"--- USER QUESTION ---\n{query}\n\n"
        "--- STRUCTURED ANSWER ---"
    )


def build_labeled_context(evidence_list: list[Evidence], max_tokens: int = 2048) -> str:
    """Format evidence as ``[cN]`` labeled blocks for structured generation."""
    blocks: list[str] = []
    total_words = 0

    for i, ev in enumerate(evidence_list, start=1):
        chunk_word_count = len(ev.text.split())
        if total_words + chunk_word_count > max_tokens:
            break

        pages = (
            str(ev.page_start)
            if ev.page_start == ev.page_end
            else f"{ev.page_start}-{ev.page_end}"
        )
        header = f"[c{i}] ({ev.document_title}, page {pages}):"
        blocks.append(f"{header}\n{ev.text}")
        total_words += chunk_word_count

    return "\n\n".join(blocks)


def build_structured_prompt(
    query: str, evidence_list: list[Evidence], max_tokens: int = 2048
) -> str:
    """Build the user-turn prompt; the contract lives in SYSTEM_PROMPT."""
    context = build_labeled_context(evidence_list, max_tokens=max_tokens)
    return (
        f"--- EVIDENCE ---\n{context}\n\n"
        f"--- QUESTION ---\n{query}"
    )
