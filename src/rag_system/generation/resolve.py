"""Resolve provider citation labels (``c1..cN``) to verified chunk IDs.

The provider only ever sees positional labels; this module is the single
place where labels become chunk IDs. Unknown labels are rejected: claims
that lose all citations are dropped, and if nothing survives the response
becomes a controlled abstention.
"""

from __future__ import annotations

import re

from rag_system.generation.base import Claim, GenerationResponse
from rag_system.generation.provider import ABSTAIN_ANSWER
from rag_system.retrieval.schema import Evidence

_LABEL_IN_TEXT = re.compile(r"\[c(\d+)\]")


def _normalize_label(raw: str) -> str:
    """Canonicalize a provider label: strip brackets/space, lowercase."""
    return raw.strip().strip("[]()").strip().lower()


def _label_to_chunk_id(evidence: list[Evidence]) -> dict[str, str]:
    return {f"c{i}": ev.chunk_id for i, ev in enumerate(evidence, start=1)}


def _replace_labels_in_text(text: str, label_map: dict[str, str]) -> str:
    """Rewrite ``[cN]`` markers in free text to resolved ``[chunk_id]``."""

    def _sub(match: re.Match[str]) -> str:
        label = f"c{match.group(1)}"
        chunk_id = label_map.get(label)
        return f"[{chunk_id}]" if chunk_id else match.group(0)

    return _LABEL_IN_TEXT.sub(_sub, text)


def resolve_generation_response(
    response: GenerationResponse,
    evidence: list[Evidence],
) -> tuple[GenerationResponse, tuple[str, ...]]:
    """Map claim labels to chunk IDs and render the final answer text.

    Returns the resolved response plus human-readable warnings. Abstained
    and parse-failure responses pass through unchanged.
    """
    if response.status == "abstained" or response.abstain or response.parse_failure:
        return response, ()

    label_map = _label_to_chunk_id(evidence)
    warnings: list[str] = []

    resolved_claims: list[Claim] = []
    all_chunk_ids: list[str] = []
    for claim in response.claims:
        chunk_ids: list[str] = []
        for label in claim.citation_labels:
            chunk_id = label_map.get(_normalize_label(label))
            if chunk_id is None:
                warnings.append(f"rejected unknown citation label '{label}'")
                continue
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
        if not chunk_ids:
            warnings.append("dropped claim with no valid citations")
            continue
        for cid in chunk_ids:
            if cid not in all_chunk_ids:
                all_chunk_ids.append(cid)
        resolved_claims.append(
            Claim(
                text=claim.text,
                citation_labels=(),
                chunk_ids=tuple(chunk_ids),
            )
        )

    if not resolved_claims:
        return (
            GenerationResponse(
                answer=ABSTAIN_ANSWER,
                status="abstained",
                abstain=True,
                abstain_reason="no claims with valid citations",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                raw_output=response.raw_output,
            ),
            tuple(warnings),
        )

    rendered_claims = [
        claim.text + "".join(f" [{cid}]" for cid in claim.chunk_ids)
        for claim in resolved_claims
    ]
    lead = _replace_labels_in_text(response.answer, label_map).strip()
    parts = ([lead] if lead else []) + rendered_claims
    answer = " ".join(parts)

    return (
        GenerationResponse(
            answer=answer,
            status="answered",
            claims=tuple(resolved_claims),
            citations=tuple(all_chunk_ids),
            abstain=False,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            raw_output=response.raw_output,
        ),
        tuple(warnings),
    )
