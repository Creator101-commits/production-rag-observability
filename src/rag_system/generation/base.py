"""Generation interfaces and the structured answer contract.

Providers emit claims whose citations are positional labels ``c1..cN``
referring to ``GenerationRequest.evidence_list`` order. Code resolves
labels to chunk IDs; raw label text is never trusted as a citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_system.retrieval.schema import Evidence


@dataclass(frozen=True)
class Claim:
    """One generated claim with positional evidence labels.

    ``citation_labels`` holds provider-side labels (``c1``, ``c2``, ...).
    ``chunk_ids`` is filled by the resolver with verified chunk IDs.
    """

    text: str
    citation_labels: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationRequest:
    """Request payload sent to the generation provider."""

    query: str
    evidence_list: list[Evidence]
    temperature: float = 0.0
    evidence_score_threshold: float = 0.30
    max_context_tokens: int = 2048


@dataclass(frozen=True)
class GenerationResponse:
    """Structured provider output before citation resolution."""

    answer: str
    status: str  # "answered" or "abstained"
    claims: tuple[Claim, ...] = ()
    citations: tuple[str, ...] = ()
    abstain: bool = False
    abstain_reason: str | None = None
    parse_failure: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    # Raw provider output text; used for eval artifacts, never telemetry.
    raw_output: str = ""


class GenerationProvider(Protocol):
    """Protocol for pluggable generation backends."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a structured response over the supplied evidence."""
        ...
