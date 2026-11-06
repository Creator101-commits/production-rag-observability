"""Generation interfaces and data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_system.retrieval.schema import Evidence


@dataclass(frozen=True)
class GenerationRequest:
    """Request payload sent to the LLM generation provider."""

    query: str
    evidence_list: list[Evidence]
    temperature: float = 0.0
    evidence_score_threshold: float = 0.30


@dataclass(frozen=True)
class GenerationResponse:
    """Structured response emitted by the LLM generation provider."""

    answer: str
    status: str  # "answered", "abstained", "partial"
    citations: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    abstention_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class GenerationProvider(Protocol):
    """Protocol for pluggable generation backends."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate structured response with citations over evidence context."""
        ...
