"""Deterministic test and offline generation provider."""

from __future__ import annotations

import re
import time

from rag_system.generation.base import (
    Claim,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
)
from rag_system.generation.prompts import build_rag_prompt

ABSTAIN_ANSWER = (
    "I cannot answer this question based on the provided technical documents."
)


class DeterministicFakeProvider(GenerationProvider):
    """Deterministic offline provider for CI, testing, and benchmark replay.

    Emits the structured claim/label contract without any model calls.
    """

    def __init__(self, default_abstain_threshold: float = 0.30) -> None:
        self.default_abstain_threshold = default_abstain_threshold

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a deterministic structured response from evidence."""
        start_time = time.perf_counter()

        prompt = build_rag_prompt(
            request.query, request.evidence_list, max_tokens=request.max_context_tokens
        )
        prompt_tokens = len(re.findall(r"\w+|[^\w\s]", prompt))

        threshold = (
            request.evidence_score_threshold
            if request.evidence_score_threshold is not None
            else self.default_abstain_threshold
        )

        valid = [
            (i, ev)
            for i, ev in enumerate(request.evidence_list)
            if ev.score >= threshold
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not valid:
            return GenerationResponse(
                answer=ABSTAIN_ANSWER,
                status="abstained",
                abstain=True,
                abstain_reason="insufficient evidence score",
                prompt_tokens=prompt_tokens,
                completion_tokens=15,
                latency_ms=latency_ms,
            )

        claims = tuple(
            Claim(text=ev.text, citation_labels=(f"c{i + 1}",))
            for i, ev in valid[:2]
        )
        # The resolver renders the final answer from claims; leave ``answer``
        # empty so claim text is not duplicated in the emitted response.
        answer = ""
        completion_tokens = sum(
            len(re.findall(r"\w+|[^\w\s]", claim.text)) for claim in claims
        )

        return GenerationResponse(
            answer=answer,
            status="answered",
            claims=claims,
            abstain=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
