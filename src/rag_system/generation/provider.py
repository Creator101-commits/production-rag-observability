"""Deterministic test and offline generation provider."""

from __future__ import annotations

import re
import time

from rag_system.generation.base import GenerationProvider, GenerationRequest, GenerationResponse
from rag_system.generation.prompts import build_rag_prompt


class DeterministicFakeProvider(GenerationProvider):
    """Deterministic offline generation provider for CI, testing, and benchmark replay."""

    def __init__(self, default_abstain_threshold: float = 0.30) -> None:
        self.default_abstain_threshold = default_abstain_threshold

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate deterministic citation-grounded response."""
        start_time = time.perf_counter()

        prompt = build_rag_prompt(request.query, request.evidence_list)
        prompt_tokens = len(re.findall(r"\w+|[^\w\s]", prompt))

        # Check evidence threshold
        threshold = (
            request.evidence_score_threshold
            if request.evidence_score_threshold is not None
            else self.default_abstain_threshold
        )

        valid_evidence = [ev for ev in request.evidence_list if ev.score >= threshold]

        if not valid_evidence:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResponse(
                answer="I cannot answer this question based on the provided technical documents.",
                status="abstained",
                citations=[],
                claims=[],
                abstention_reason="insufficient evidence in retrieved context",
                prompt_tokens=prompt_tokens,
                completion_tokens=15,
                latency_ms=latency_ms,
            )

        # Synthesize answers from top valid evidence
        top_ev = valid_evidence[0]
        summary_sentence = top_ev.text
        citation_id = top_ev.chunk_id

        claims = [f"{summary_sentence} [{citation_id}]"]

        # If a second piece of evidence is also strong, include it
        if len(valid_evidence) > 1 and valid_evidence[1].score >= threshold:
            second_ev = valid_evidence[1]
            second_sentence = second_ev.text
            claims.append(f"{second_sentence} [{second_ev.chunk_id}]")

        answer = " ".join(claims)
        citations = [ev.chunk_id for ev in valid_evidence[: len(claims)]]
        completion_tokens = len(re.findall(r"\w+|[^\w\s]", answer))
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return GenerationResponse(
            answer=answer,
            status="answered",
            citations=citations,
            claims=claims,
            abstention_reason=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
