"""Citation extraction, ground-truth provenance verification, and hallucination rejection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from rag_system.generation.base import GenerationResponse
from rag_system.retrieval.schema import Evidence


@dataclass(frozen=True)
class ResolvedCitation:
    """Detailed metadata for a validated citation reference."""

    citation_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    text_snippet: str
    is_valid: bool


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of citation verification against retrieved evidence."""

    is_valid: bool
    resolved_citations: list[ResolvedCitation] = field(default_factory=list)
    unsupported_citation_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CitationValidator:
    """Verifies that generated citations reference genuine retrieved evidence."""

    @staticmethod
    def extract_citation_ids(answer_text: str) -> list[str]:
        """Extract bracketed citation identifiers from generated answer text."""
        # Match patterns like [nist_ai_rmf_1_0_p20_c0_abc123]
        matches = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", answer_text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        return deduped

    @classmethod
    def validate(
        cls,
        answer_text: str,
        provided_evidence: list[Evidence],
    ) -> ValidationResult:
        """Validate all citations in answer text against supplied evidence passages."""
        extracted_ids = cls.extract_citation_ids(answer_text)
        evidence_by_id = {ev.chunk_id: ev for ev in provided_evidence}

        resolved: list[ResolvedCitation] = []
        unsupported: list[str] = []
        warnings: list[str] = []

        for cite_id in extracted_ids:
            if cite_id in evidence_by_id:
                ev = evidence_by_id[cite_id]
                snippet = ev.text[:120] + "..." if len(ev.text) > 120 else ev.text
                resolved.append(
                    ResolvedCitation(
                        citation_id=cite_id,
                        document_id=ev.document_id,
                        document_title=ev.document_title,
                        page_start=ev.page_start,
                        page_end=ev.page_end,
                        text_snippet=snippet,
                        is_valid=True,
                    )
                )
            else:
                unsupported.append(cite_id)
                warnings.append(
                    f"Phantom citation '{cite_id}' was not in the retrieved evidence pool."
                )

        is_valid = len(unsupported) == 0

        return ValidationResult(
            is_valid=is_valid,
            resolved_citations=resolved,
            unsupported_citation_ids=unsupported,
            warnings=warnings,
        )

    @classmethod
    def enforce(
        cls, response: GenerationResponse, evidence: list[Evidence], minimum_citations: int = 1
    ) -> GenerationResponse:
        """Fail closed when an answered response has missing or unknown citations.

        ID membership does not establish semantic support for a claim.
        """
        result = cls.validate(response.answer, evidence)
        if response.status == "answered" and (
            not result.is_valid or len(result.resolved_citations) < minimum_citations
        ):
            return replace(
                response,
                status="abstained",
                answer="I cannot answer this question based on the provided technical documents.",
                claims=(),
                citations=(),
                abstain=True,
                abstain_reason="citation validation failed",
            )
        return response
