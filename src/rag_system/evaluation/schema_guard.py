"""Post-generation guard: count records that violate OUTPUT_SCHEMA bounds.

Constrained decoding should make over-bound fields impossible; this guard
verifies it after the fact by re-parsing each record's ``raw_output`` and
checking the provider JSON fields against the schema bounds. A record
counts once even when several fields violate.

Only ``raw_output`` is checked: the stored ``answer`` is the
citation-resolved rendering (answer text plus claim texts and ``[chunk_id]``
markers appended by ``resolve_generation_response``), not the schema
``answer`` field, so its length is not schema-bounded.
"""

from __future__ import annotations

from typing import Any, cast

from rag_system.generation.hf_local import OUTPUT_SCHEMA, parse_structured_output

_SCHEMA = cast("dict[str, Any]", OUTPUT_SCHEMA["properties"])
_CLAIM_PROPS = cast("dict[str, Any]", _SCHEMA["claims"]["items"]["properties"])
_REASON_SCHEMA = cast("dict[str, Any]", _SCHEMA["reason"])
_REASON_BRANCH = next(
    b
    for b in cast("list[dict[str, Any]]", _REASON_SCHEMA.get("anyOf", [_REASON_SCHEMA]))
    if b.get("type") == "string"
)

ANSWER_MAX = int(_SCHEMA["answer"]["maxLength"])
CLAIMS_MAX = int(_SCHEMA["claims"]["maxItems"])
CLAIM_TEXT_MAX = int(_CLAIM_PROPS["text"]["maxLength"])
CITATIONS_MIN = int(_CLAIM_PROPS["citations"]["minItems"])
CITATIONS_MAX = int(_CLAIM_PROPS["citations"]["maxItems"])
REASON_MAX = int(_REASON_BRANCH["maxLength"])


def record_schema_violations(record: dict[str, Any]) -> list[str]:
    """Return the names of schema bounds this record violates (may be empty)."""
    violations: set[str] = set()

    raw = record.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = parse_structured_output(raw)
        except ValueError:
            parsed = None
        if parsed is not None:
            if len(parsed.answer) > ANSWER_MAX:
                violations.add("answer_length")
            if len(parsed.claims) > CLAIMS_MAX:
                violations.add("claims_count")
            for claim_text, labels in parsed.claims:
                if len(claim_text) > CLAIM_TEXT_MAX:
                    violations.add("claim_text_length")
                if not CITATIONS_MIN <= len(labels) <= CITATIONS_MAX:
                    violations.add("claim_citations_count")
            if parsed.reason is not None and len(parsed.reason) > REASON_MAX:
                violations.add("reason_length")

    # The stored claims keep the provider claim texts, so over-bound text
    # or an over-cap claim list still surfaces here when raw_output is
    # absent. The stored answer is exempt: it is the citation-resolved
    # rendering, not the schema answer field.
    claims = record.get("claims")
    if isinstance(claims, list):
        if len(claims) > CLAIMS_MAX:
            violations.add("claims_count")
        for claim in claims:
            if isinstance(claim, dict):
                stored_text = claim.get("text")
                if isinstance(stored_text, str) and len(stored_text) > CLAIM_TEXT_MAX:
                    violations.add("claim_text_length")
    return sorted(violations)


def count_schema_bound_violations(records: list[dict[str, Any]]) -> int:
    """Count records with at least one schema-bound violation."""
    return sum(1 for r in records if record_schema_violations(r))
