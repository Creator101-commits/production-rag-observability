"""Tests for the post-generation schema-bound guard."""

from __future__ import annotations

import json

from rag_system.evaluation.schema_guard import (
    count_schema_bound_violations,
    record_schema_violations,
)


def _raw_output(**overrides: object) -> str:
    base: dict[str, object] = {
        "answer": "Grounded answer.",
        "claims": [{"text": "A claim.", "citations": ["C1"]}],
        "abstain": False,
        "reason": None,
    }
    base.update(overrides)
    return json.dumps(base)


def _record(raw: str | None = None, **fields: object) -> dict[str, object]:
    record: dict[str, object] = {"raw_output": raw, **fields}
    return record


def test_clean_record_has_no_violations() -> None:
    rec = _record(
        _raw_output(),
        answer="Grounded answer.",
        claims=[{"text": "A claim.", "chunk_ids": ["c1"]}],
    )
    assert record_schema_violations(rec) == []


def test_overlong_reason_in_raw_output_flagged() -> None:
    rec = _record(_raw_output(reason="r" * 200))
    assert record_schema_violations(rec) == ["reason_length"]


def test_overlong_answer_in_raw_output_flagged() -> None:
    rec = _record(_raw_output(answer="a" * 301))
    assert "answer_length" in record_schema_violations(rec)


def test_too_many_claims_in_raw_output_flagged() -> None:
    claims = [
        {"text": f"claim {i}", "citations": ["C1"]} for i in range(5)
    ]
    rec = _record(_raw_output(claims=claims))
    assert "claims_count" in record_schema_violations(rec)


def test_zero_citations_in_raw_output_flagged() -> None:
    claims = [{"text": "a claim", "citations": []}]
    rec = _record(_raw_output(claims=claims))
    assert "claim_citations_count" in record_schema_violations(rec)


def test_unparseable_raw_output_is_not_a_bound_violation() -> None:
    # Truncated JSON already surfaces as parse_failure/truncated; the guard
    # only counts records whose fields exceed schema bounds.
    rec = _record('{"answer": "cut off mid-stri', answer="", claims=[])
    assert record_schema_violations(rec) == []


def test_missing_raw_output_still_checks_stored_claims() -> None:
    rec = _record(
        None,
        answer="a" * 301,
        claims=[{"text": "t" * 301, "chunk_ids": []}],
    )
    # The stored answer is the citation-resolved rendering, not the schema
    # answer field, so only the claim text bound applies.
    assert record_schema_violations(rec) == ["claim_text_length"]


def test_resolved_answer_rendering_is_not_a_violation() -> None:
    # Smoke regression: constrained decoding caps the provider answer at
    # 300 chars, but resolve_generation_response appends claim texts and
    # [chunk_id] markers, so the stored answer legitimately exceeds 300.
    rec = _record(
        _raw_output(answer="a" * 300),
        answer="a" * 300 + " claim text [c1_chunk_id]",
        claims=[{"text": "claim text", "chunk_ids": ["c1_chunk_id"]}],
    )
    assert record_schema_violations(rec) == []


def test_stored_claims_over_max_flagged() -> None:
    rec = _record(
        None,
        answer="ok",
        claims=[{"text": f"c{i}", "chunk_ids": []} for i in range(5)],
    )
    assert "claims_count" in record_schema_violations(rec)


def test_count_counts_each_violating_record_once() -> None:
    bad = _record(_raw_output(reason="r" * 200, answer="a" * 301))
    good = _record(_raw_output())
    assert count_schema_bound_violations([bad, good, bad]) == 2
