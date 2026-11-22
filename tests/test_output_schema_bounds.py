"""OUTPUT_SCHEMA bounds enforced by the real lm-format-enforcer parser.

These tests feed candidate JSON to ``JsonSchemaParser`` character by
character, the same mechanism constrained decoding uses, so they verify
the schema itself without a tokenizer or model. ``lm-format-enforcer``
is a dev dependency for exactly this reason.
"""

from __future__ import annotations

import json

import pytest
from lmformatenforcer import JsonSchemaParser

from rag_system.generation.hf_local import OUTPUT_SCHEMA


def _schema_accepts(text: str) -> bool:
    """True when every character is allowed and the doc can end validly."""
    parser = JsonSchemaParser(OUTPUT_SCHEMA)
    for ch in text:
        if ch not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(ch)
    return parser.can_end()


def _payload(**overrides: object) -> str:
    base: dict[str, object] = {
        "answer": "The framework recommends testing AI systems before deployment.",
        "claims": [
            {
                "text": "AI systems should be tested before deployment.",
                "citations": ["C1"],
            }
        ],
        "abstain": False,
        "reason": None,
    }
    base.update(overrides)
    return json.dumps(base)


def test_reason_uses_anyof_not_type_union() -> None:
    # Regression: {"type": ["string", "null"], "maxLength": N} is not
    # enforced by lm-format-enforcer; anyOf branches are.
    reason = OUTPUT_SCHEMA["properties"]["reason"]
    assert "anyOf" in reason
    assert "maxLength" not in reason


def test_valid_example_accepted() -> None:
    assert _schema_accepts(_payload())


def test_reason_null_accepted() -> None:
    assert _schema_accepts(_payload(reason=None))


def test_reason_string_accepted() -> None:
    assert _schema_accepts(_payload(abstain=True, claims=[], reason="insufficient evidence"))


@pytest.mark.parametrize("size", [301, 400])
def test_answer_over_max_rejected(size: int) -> None:
    assert not _schema_accepts(_payload(answer="x" * size))


def test_answer_at_max_accepted() -> None:
    assert _schema_accepts(_payload(answer="x" * 300))


def test_claim_text_over_max_rejected() -> None:
    claims = [
        {"text": "y" * 301, "citations": ["C1"]},
    ]
    assert not _schema_accepts(_payload(claims=claims))


def test_claims_over_max_items_rejected() -> None:
    claims = [
        {"text": f"claim {i}", "citations": ["C1"]} for i in range(5)
    ]
    assert not _schema_accepts(_payload(claims=claims))


def test_claims_at_max_items_accepted() -> None:
    claims = [
        {"text": f"claim {i}", "citations": ["C1"]} for i in range(4)
    ]
    assert _schema_accepts(_payload(claims=claims))


@pytest.mark.parametrize("count", [0, 4])
def test_claim_citations_out_of_bounds_rejected(count: int) -> None:
    claims = [
        {"text": "a claim", "citations": [f"C{i}" for i in range(count)]},
    ]
    assert not _schema_accepts(_payload(claims=claims))


def test_claim_citations_at_bounds_accepted() -> None:
    for count in (1, 3):
        claims = [
            {"text": "a claim", "citations": [f"C{i}" for i in range(count)]},
        ]
        assert _schema_accepts(_payload(claims=claims))


@pytest.mark.parametrize("size", [161, 300])
def test_reason_over_max_rejected(size: int) -> None:
    assert not _schema_accepts(_payload(reason="z" * size))


def test_reason_at_max_accepted() -> None:
    assert _schema_accepts(_payload(reason="z" * 160))
