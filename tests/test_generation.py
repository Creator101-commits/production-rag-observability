"""Tests for prompt construction, providers, JSON parsing, and resolution."""

from __future__ import annotations

import json

import pytest

from rag_system.generation.base import Claim, GenerationRequest, GenerationResponse
from rag_system.generation.hf_local import (
    OUTPUT_SCHEMA,
    HFLocalProvider,
    parse_structured_output,
)
from rag_system.generation.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_labeled_context,
    build_rag_prompt,
    build_structured_prompt,
)
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.generation.resolve import resolve_generation_response
from rag_system.retrieval.schema import Evidence


def _ev(chunk_id: str, text: str, score: float = 0.9) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id="doc1",
        document_title="NIST AI RMF",
        page_start=1,
        page_end=1,
        text=text,
        score=score,
        rank=1,
        retriever_name="hybrid",
    )


def test_build_context_and_prompt() -> None:
    evidence = [_ev("chunk_1", "Executive summary content.")]
    context = build_context_block(evidence)
    assert "[chunk_1]" in context
    assert "Page 1" in context

    prompt = build_rag_prompt("What is this?", evidence)
    assert "--- USER QUESTION ---" in prompt
    assert "What is this?" in prompt


def test_build_labeled_context_and_structured_prompt() -> None:
    evidence = [_ev("chunk_1", "Executive summary content.")]
    context = build_labeled_context(evidence)
    assert "[c1]" in context
    assert "chunk_1" not in context

    prompt = build_structured_prompt("What is this?", evidence)
    # The JSON contract lives in the system prompt, not the user turn.
    assert '"abstain"' in SYSTEM_PROMPT
    assert "I cannot answer this question" not in SYSTEM_PROMPT
    assert "at most two sentences" in SYSTEM_PROMPT
    assert "at most 4 claims" in SYSTEM_PROMPT
    assert "[c1]" in prompt
    assert "What is this?" in prompt


def test_fake_generator_answered_and_resolved() -> None:
    generator = DeterministicFakeProvider(default_abstain_threshold=0.30)
    evidence = [
        _ev("chunk_1", "The GOVERN function establishes risk appetite and oversight."),
        _ev("chunk_2", "The MAP function identifies context and impacts.", score=0.5),
    ]
    req = GenerationRequest(query="What does GOVERN do?", evidence_list=evidence)
    resp = generator.generate(req)

    assert resp.status == "answered"
    assert resp.abstain is False
    assert len(resp.claims) == 2
    assert resp.claims[0].citation_labels == ("c1",)
    assert resp.claims[1].citation_labels == ("c2",)

    resolved, warnings = resolve_generation_response(resp, evidence)
    assert resolved.status == "answered"
    assert resolved.citations == ("chunk_1", "chunk_2")
    assert "[chunk_1]" in resolved.answer
    assert "[chunk_2]" in resolved.answer
    assert resolved.claims[0].chunk_ids == ("chunk_1",)
    assert warnings == ()


def test_fake_generator_abstain() -> None:
    generator = DeterministicFakeProvider(default_abstain_threshold=0.30)
    evidence = [_ev("chunk_low", "Unrelated text.", score=0.15)]
    req = GenerationRequest(query="Stock price?", evidence_list=evidence)
    resp = generator.generate(req)

    assert resp.status == "abstained"
    assert resp.abstain is True
    assert resp.abstain_reason is not None
    assert "cannot answer" in resp.answer.lower()
    assert resp.claims == ()


def test_parse_structured_output_valid() -> None:
    raw = json.dumps(
        {
            "answer": "GOVERN sets risk appetite.",
            "claims": [
                {"text": "GOVERN sets risk appetite.", "citations": ["c1"]},
                {"text": "MAP identifies context.", "citations": ["c1", "c2"]},
            ],
            "abstain": False,
            "reason": None,
        }
    )
    parsed = parse_structured_output(raw)
    assert parsed.answer.startswith("GOVERN")
    assert parsed.claims[0] == ("GOVERN sets risk appetite.", ("c1",))
    assert parsed.abstain is False


def test_parse_structured_output_fenced_and_embedded() -> None:
    raw = 'Sure! ```json\n{"answer": "A", "claims": [], "abstain": true, "reason": "none"}\n```'
    parsed = parse_structured_output(raw)
    assert parsed.abstain is True
    assert parsed.reason == "none"


@pytest.mark.parametrize(
    "raw",
    [
        "no json here",
        '{"answer": 5, "claims": [], "abstain": false, "reason": null}',
        '{"answer": "x", "claims": [{"text": ""}], "abstain": false, "reason": null}',
        '{"answer": "x", "claims": [{"text": "t", "citations": "c1"}], '
        '"abstain": false, "reason": null}',
        '{"answer": "x", "claims": [], "abstain": "yes", "reason": null}',
        '{"answer": "x", "claims": [{"text": "t"}], "abstain": true, "reason": null}',
    ],
)
def test_parse_structured_output_rejects_malformed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_structured_output(raw)


class _FakeEngine:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate_text(
        self, prompt: str, system: str = ""
    ) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.raw, 120, 40


def test_hf_local_provider_parses_and_labels() -> None:
    raw = json.dumps(
        {
            "answer": "GOVERN oversees risk.",
            "claims": [{"text": "GOVERN oversees risk.", "citations": ["c1"]}],
            "abstain": False,
            "reason": None,
        }
    )
    provider = HFLocalProvider(engine=_FakeEngine(raw))
    evidence = [_ev("chunk_a", "GOVERN text.")]
    resp = provider.generate(
        GenerationRequest(query="GOVERN?", evidence_list=evidence)
    )
    assert resp.status == "answered"
    assert resp.claims[0].citation_labels == ("c1",)
    # Labels stay unresolved until resolve_generation_response runs.
    assert resp.claims[0].chunk_ids == ()

    resolved, _ = resolve_generation_response(resp, evidence)
    assert resolved.citations == ("chunk_a",)
    assert "[chunk_a]" in resolved.answer


def test_hf_local_provider_parse_failure_abstains() -> None:
    provider = HFLocalProvider(engine=_FakeEngine("I cannot comply."))
    resp = provider.generate(
        GenerationRequest(query="Q?", evidence_list=[_ev("c1", "text")])
    )
    assert resp.status == "abstained"
    assert resp.parse_failure is True
    assert "cannot answer" in resp.answer.lower()


def test_output_schema_string_and_array_limits_enforced() -> None:
    """JsonSchemaParser must reject output beyond OUTPUT_SCHEMA limits.

    lm-format-enforcer is a neural-extra dependency, so this test skips in
    the base environment; it runs wherever the extra is installed.
    """
    lmfe = pytest.importorskip("lmformatenforcer")

    def feed(parser: object, text: str) -> object:
        for ch in text:
            parser = parser.add_character(ch)  # type: ignore[attr-defined]
        return parser

    # answer maxLength=300: the 301st content character is not allowed.
    parser = lmfe.JsonSchemaParser(OUTPUT_SCHEMA)
    parser = feed(parser, '{"answer":"' + "a" * 300)
    allowed = parser.get_allowed_characters()
    assert '"' in allowed
    assert "a" not in allowed

    # claims maxItems=4: a fifth item (comma) is not allowed.
    item = '{"text":"t","citations":["c1"]}'
    parser = lmfe.JsonSchemaParser(OUTPUT_SCHEMA)
    parser = feed(
        parser,
        '{"answer":"a","claims":[' + ",".join([item] * 4),
    )
    allowed = parser.get_allowed_characters()
    assert "]" in allowed
    assert "," not in allowed

    # citations minItems=1: an empty citations array is not allowed.
    parser = lmfe.JsonSchemaParser(OUTPUT_SCHEMA)
    parser = feed(
        parser, '{"answer":"a","claims":[{"text":"t","citations":['
    )
    allowed = parser.get_allowed_characters()
    assert "]" not in allowed


def test_hf_local_provider_model_abstain() -> None:
    raw = json.dumps(
        {"answer": "", "claims": [], "abstain": True, "reason": "no evidence"}
    )
    provider = HFLocalProvider(engine=_FakeEngine(raw))
    resp = provider.generate(
        GenerationRequest(query="Q?", evidence_list=[_ev("c1", "text")])
    )
    assert resp.status == "abstained"
    assert resp.abstain_reason == "no evidence"


def test_resolve_rejects_unknown_labels() -> None:
    evidence = [_ev("chunk_a", "text")]
    resp = GenerationResponse(
        answer="",
        status="answered",
        claims=(Claim(text="Claim one.", citation_labels=("c1", "c9")),),
    )
    resolved, warnings = resolve_generation_response(resp, evidence)
    assert resolved.status == "answered"
    assert resolved.claims[0].chunk_ids == ("chunk_a",)
    assert any("c9" in w for w in warnings)


def test_resolve_normalizes_label_formatting() -> None:
    evidence = [_ev("chunk_a", "text")]
    resp = GenerationResponse(
        answer="",
        status="answered",
        claims=(
            Claim(text="Claim one.", citation_labels=("[C1]", " c2 ", "(c1)")),
        ),
    )
    # c2 does not exist; [C1], " c2 "->c2 invalid, (c1)->c1 valid.
    resolved, warnings = resolve_generation_response(resp, evidence)
    assert resolved.claims[0].chunk_ids == ("chunk_a",)
    assert any("c2" in w for w in warnings)


def test_resolve_abstains_when_all_claims_invalid() -> None:
    evidence = [_ev("chunk_a", "text")]
    resp = GenerationResponse(
        answer="",
        status="answered",
        claims=(Claim(text="Bad claim.", citation_labels=("c7",)),),
    )
    resolved, _ = resolve_generation_response(resp, evidence)
    assert resolved.status == "abstained"
    assert resolved.abstain_reason == "no claims with valid citations"
