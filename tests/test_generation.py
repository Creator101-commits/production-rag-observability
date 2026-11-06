"""Tests for prompt construction, fake generation provider, and abstention."""

from __future__ import annotations

from rag_system.generation.base import GenerationRequest
from rag_system.generation.prompts import build_context_block, build_rag_prompt
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.retrieval.schema import Evidence


def test_build_context_and_prompt() -> None:
    evidence = [
        Evidence(
            chunk_id="chunk_1",
            document_id="doc1",
            document_title="NIST AI RMF",
            page_start=1,
            page_end=1,
            text="Executive summary content.",
            score=0.8,
            rank=1,
            retriever_name="hybrid",
        )
    ]
    context = build_context_block(evidence)
    assert "[chunk_1]" in context
    assert "Page 1" in context

    prompt = build_rag_prompt("What is this?", evidence)
    assert "--- USER QUESTION ---" in prompt
    assert "What is this?" in prompt


def test_fake_generator_answered() -> None:
    generator = DeterministicFakeProvider(default_abstain_threshold=0.30)
    evidence = [
        Evidence(
            chunk_id="chunk_1",
            document_id="doc1",
            document_title="NIST AI RMF",
            page_start=1,
            page_end=1,
            text="The GOVERN function establishes risk appetite and oversight.",
            score=0.85,
            rank=1,
            retriever_name="hybrid",
        )
    ]
    req = GenerationRequest(
        query="What does GOVERN do?",
        evidence_list=evidence,
        evidence_score_threshold=0.30,
    )
    resp = generator.generate(req)

    assert resp.status == "answered"
    assert resp.abstention_reason is None
    assert "chunk_1" in resp.citations
    assert "[chunk_1]" in resp.answer
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0


def test_fake_generator_abstain() -> None:
    generator = DeterministicFakeProvider(default_abstain_threshold=0.30)
    # Low score evidence below threshold
    evidence = [
        Evidence(
            chunk_id="chunk_irrelevant",
            document_id="doc1",
            document_title="NIST AI RMF",
            page_start=1,
            page_end=1,
            text="Unrelated text about other topics.",
            score=0.15,
            rank=1,
            retriever_name="hybrid",
        )
    ]
    req = GenerationRequest(
        query="What is the stock price of Apple?",
        evidence_list=evidence,
        evidence_score_threshold=0.30,
    )
    resp = generator.generate(req)

    assert resp.status == "abstained"
    assert resp.abstention_reason is not None
    assert "cannot answer" in resp.answer.lower()
    assert len(resp.citations) == 0
