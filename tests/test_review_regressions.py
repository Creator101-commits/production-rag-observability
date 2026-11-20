"""Regression checks for citation enforcement and approximate token budgets."""

import pytest
from fastapi.testclient import TestClient

from rag_system.api.app import create_app
from rag_system.citations.validator import CitationValidator
from rag_system.generation.base import GenerationResponse
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.ingestion.chunk import approximate_token_count, chunk_page
from rag_system.ingestion.schema import Document, Page


@pytest.mark.parametrize("answer", ["Uncited answer.", "Bad citation [not_in_context]."])
def test_invalid_answer_citations_abstain(answer):
    response = GenerationResponse(answer=answer, status="answered", citations=["not_in_context"])
    result = CitationValidator.enforce(response, [])
    assert result.status == "abstained" and not result.claims and not result.citations


def test_http_endpoint_enforces_citations(monkeypatch):
    def bad_generation(self, request):
        return GenerationResponse(answer="Fabricated fact [phantom].", status="answered")

    monkeypatch.setattr(DeterministicFakeProvider, "generate", bad_generation)
    response = TestClient(create_app()).post("/query", json={"query": "GOVERN function"})
    assert response.status_code == 200
    assert response.json()["status"] == "abstained"
    assert "Fabricated" not in response.json()["answer"]


def test_punctuation_tokens_obey_chunk_limit():
    doc = Document("fixture", "fixture", "https://example.com", "fixture-v1")
    page = Page(1, "alpha,beta;gamma!delta?epsilon:zeta. eta")
    chunks = chunk_page(doc, page, chunk_size_tokens=4, chunk_overlap_tokens=1)
    assert len(chunks) > 1
    assert all(approximate_token_count(c.text) <= 4 for c in chunks)
    with pytest.raises(ValueError):
        chunk_page(doc, page, 4, 4)


def test_document_name_boilerplate_cannot_support_unanswerable_questions():
    from rag_system.evaluation.dataset import load_eval_questions
    from rag_system.evaluation.runner import BenchmarkRunner

    questions = [q for q in load_eval_questions() if not q.answerable]
    metrics = BenchmarkRunner().run_evaluation(questions)
    assert metrics.false_answer_rate == 0.0
