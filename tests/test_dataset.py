"""Tests for the frozen JSONL evaluation set loader and validator."""

from __future__ import annotations

import json

import pytest

from rag_system.evaluation.dataset import (
    EvalQuestion,
    EvidenceRef,
    default_eval_path,
    load_eval_questions,
    split_questions,
    validate_eval_questions,
)


def test_load_frozen_eval_set() -> None:
    questions = load_eval_questions()
    assert len(questions) == 70
    dev = split_questions(questions, "dev")
    test = split_questions(questions, "test")
    assert len(dev) == 24
    assert len(test) == 46
    assert sum(1 for q in dev if q.answerable) == 19
    assert sum(1 for q in dev if not q.answerable) == 5
    assert sum(1 for q in test if q.answerable) == 37
    assert sum(1 for q in test if not q.answerable) == 9


def test_eval_question_fields() -> None:
    questions = load_eval_questions()
    q = questions[0]
    assert q.question_id == "q001"
    assert q.split == "dev"
    assert q.answerable is True
    assert q.evidence[0].document_id == "nist_ai_100_1"
    assert isinstance(q.evidence[0].pages, tuple)
    assert q.required_points
    assert all(isinstance(alt, str) for point in q.required_points for alt in point)
    assert q.expected_doc_ids == ("nist_ai_100_1",)


def test_unanswerable_questions_have_no_labels() -> None:
    questions = load_eval_questions()
    for q in questions:
        if not q.answerable:
            assert q.evidence == ()
            assert q.required_points == ()


def test_default_eval_path_finds_repo_file() -> None:
    path = default_eval_path()
    assert path.name == "nist_qa_v1.jsonl"
    assert path.is_file()


def _write_jsonl(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return str(path)


def _valid_row(**overrides) -> dict:
    row = {
        "question_id": "qX",
        "split": "dev",
        "category": "direct_lookup",
        "difficulty": "easy",
        "answerable": True,
        "question": "What is the GOVERN function?",
        "evidence": [{"document_id": "nist_ai_100_1", "pages": [8]}],
        "required_points": [["govern"]],
    }
    row.update(overrides)
    return row


def test_loader_rejects_bad_split(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_valid_row(split="holdout")])
    with pytest.raises(ValueError, match="split"):
        load_eval_questions(path)


def test_loader_rejects_duplicate_ids(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_valid_row(), _valid_row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_questions(path)


def test_loader_rejects_answerable_without_evidence(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_valid_row(evidence=[])])
    with pytest.raises(ValueError, match="evidence"):
        load_eval_questions(path)


def test_loader_rejects_unanswerable_with_evidence(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_valid_row(answerable=False)])
    with pytest.raises(ValueError, match="must not cite evidence"):
        load_eval_questions(path)


def test_loader_rejects_empty_required_point(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [_valid_row(required_points=[[]])])
    with pytest.raises(ValueError, match="required point"):
        load_eval_questions(path)


def test_validate_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_eval_questions([])


def test_split_questions_rejects_unknown() -> None:
    q = EvalQuestion(
        question_id="q1",
        split="dev",
        category="c",
        difficulty="easy",
        answerable=False,
        question="?",
        evidence=(),
        required_points=(),
    )
    with pytest.raises(ValueError, match="split"):
        split_questions([q], "holdout")


def test_evidence_ref_pages_are_sorted_deduped(tmp_path) -> None:
    row = _valid_row(
        evidence=[{"document_id": "nist_ai_600_1", "pages": [9, 4, 9]}]
    )
    path = _write_jsonl(tmp_path, [row])
    (q,) = load_eval_questions(path)
    assert q.evidence[0].pages == (4, 9)
    assert isinstance(q.evidence[0], EvidenceRef)
