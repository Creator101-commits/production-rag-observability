"""Tests for evaluation metrics and benchmark runner."""

from __future__ import annotations

import pytest

from rag_system.evaluation.dataset import EvalQuestion, load_evaluation_dataset
from rag_system.evaluation.metrics import (
    compute_abstention_metrics,
    compute_latency_percentiles,
)
from rag_system.evaluation.runner import BenchmarkRunner


def test_compute_abstention_metrics_perfect() -> None:
    y_true_answerable = [True, True, False, False]
    y_pred_answered = [True, True, False, False]

    prec, rec, false_ans, false_abs = compute_abstention_metrics(
        y_true_answerable, y_pred_answered
    )

    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert false_ans == pytest.approx(0.0)
    assert false_abs == pytest.approx(0.0)


def test_compute_abstention_metrics_errors() -> None:
    y_true_answerable = [True, False]
    y_pred_answered = [False, True]  # false abstention and false answer

    prec, rec, false_ans, false_abs = compute_abstention_metrics(
        y_true_answerable, y_pred_answered
    )

    assert prec == pytest.approx(0.0)
    assert rec == pytest.approx(0.0)
    assert false_ans == pytest.approx(1.0)
    assert false_abs == pytest.approx(1.0)


def test_compute_latency_percentiles() -> None:
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    mean, p95 = compute_latency_percentiles(latencies)
    assert mean == pytest.approx(30.0)
    assert p95 > 40.0

    empty_mean, empty_p95 = compute_latency_percentiles([])
    assert empty_mean == 0.0
    assert empty_p95 == 0.0


def test_benchmark_runner() -> None:
    questions = [
        EvalQuestion(
            question_id="t1",
            question="What is the GOVERN function in NIST AI RMF?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_1_0"],
            category="direct_lookup",
            split="test",
        ),
        EvalQuestion(
            question_id="t2",
            question="What is the price of gold in Mars?",
            answerable=False,
            expected_doc_ids=[],
            category="unanswerable",
            split="test",
        ),
    ]

    runner = BenchmarkRunner()
    metrics = runner.run_evaluation(questions)

    assert 0.0 <= metrics.retrieval_recall_at_5 <= 1.0
    assert 0.0 <= metrics.citation_validity_rate <= 1.0
    assert 0.0 <= metrics.abstention_precision <= 1.0
    assert metrics.mean_latency_ms >= 0.0


def test_load_evaluation_dataset() -> None:
    questions = load_evaluation_dataset()
    assert len(questions) > 5
    splits = {q.split for q in questions}
    assert "test" in splits
    assert "dev" in splits
