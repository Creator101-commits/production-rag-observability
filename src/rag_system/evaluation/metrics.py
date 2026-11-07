"""Evaluation metrics for retrieval, citations, abstention, and latency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OverallRAGMetrics:
    """Consolidated benchmark metrics across all evaluated queries."""

    # Retrieval
    retrieval_recall_at_5: float
    retrieval_mrr: float

    # Citation
    citation_validity_rate: float
    citation_precision: float

    # Abstention
    abstention_precision: float
    abstention_recall: float
    false_answer_rate: float
    false_abstention_rate: float

    # Latency and cost
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float


def compute_abstention_metrics(
    y_true_answerable: list[bool],
    y_pred_answered: list[bool],
) -> tuple[float, float, float, float]:
    """Compute precision, recall, false answers, and false abstentions.

    Here 'abstain' is positive class when question is unanswerable.
    """
    if len(y_true_answerable) != len(y_pred_answered):
        raise ValueError("Answerability labels and predictions must be aligned.")
    total = len(y_true_answerable)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    # True unanswerable questions
    n_unanswerable = sum(1 for a in y_true_answerable if not a)
    n_answerable = total - n_unanswerable

    # Predicted abstentions (not answered)
    pred_abstained = [not ans for ans in y_pred_answered]

    # True positive abstentions: unanswerable AND abstained
    tp_abstain = sum(
        1
        for true_ans, pred_abs in zip(y_true_answerable, pred_abstained, strict=False)
        if not true_ans and pred_abs
    )
    # False positive abstention: answerable BUT abstained
    fp_abstain = sum(
        1
        for true_ans, pred_abs in zip(y_true_answerable, pred_abstained, strict=False)
        if true_ans and pred_abs
    )
    # False negative abstention: unanswerable BUT answered (hallucination / false answer)
    fn_abstain = sum(
        1
        for true_ans, pred_ans in zip(y_true_answerable, y_pred_answered, strict=False)
        if not true_ans and pred_ans
    )

    abstain_prec = tp_abstain / (tp_abstain + fp_abstain) if (tp_abstain + fp_abstain) > 0 else 1.0
    abstain_rec = tp_abstain / n_unanswerable if n_unanswerable > 0 else 1.0

    false_answer_rate = fn_abstain / n_unanswerable if n_unanswerable > 0 else 0.0
    false_abstention_rate = fp_abstain / n_answerable if n_answerable > 0 else 0.0

    return abstain_prec, abstain_rec, false_answer_rate, false_abstention_rate


def compute_latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """Compute mean and 95th percentile latency in milliseconds."""
    if not latencies_ms:
        return 0.0, 0.0
    arr = np.array(latencies_ms)
    return float(np.mean(arr)), float(np.percentile(arr, 95))
