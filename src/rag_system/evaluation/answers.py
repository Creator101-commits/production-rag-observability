"""Answer-quality scoring for the QA benchmark.

Required-point matching normalizes both sides with NFKC, lowercase,
whitespace collapse, and hyphen-insensitivity before substring checks.
"""

from __future__ import annotations

import re
import unicodedata

_HYPHENS = re.compile("[-‐‑‒–\u2014―]")
_WS = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """NFKC, lowercase, strip hyphens, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _HYPHENS.sub("", text)
    return _WS.sub(" ", text).strip()


def required_point_recall(
    answer: str, required_points: list[list[str]]
) -> tuple[float, int, int, bool]:
    """Return (recall, covered, total, all_covered) for one answer."""
    if not required_points:
        return 1.0, 0, 0, True
    norm_answer = normalize_for_matching(answer)
    covered = 0
    for point in required_points:
        if any(
            normalize_for_matching(alt) in norm_answer for alt in point
        ):
            covered += 1
    total = len(required_points)
    return covered / total, covered, total, covered == total


def abstention_counts(
    rows: list[tuple[bool, bool]],
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) with positive = unanswerable class.

    tp: unanswerable + abstained; fp: answerable + abstained;
    fn: unanswerable + answered; tn: answerable + answered.
    """
    tp = fp = fn = tn = 0
    for answerable, abstained in rows:
        if answerable and abstained:
            fp += 1
        elif answerable:
            tn += 1
        elif abstained:
            tp += 1
        else:
            fn += 1
    return tp, fp, fn, tn


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts; zero-safe."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def abstention_metrics(
    rows: list[tuple[bool, bool]],
) -> dict[str, float]:
    """Abstention metrics; rows are (answerable, abstained) pairs."""
    tp, fp, fn, tn = abstention_counts(rows)
    precision, recall, f1 = prf1(tp, fp, fn)
    unanswerable_total = tp + fn
    answerable_total = fp + tn
    return {
        "abstain_precision": precision,
        "abstain_recall": recall,
        "abstain_f1": f1,
        # Unanswerable questions that were answered anyway.
        "false_answer_rate": fn / unanswerable_total
        if unanswerable_total
        else 0.0,
        # Answerable questions that were abstained on.
        "false_abstain_rate": fp / answerable_total
        if answerable_total
        else 0.0,
        "answerable_answer_rate": tn / answerable_total
        if answerable_total
        else 0.0,
    }
