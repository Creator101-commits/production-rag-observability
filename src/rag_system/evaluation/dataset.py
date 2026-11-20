"""Loader and validator for the frozen NIST QA evaluation set (JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SPLITS = ("dev", "test")


@dataclass(frozen=True)
class EvidenceRef:
    """Judged evidence location: a document plus 1-based physical PDF pages."""

    document_id: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class EvalQuestion:
    """One frozen evaluation question with answerability and evidence labels."""

    question_id: str
    split: str
    category: str
    difficulty: str
    answerable: bool
    question: str
    evidence: tuple[EvidenceRef, ...]
    # Each inner tuple holds acceptable alternative phrases for one required point.
    required_points: tuple[tuple[str, ...], ...]

    @property
    def expected_doc_ids(self) -> tuple[str, ...]:
        """Document IDs cited by the evidence labels (deduplicated, ordered)."""
        return tuple(dict.fromkeys(ref.document_id for ref in self.evidence))


def default_eval_path(start: Path | None = None) -> Path:
    """Locate ``data/eval/nist_qa_v1.jsonl`` by walking up from ``start``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        path = candidate / "data" / "eval" / "nist_qa_v1.jsonl"
        if path.is_file():
            return path
    raise FileNotFoundError(
        "data/eval/nist_qa_v1.jsonl not found above " + str(current)
    )


def _parse_evidence(raw: Any, question_id: str) -> tuple[EvidenceRef, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{question_id}: 'evidence' must be a list")
    refs: list[EvidenceRef] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{question_id}: evidence entries must be objects")
        doc_id = entry.get("document_id")
        pages = entry.get("pages")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{question_id}: evidence document_id must be a string")
        if (
            not isinstance(pages, list)
            or not pages
            or not all(isinstance(p, int) and p >= 1 for p in pages)
        ):
            raise ValueError(
                f"{question_id}: evidence pages must be a non-empty list of "
                "positive integers"
            )
        refs.append(EvidenceRef(document_id=doc_id, pages=tuple(sorted(set(pages)))))
    return tuple(refs)


def _parse_required_points(raw: Any, question_id: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{question_id}: 'required_points' must be a list")
    points: list[tuple[str, ...]] = []
    for point in raw:
        if (
            not isinstance(point, list)
            or not point
            or not all(isinstance(alt, str) and alt.strip() for alt in point)
        ):
            raise ValueError(
                f"{question_id}: each required point must be a non-empty list "
                "of non-empty alternative strings"
            )
        points.append(tuple(point))
    return tuple(points)


def _parse_question(raw: Any, line_no: int) -> EvalQuestion:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_no}: question must be a JSON object")
    qid = raw.get("question_id")
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"line {line_no}: question_id must be a non-empty string")
    split = raw.get("split")
    if split not in VALID_SPLITS:
        raise ValueError(f"{qid}: split must be one of {VALID_SPLITS}")
    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError(f"{qid}: answerable must be a boolean")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{qid}: question must be a non-empty string")
    for field_name in ("category", "difficulty"):
        value = raw.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{qid}: {field_name} must be a non-empty string")
    evidence = _parse_evidence(raw.get("evidence"), qid)
    points = _parse_required_points(raw.get("required_points"), qid)
    return EvalQuestion(
        question_id=qid,
        split=split,
        category=str(raw["category"]),
        difficulty=str(raw["difficulty"]),
        answerable=answerable,
        question=question,
        evidence=evidence,
        required_points=points,
    )


def load_eval_questions(path: str | Path | None = None) -> list[EvalQuestion]:
    """Load and validate the frozen JSONL evaluation set."""
    eval_path = Path(path) if path is not None else default_eval_path()
    questions: list[EvalQuestion] = []
    with eval_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{eval_path}:{line_no}: invalid JSON: {exc}") from exc
            questions.append(_parse_question(raw, line_no))
    validate_eval_questions(questions)
    return questions


def load_evaluation_dataset() -> list[EvalQuestion]:
    """Backwards-compatible alias for ``load_eval_questions``."""
    return load_eval_questions()


def validate_eval_questions(questions: list[EvalQuestion]) -> None:
    """Check cross-question invariants of the frozen evaluation set."""
    if not questions:
        raise ValueError("evaluation set is empty")
    seen: set[str] = set()
    for q in questions:
        if q.question_id in seen:
            raise ValueError(f"duplicate question_id: {q.question_id}")
        seen.add(q.question_id)
        if q.answerable:
            if not q.evidence:
                raise ValueError(f"{q.question_id}: answerable question lacks evidence")
            if not q.required_points:
                raise ValueError(
                    f"{q.question_id}: answerable question lacks required_points"
                )
        else:
            if q.evidence:
                raise ValueError(
                    f"{q.question_id}: unanswerable question must not cite evidence"
                )
            if q.required_points:
                raise ValueError(
                    f"{q.question_id}: unanswerable question must not list "
                    "required_points"
                )


def split_questions(
    questions: list[EvalQuestion], split: str
) -> list[EvalQuestion]:
    """Return questions for one split, preserving file order."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}")
    return [q for q in questions if q.split == split]
