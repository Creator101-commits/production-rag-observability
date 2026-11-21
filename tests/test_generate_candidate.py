"""Subprocess isolation test for the stage-B candidate worker."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rag_system.generation.base import GenerationResponse
from rag_system.observability.tracing import OTelTracer


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    contexts = {
        "k_ctxs": [3],
        "temperature": 0.0,
        "max_context_tokens": 2048,
        "chunk_meta": {
            "c1": {
                "document_id": "doc_a",
                "page_start": 1,
                "page_end": 2,
                "text": "Governance of AI risk is documented here.",
            }
        },
        "questions": [
            {
                "question_id": "q1",
                "split": "dev",
                "category": "cat",
                "difficulty": "easy",
                "answerable": True,
                "question": "What does the document say about governance?",
                "required_points": [["governance"]],
                "top_score": 0.9,
                "retrieved_chunk_ids": ["c1"],
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "document_id": "doc_a",
                        "document_title": "Doc A",
                        "page_start": 1,
                        "page_end": 2,
                        "text": "Governance of AI risk is documented here.",
                        "score": 0.9,
                        "rank": 1,
                        "retriever_name": "hybrid",
                    }
                ],
            },
            {
                "question_id": "t1",
                "split": "test",
                "category": "cat",
                "difficulty": "easy",
                "answerable": False,
                "question": "Unanswerable probe question?",
                "required_points": [],
                "top_score": 0.1,
                "retrieved_chunk_ids": ["c1"],
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "document_id": "doc_a",
                        "document_title": "Doc A",
                        "page_start": 1,
                        "page_end": 2,
                        "text": "Governance of AI risk is documented here.",
                        "score": 0.1,
                        "rank": 1,
                        "retriever_name": "hybrid",
                    }
                ],
            },
        ],
    }
    spec = {"name": "fake-candidate", "kind": "fake"}
    ctx_path = tmp_path / "contexts.json"
    spec_path = tmp_path / "spec.json"
    ctx_path.write_text(json.dumps(contexts), encoding="utf-8")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return ctx_path, spec_path, tmp_path / "out"


def test_generate_candidate_subprocess_writes_records(tmp_path: Path) -> None:
    ctx_path, spec_path, out_dir = _write_inputs(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_system.evaluation.generate_candidate",
            "--contexts",
            str(ctx_path),
            "--spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fake-candidate" in proc.stdout

    dev_file = out_dir / "fake-candidate__k3__dev.jsonl"
    test_file = out_dir / "fake-candidate__k3__test.jsonl"
    assert dev_file.is_file() and test_file.is_file()
    assert (out_dir / "traces_fake-candidate.jsonl").is_file()

    dev_rec = json.loads(dev_file.read_text().strip())
    for key in (
        "question_id",
        "split",
        "answerable",
        "top_score",
        "status",
        "abstain_source",
        "answer",
        "raw_output",
        "claims",
        "citations",
        "prompt_tokens",
        "completion_tokens",
        "parse_failure",
        "latency_ms",
        "citation_label_validity",
    ):
        assert key in dev_rec, key
    assert dev_rec["question_id"] == "q1"
    assert dev_rec["split"] == "dev"

    test_rec = json.loads(test_file.read_text().strip())
    assert test_rec["question_id"] == "t1"
    assert test_rec["answerable"] is False


def test_generate_candidate_k_ctx_filter_writes_only_that_size(
    tmp_path: Path,
) -> None:
    ctx_path, spec_path, out_dir = _write_inputs(tmp_path)
    contexts = json.loads(ctx_path.read_text())
    contexts["k_ctxs"] = [3, 5]
    ctx_path.write_text(json.dumps(contexts), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "rag_system.evaluation.generate_candidate",
            "--contexts",
            str(ctx_path),
            "--spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
            "--k-ctx",
            "5",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (out_dir / "fake-candidate__k5__dev.jsonl").is_file()
    assert (out_dir / "fake-candidate__k5__test.jsonl").is_file()
    assert not (out_dir / "fake-candidate__k3__dev.jsonl").exists()


class _ParseFailingProvider:
    """Always returns a parse-failure abstention with fixed token counts."""

    def __init__(self, completion_tokens: int) -> None:
        self.completion_tokens = completion_tokens

    def generate(self, request: object) -> GenerationResponse:
        return GenerationResponse(
            answer="I cannot answer.",
            status="abstained",
            abstain=True,
            abstain_reason="generation parse failure",
            parse_failure=True,
            prompt_tokens=10,
            completion_tokens=self.completion_tokens,
        )


def _one_question() -> dict[str, object]:
    return {
        "question_id": "q1",
        "split": "test",
        "category": "cat",
        "difficulty": "easy",
        "answerable": True,
        "question": "What does the document say?",
        "top_score": 0.9,
        "evidence": [
            {
                "chunk_id": "c1",
                "document_id": "doc_a",
                "document_title": "Doc A",
                "page_start": 1,
                "page_end": 2,
                "text": "Governance of AI risk is documented here.",
                "score": 0.9,
                "rank": 1,
                "retriever_name": "hybrid",
            }
        ],
    }


def test_generate_records_truncation_accounting() -> None:
    """Cap-hit parse failures are 'truncated', others stay 'parse_failure'."""
    from rag_system.evaluation.generate_candidate import _generate_records

    tracer = OTelTracer(
        service_name="t", span_exporters=[], enable_otlp=False
    )
    try:
        at_cap = list(
            _generate_records(
                _ParseFailingProvider(20),
                [_one_question()],
                3,
                tracer,
                max_new_tokens=20,
            )
        )
        assert at_cap[0]["hit_token_cap"] is True
        assert at_cap[0]["abstain_source"] == "truncated"
        assert at_cap[0]["parse_failure"] is True

        below_cap = list(
            _generate_records(
                _ParseFailingProvider(19),
                [_one_question()],
                3,
                tracer,
                max_new_tokens=20,
            )
        )
        assert below_cap[0]["hit_token_cap"] is False
        assert below_cap[0]["abstain_source"] == "parse_failure"

        no_cap = list(
            _generate_records(
                _ParseFailingProvider(10_000),
                [_one_question()],
                3,
                tracer,
            )
        )
        assert no_cap[0]["hit_token_cap"] is False
        assert no_cap[0]["abstain_source"] == "parse_failure"
    finally:
        tracer.shutdown()
