"""Tests for QA scoring, relevance, abstention tuning, and benchmark plumbing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag_system.evaluation.answers import (
    abstention_metrics,
    normalize_for_matching,
    required_point_recall,
)
from rag_system.evaluation.benchmark import (
    Components,
    FrozenConfig,
    RetrievalSpec,
    aggregate_e2e_metrics,
    build_retriever,
    eval_retrieval,
    retrieval_grid,
    run_full_benchmark,
    run_pipeline_question,
    stage_latency_table,
    tune_abstention,
)
from rag_system.evaluation.dataset import EvalQuestion, EvidenceRef
from rag_system.evaluation.nli import claim_supported, sentence_windows
from rag_system.evaluation.relevance import (
    chunk_is_relevant,
    retrieval_scores,
)
from rag_system.generation.base import Claim, GenerationRequest, GenerationResponse
from rag_system.ingestion.schema import Chunk, Document, Page
from rag_system.ingestion.tokens import RegexTokenizer
from rag_system.observability.tracing import OTelTracer


class _FakeEncoder:
    model_id = "fake-encoder"
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        # Bag-of-chars hashing into 16 dims; deterministic.
        mat = np.zeros((len(texts), 16), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                mat[i, ord(ch) % 16] += 1.0
        return mat


class _FakeScorer:
    model_id = "fake-scorer"
    revision = "test"

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        # Score by token overlap with the query.
        q_tokens = set(query.lower().split())
        return [
            float(len(q_tokens & set(t.lower().split()))) for t in texts
        ]


class _FakeNLI:
    model_id = "fake-nli"
    revision = "test"

    def predict_labels(self, pairs: list[tuple[str, str]]) -> list[str]:
        return ["entailment"] * len(pairs)


class _ScriptedProvider:
    """Provider returning canned claims for the first evidence item."""

    def __init__(self, abstain_on_unanswerable: bool = True) -> None:
        self.abstain_on_unanswerable = abstain_on_unanswerable

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.evidence_list:
            return GenerationResponse(
                answer="I cannot answer.",
                status="abstained",
                abstain=True,
                abstain_reason="no evidence",
            )
        ev = request.evidence_list[0]
        if self.abstain_on_unanswerable and "unanswerable-marker" in request.query:
            return GenerationResponse(
                answer="I cannot answer.",
                status="abstained",
                abstain=True,
                abstain_reason="model abstained",
            )
        return GenerationResponse(
            answer="",
            status="answered",
            claims=(Claim(text=ev.text[:60], citation_labels=("c1",)),),
            prompt_tokens=10,
            completion_tokens=20,
        )


def _doc(doc_id: str = "doc_a", pages: int = 3) -> Document:
    return Document(
        document_id=doc_id,
        title=f"Doc {doc_id}",
        source_url="https://example.test/x",
        version="v1",
        pages=[
            Page(
                page_number=i + 1,
                text=(
                    f"alpha beta gamma page {i + 1} content about governance "
                    "and risk management frameworks"
                ),
            )
            for i in range(pages)
        ],
    )


def _question(
    qid: str,
    split: str,
    answerable: bool,
    question: str,
    doc_id: str = "doc_a",
    pages: tuple[int, ...] = (1,),
    points: tuple[tuple[str, ...], ...] = (("governance",),),
) -> EvalQuestion:
    return EvalQuestion(
        question_id=qid,
        split=split,
        category="test",
        difficulty="easy",
        answerable=answerable,
        question=question,
        evidence=(EvidenceRef(document_id=doc_id, pages=pages),)
        if answerable
        else (),
        required_points=points if answerable else (),
    )


def _comps(provider: _ScriptedProvider | None = None) -> Components:
    return Components(
        tokenizer=RegexTokenizer(),
        encoder=_FakeEncoder(),
        query_prefix="",
        pair_scorer=_FakeScorer(),
        nli_scorer=_FakeNLI(),
        provider=provider or _ScriptedProvider(),
        provider_name="scripted",
        provider_model_id="scripted-v1",
        dense_model_id="fake-encoder",
        dense_revision="test",
        reranker_model_id="fake-scorer",
        reranker_revision="test",
        nli_model_id="fake-nli",
        nli_revision="test",
    )


# --- answers.py ---


def test_normalize_for_matching() -> None:
    assert normalize_for_matching("Risk‑Based  Approaches") == (
        "riskbased approaches"
    )


def test_required_point_recall() -> None:
    recall, covered, total, all_cov = required_point_recall(
        "The GOVERN function governs risk.",
        [["govern function", "govern"], ["risk management"]],
    )
    assert recall == 0.5
    assert covered == 1 and total == 2 and all_cov is False


def test_abstention_metrics() -> None:
    rows = [(True, False), (True, True), (False, True), (False, False)]
    m = abstention_metrics(rows)
    assert m["abstain_precision"] == pytest.approx(0.5)
    assert m["abstain_recall"] == pytest.approx(0.5)
    assert m["abstain_f1"] == pytest.approx(0.5)


def test_abstention_metrics_hand_computed() -> None:
    # tp=3 unans-abstained, fp=1 ans-abstained, fn=1 unans-answered,
    # tn=2 ans-answered.
    rows = [
        (True, False),
        (True, False),
        (True, True),
        (False, True),
        (False, True),
        (False, True),
        (False, False),
    ]
    m = abstention_metrics(rows)
    assert m["abstain_precision"] == pytest.approx(0.75)
    assert m["abstain_recall"] == pytest.approx(0.75)
    assert m["abstain_f1"] == pytest.approx(0.75)
    assert m["false_answer_rate"] == pytest.approx(0.25)
    assert m["false_abstain_rate"] == pytest.approx(1 / 3)
    assert m["answerable_answer_rate"] == pytest.approx(2 / 3)


# --- relevance.py ---


def test_chunk_is_relevant_page_overlap() -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="doc_a",
        document_title="t",
        page_start=1,
        page_end=2,
        heading="",
        text="x",
        token_count=1,
    )
    q = _question("q1", "dev", True, "what?", pages=(2,))
    assert chunk_is_relevant(chunk, q)
    q2 = _question("q2", "dev", True, "what?", doc_id="other", pages=(1,))
    assert not chunk_is_relevant(chunk, q2)


def test_retrieval_scores() -> None:
    scores = retrieval_scores(["a", "b", "c"], {"b"})
    assert scores.hit_at_1 == 0.0
    assert scores.hit_at_3 == 1.0
    assert scores.mrr_at_10 == pytest.approx(0.5)
    assert scores.ndcg_at_10 > 0.0


def test_bootstrap_ci_deterministic() -> None:
    from rag_system.evaluation.relevance import bootstrap_ci

    a = bootstrap_ci([0.5] * 10)
    b = bootstrap_ci([0.5] * 10)
    assert a == b


# --- nli ---


def test_sentence_windows() -> None:
    text = " ".join(f"Sentence number {i}." for i in range(7))
    windows = sentence_windows(text, size=3, stride=2)
    # starts at sentence 0, 2, 4 -> three overlapping windows
    assert len(windows) == 3
    assert windows[0].startswith("Sentence number 0.")
    assert "Sentence number 2." in windows[1]
    short = sentence_windows("Only one sentence here.", size=3, stride=2)
    assert short == ["Only one sentence here."]


def test_claim_supported() -> None:
    assert claim_supported("entailment")
    assert not claim_supported("neutral")


# --- benchmark plumbing ---


def test_build_retriever_kinds() -> None:
    chunks = []
    for doc in [_doc()]:
        from rag_system.ingestion.chunk_tokens import chunk_document_tokens

        chunks.extend(
            chunk_document_tokens(doc, RegexTokenizer(), target_tokens=64)
        )
    assert chunks
    comps = _comps()
    for spec in retrieval_grid():
        retriever = build_retriever(spec, chunks, comps)
        out = retriever.retrieve("governance risk", top_k=3)
        assert len(out) <= 3


def test_eval_retrieval_and_tune_abstention() -> None:
    docs = [_doc()]
    chunks = []
    from rag_system.ingestion.chunk_tokens import chunk_document_tokens

    for doc in docs:
        chunks.extend(
            chunk_document_tokens(doc, RegexTokenizer(), target_tokens=64)
        )
    comps = _comps()
    retriever = build_retriever(
        RetrievalSpec(name="bm25", kind="bm25"), chunks, comps
    )
    dev_qs = [
        _question("d1", "dev", True, "governance risk management"),
        _question("d2", "dev", False, "unanswerable-marker question"),
    ]
    scores, records = eval_retrieval(retriever, dev_qs, chunks)
    assert len(scores) == 1  # only answerable scored
    assert records[0]["question_id"] == "d1"

    # Abstention tuning: threshold sweep over synthetic records.
    dev_records = [
        {"answerable": True, "top_score": 0.9, "status": "answered"},
        {"answerable": True, "top_score": 0.2, "status": "answered"},
        {"answerable": False, "top_score": 0.1, "status": "abstained"},
        {"answerable": False, "top_score": 0.8, "status": "answered"},
    ]
    threshold, objective, table = tune_abstention(dev_records)
    # All dev unanswerables abstain only if threshold > 0.8; that also
    # abstains the 0.9 answerable? No: 0.9 >= t required. Feasible t in
    # (0.8, 0.9] keeps answerable@0.9 answered... 0.2 answerable abstains.
    assert objective.startswith("max_")
    assert threshold is not None
    assert len(table) > 1


def test_run_pipeline_question_spans(tmp_path) -> None:
    docs = [_doc()]
    chunks = []
    from rag_system.ingestion.chunk_tokens import chunk_document_tokens

    for doc in docs:
        chunks.extend(
            chunk_document_tokens(doc, RegexTokenizer(), target_tokens=64)
        )
    chunk_map = {c.chunk_id: c for c in chunks}
    comps = _comps()
    retriever = build_retriever(
        RetrievalSpec(name="bm25", kind="bm25"), chunks, comps
    )
    cfg = FrozenConfig(
        chunk_tokens=64,
        overlap_ratio=0.15,
        retrieval=RetrievalSpec(name="bm25", kind="bm25"),
        rerank="none",
        abstain_threshold=None,
    )
    tracer = OTelTracer(
        service_name="t", span_exporters=[], enable_otlp=False
    )
    q = _question("q1", "test", True, "governance risk management")
    rec = run_pipeline_question(
        q, retriever, None, comps.provider, comps, cfg, chunk_map, tracer,
        abstain_threshold=None, record_nli=True,
    )
    assert rec["status"] == "answered"
    assert rec["nli_total_claims"] >= 1
    assert rec["required_point_recall"] == pytest.approx(1.0)
    tracer.shutdown()


def test_required_point_found_only_in_claim_text(tmp_path) -> None:
    """A required point present only in a claim's text still counts."""

    class _ClaimOnlyProvider:
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                answer="",  # point absent from the answer field itself
                status="answered",
                claims=(
                    Claim(
                        text="The zephyr protocol governs cold storage.",
                        citation_labels=("c1",),
                    ),
                ),
                prompt_tokens=5,
                completion_tokens=5,
            )

    docs = [_doc()]
    chunks = []
    from rag_system.ingestion.chunk_tokens import chunk_document_tokens

    for doc in docs:
        chunks.extend(
            chunk_document_tokens(doc, RegexTokenizer(), target_tokens=64)
        )
    chunk_map = {c.chunk_id: c for c in chunks}
    comps = _comps(_ClaimOnlyProvider())
    retriever = build_retriever(
        RetrievalSpec(name="bm25", kind="bm25"), chunks, comps
    )
    cfg = FrozenConfig(
        chunk_tokens=64,
        overlap_ratio=0.15,
        retrieval=RetrievalSpec(name="bm25", kind="bm25"),
        rerank="none",
        abstain_threshold=None,
    )
    tracer = OTelTracer(
        service_name="t", span_exporters=[], enable_otlp=False
    )
    q = _question(
        "q_claim", "test", True, "cold storage governance",
        points=(("zephyr protocol",),),
    )
    rec = run_pipeline_question(
        q, retriever, None, comps.provider, comps, cfg, chunk_map, tracer,
        abstain_threshold=None, record_nli=False,
    )
    assert rec["status"] == "answered"
    assert rec["required_point_recall"] == pytest.approx(1.0)
    assert rec["all_points_covered"] is True
    tracer.shutdown()


def test_aggregate_e2e_metrics_and_stage_table(tmp_path) -> None:
    records = [
        {
            "question_id": "a",
            "answerable": True,
            "status": "answered",
            "required_point_recall": 1.0,
            "all_points_covered": True,
            "citation_label_validity": 1.0,
            "citation_page_precision": 1.0,
            "citation_hit": 1.0,
            "nli_supported_claims": 1,
            "nli_total_claims": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
        {
            "question_id": "b",
            "answerable": False,
            "status": "abstained",
            "required_point_recall": 0.0,
            "all_points_covered": False,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    ]
    m = aggregate_e2e_metrics(records)
    assert m["abstain_precision"] == pytest.approx(1.0)
    assert m["abstain_recall"] == pytest.approx(1.0)
    assert m["nli_claim_support_rate"] == pytest.approx(1.0)

    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps({"name": "generation", "duration_ms": 100.0}) + "\n"
        + json.dumps({"name": "generation", "duration_ms": 300.0}) + "\n"
    )
    table = stage_latency_table(traces)
    assert table["generation"]["count"] == 2
    assert table["generation"]["p50_ms"] == pytest.approx(200.0)


def test_run_full_benchmark_writes_artifacts(tmp_path) -> None:
    docs = [_doc(), _doc("doc_b", pages=2)]
    questions = [
        _question("d1", "dev", True, "governance risk management"),
        _question("d2", "dev", False, "unanswerable-marker question"),
        _question("t1", "test", True, "governance risk management"),
        _question("t2", "test", False, "unanswerable-marker question"),
    ]
    out = tmp_path / "out"
    result = run_full_benchmark(
        docs, {"documents": []}, questions, _comps(), out
    )
    assert (out / "metrics.json").is_file()
    assert (out / "manifest.json").is_file()
    assert (out / "report.md").is_file()
    assert (out / "traces.jsonl").is_file()
    assert (out / "test_questions.jsonl").is_file()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["frozen_config"]["chunk_tokens"] in (256, 512, 768)
    e2e = result["metrics"]["e2e_test"]
    assert e2e["num_questions"] == 2
