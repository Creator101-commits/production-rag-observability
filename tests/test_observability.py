"""Tests for OpenTelemetry spans, redaction, exporters, and resilience."""

from __future__ import annotations

import json

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rag_system.observability.redaction import redact_sensitive_text
from rag_system.observability.tracing import (
    JsonlSpanExporter,
    OTelTracer,
    fingerprint_query,
    sanitize_span_attributes,
)


def test_redact_sensitive_text() -> None:
    text = (
        "User contact is user@example.com, phone is 555-123-4567, "
        "SSN is 123-45-6789, key is sk-12345678901234567890abcdef."
    )
    redacted = redact_sensitive_text(text)
    assert "user@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "555-123-4567" not in redacted
    assert "123-45-6789" not in redacted
    assert "sk-12345678901234567890abcdef" not in redacted


def test_fingerprint_query_never_returns_text() -> None:
    sha, length = fingerprint_query("What is GOVERN?")
    assert len(sha) == 64
    assert length == len("What is GOVERN?")
    assert "GOVERN" not in sha


def test_sanitize_span_attributes_drops_raw_content() -> None:
    clean = sanitize_span_attributes(
        {
            "query": "secret question text",
            "query_sha256": "ab" * 32,
            "query_length": 18,
            "chunk_ids": ["c1", "c2"],
            "top_score": 0.42,
            "evidence_count": 5,
            "rerank_enabled": True,
            "nested": {"bad": "object"},
        }
    )
    assert "query" not in clean
    assert clean["query_sha256"] == "ab" * 32
    assert clean["query_length"] == 18
    assert clean["chunk_ids"] == ["c1", "c2"]
    assert clean["top_score"] == 0.42
    assert clean["rerank_enabled"] is True
    assert "nested" not in clean


def test_otel_tracer_exports_spans_in_memory() -> None:
    memory = InMemorySpanExporter()
    tracer = OTelTracer(
        service_name="test-service", span_exporters=[memory], enable_otlp=False
    )

    with tracer.span("request", {"request_id": "r1", "query_length": 12}):
        with tracer.span("retrieval", {"candidate_count": 3}):
            pass
        with tracer.span("generation", {"status": "answered"}):
            pass

    tracer.force_flush()
    spans = memory.get_finished_spans()
    names = [s.name for s in spans]
    assert "request" in names
    assert "retrieval" in names
    assert "generation" in names

    request_span = next(s for s in spans if s.name == "request")
    assert request_span.attributes["request_id"] == "r1"
    assert request_span.attributes["query_length"] == 12
    # Child spans share the request trace id.
    retrieval_span = next(s for s in spans if s.name == "retrieval")
    assert retrieval_span.context.trace_id == request_span.context.trace_id
    tracer.shutdown()


def test_otel_tracer_marks_error_spans() -> None:
    memory = InMemorySpanExporter()
    tracer = OTelTracer(
        service_name="test-service", span_exporters=[memory], enable_otlp=False
    )
    try:
        with tracer.span("flaky_stage"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    tracer.force_flush()
    span = memory.get_finished_spans()[0]
    assert span.status is not None
    assert span.status.status_code.name == "ERROR"
    tracer.shutdown()


def test_jsonl_span_exporter_writes_records(tmp_path) -> None:
    out = tmp_path / "traces.jsonl"
    exporter = JsonlSpanExporter(out)
    tracer = OTelTracer(
        service_name="test-service",
        span_exporters=[exporter],
        enable_otlp=False,
    )
    with tracer.span("request", {"query_sha256": "ff" * 32}):  # noqa: SIM117
        with tracer.span("reranking", {"rerank_ms": 12.5}):
            pass
    tracer.force_flush()
    tracer.shutdown()

    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert {r["name"] for r in lines} == {"request", "reranking"}
    request = next(r for r in lines if r["name"] == "request")
    assert request["attributes"]["query_sha256"] == "ff" * 32
    assert request["duration_ms"] >= 0.0
    rerank = next(r for r in lines if r["name"] == "reranking")
    assert rerank["attributes"]["rerank_ms"] == 12.5
    assert rerank["parent_span_id"] == request["span_id"]


class _FailingExporter:
    """Exporter that always raises; the service must keep working."""

    def __init__(self) -> None:
        self.calls = 0

    def export(self, spans):  # noqa: ANN001, ANN201
        self.calls += 1
        raise RuntimeError("export backend down")

    def shutdown(self) -> None:
        raise RuntimeError("shutdown failed")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        raise RuntimeError("flush failed")


def test_failing_exporter_does_not_break_spans() -> None:
    failing = _FailingExporter()
    tracer = OTelTracer(
        service_name="test-service",
        span_exporters=[failing],  # type: ignore[list-item]
        enable_otlp=False,
    )
    with tracer.span("request"), tracer.span("generation"):
        pass
    tracer.force_flush()
    tracer.shutdown()
    assert tracer.export_failures >= 2
    # The spans themselves completed without raising.
