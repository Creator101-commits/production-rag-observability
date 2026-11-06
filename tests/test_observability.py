"""Tests for telemetry tracing, timing, redaction, and token cost calculation."""

from __future__ import annotations

import time

from rag_system.observability.redaction import redact_sensitive_text
from rag_system.observability.tracing import Tracer


def test_redact_sensitive_text() -> None:
    text = (
        "User contact is user@example.com, phone is 555-123-4567, "
        "SSN is 123-45-6789, key is sk-12345678901234567890abcdef."
    )
    redacted = redact_sensitive_text(text)
    assert "user@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "555-123-4567" not in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "sk-12345678901234567890abcdef" not in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_tracer_span_and_cost() -> None:
    tracer = Tracer(input_token_cost_per_1k=0.002, output_token_cost_per_1k=0.004)
    trace = tracer.create_trace(request_id="test_req_001")

    with tracer.span(trace, "test_stage", {"query": "test query user@test.com"}):
        time.sleep(0.01)

    assert len(trace.spans) == 1
    assert trace.spans[0].name == "test_stage"
    assert trace.spans[0].duration_ms > 0.0
    assert trace.spans[0].attributes["query"] == "test query [REDACTED_EMAIL]"

    # Token and cost accounting
    tracer.record_token_usage(trace, prompt_tokens=1000, completion_tokens=500)
    assert trace.prompt_tokens == 1000
    assert trace.completion_tokens == 500
    assert trace.estimated_cost_usd == 0.002 + 0.002  # 0.002 * 1 + 0.004 * 0.5 = 0.004
