"""Request tracing, span recording, and cost estimation."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from rag_system.observability.redaction import redact_sensitive_text


@dataclass
class Span:
    """A timed operation within a request lifecycle."""

    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


@dataclass
class RequestTrace:
    """Full lifecycle telemetry for a single query invocation."""

    request_id: str
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "success"
    error_message: str | None = None

    @property
    def total_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)


class Tracer:
    """Lightweight in-memory and local-safe tracer without an OpenTelemetry exporter."""

    def __init__(
        self,
        input_token_cost_per_1k: float = 0.0015,
        output_token_cost_per_1k: float = 0.0020,
    ) -> None:
        self.input_cost_per_1k = input_token_cost_per_1k
        self.output_cost_per_1k = output_token_cost_per_1k
        self.completed_traces: list[RequestTrace] = []

    def create_trace(self, request_id: str | None = None) -> RequestTrace:
        """Initialize a new request trace."""
        req_id = request_id or str(uuid.uuid4())
        tr_id = str(uuid.uuid4())
        return RequestTrace(request_id=req_id, trace_id=tr_id)

    @contextmanager
    def span(
        self,
        trace: RequestTrace,
        name: str,
        attributes: dict[str, Any] | None = None,
        redact: bool = True,
    ) -> Iterator[Span]:
        """Context manager to measure and record a pipeline stage span."""
        attrs = attributes or {}
        if redact:
            clean_attrs: dict[str, Any] = {}
            for k, v in attrs.items():
                if isinstance(v, str):
                    clean_attrs[k] = redact_sensitive_text(v)
                else:
                    clean_attrs[k] = v
            attrs = clean_attrs

        span_obj = Span(name=name, start_time=time.perf_counter(), attributes=attrs)
        try:
            yield span_obj
        finally:
            span_obj.finish()
            trace.spans.append(span_obj)

    def record_token_usage(
        self,
        trace: RequestTrace,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Update token counts and calculate estimated dollar cost."""
        trace.prompt_tokens += prompt_tokens
        trace.completion_tokens += completion_tokens

        prompt_cost = (trace.prompt_tokens / 1000.0) * self.input_cost_per_1k
        completion_cost = (trace.completion_tokens / 1000.0) * self.output_cost_per_1k
        trace.estimated_cost_usd = float(prompt_cost + completion_cost)

    def finish_trace(self, trace: RequestTrace) -> None:
        """Register completed trace."""
        self.completed_traces.append(trace)
        del self.completed_traces[:-1000]
