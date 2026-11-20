"""Request tracing with real OpenTelemetry SDK spans.

Spans cover request, retrieval, fusion, rerank, threshold, generation,
citation validation, and serialization. Attributes are redacted by
default: query SHA-256 and length, chunk IDs, scores, token counts,
model IDs, and outcome flags. Raw query or document text is never set.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from rag_system.observability.redaction import redact_sensitive_text

# Attribute keys that could carry raw content are dropped, not redacted:
# redaction cannot prove a free-text field is safe.
_FORBIDDEN_ATTRIBUTE_KEYS = frozenset(
    {
        "query",
        "question",
        "text",
        "answer",
        "prompt",
        "response",
        "document",
        "content",
        "claim",
        "evidence_text",
        "completion",
    }
)

_PRIMITIVE_TYPES = (str, bool, int, float)


def fingerprint_query(text: str) -> tuple[str, int]:
    """Return (sha256 hex, char length) for a query; never the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)


def _coerce_attribute_value(value: Any) -> Any:
    if isinstance(value, _PRIMITIVE_TYPES):
        return redact_sensitive_text(value) if isinstance(value, str) else value
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_attribute_value(v) for v in value]
        if all(isinstance(v, _PRIMITIVE_TYPES) for v in coerced):
            return coerced
    return None


def sanitize_span_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop forbidden keys and coerce values to OTel-safe primitives."""
    clean: dict[str, Any] = {}
    for key, value in attributes.items():
        if key.lower() in _FORBIDDEN_ATTRIBUTE_KEYS:
            continue
        coerced = _coerce_attribute_value(value)
        if coerced is not None:
            clean[key] = coerced
    return clean


class SafeSpanExporter(SpanExporter):
    """Wraps a SpanExporter so export failures never reach the service."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner
        self.failures = 0
        self.exported = 0

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        try:
            result = self._inner.export(spans)
        except Exception:
            self.failures += 1
            return SpanExportResult.FAILURE
        if result == SpanExportResult.SUCCESS:
            self.exported += len(spans)
        return result

    def shutdown(self) -> None:
        try:
            self._inner.shutdown()
        except Exception:
            self.failures += 1

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return bool(self._inner.force_flush(timeout_millis))
        except Exception:
            self.failures += 1
            return False


class JsonlSpanExporter(SpanExporter):
    """Writes finished spans as JSON lines for offline analysis."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self.failures = 0

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        try:
            for span in spans:
                record = {
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_span_id": (
                        format(span.parent.span_id, "016x")
                        if span.parent is not None
                        else None
                    ),
                    "name": span.name,
                    "start_unix_nano": int(span.start_time or 0),
                    "end_unix_nano": int(span.end_time or 0),
                    "duration_ms": (
                        (span.end_time - span.start_time) / 1e6
                        if span.end_time and span.start_time
                        else 0.0
                    ),
                    "attributes": dict(span.attributes or {}),
                    "status": span.status.status_code.name
                    if span.status is not None
                    else "UNSET",
                }
                self._handle.write(json.dumps(record, default=str) + "\n")
            self._handle.flush()
            return SpanExportResult.SUCCESS
        except Exception:
            self.failures += 1
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        try:
            self._handle.close()
        except Exception:
            self.failures += 1

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            self._handle.flush()
            return True
        except Exception:
            self.failures += 1
            return False


def _make_otlp_exporter() -> SpanExporter | None:
    """Build an OTLP HTTP exporter only when configured and installed."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
    except ImportError:
        return None
    try:
        exporter: SpanExporter = OTLPSpanExporter(endpoint=endpoint)
        return exporter
    except Exception:
        return None


class OTelTracer:
    """OpenTelemetry tracer with in-memory, JSONL, and optional OTLP export."""

    def __init__(
        self,
        service_name: str = "rag_system",
        span_exporters: Sequence[SpanExporter] | None = None,
        enable_otlp: bool = True,
    ) -> None:
        resource = Resource.create({"service.name": service_name})
        self.provider = TracerProvider(resource=resource)
        self._safe_exporters: list[SafeSpanExporter] = []

        exporters: list[SpanExporter] = list(span_exporters or [])
        if enable_otlp:
            otlp = _make_otlp_exporter()
            if otlp is not None:
                exporters.append(otlp)

        for exporter in exporters:
            safe = SafeSpanExporter(exporter)
            self._safe_exporters.append(safe)
            self.provider.add_span_processor(SimpleSpanProcessor(safe))

        trace.set_tracer_provider(self.provider)
        self._tracer = self.provider.get_tracer("rag_system")

    @property
    def export_failures(self) -> int:
        return sum(e.failures for e in self._safe_exporters)

    @property
    def spans_exported(self) -> int:
        return sum(e.exported for e in self._safe_exporters)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Open a span, set sanitized attributes, mark OK/ERROR on exit."""
        clean = sanitize_span_attributes(attributes or {})
        with self._tracer.start_as_current_span(name) as otel_span:
            for key, value in clean.items():
                otel_span.set_attribute(key, value)
            try:
                yield otel_span
            except Exception as exc:
                otel_span.set_status(
                    Status(StatusCode.ERROR, type(exc).__name__)
                )
                raise
            else:
                otel_span.set_status(Status(StatusCode.OK))

    def force_flush(self, timeout_millis: int = 30000) -> None:
        with suppress(Exception):
            self.provider.force_flush(timeout_millis)

    def shutdown(self) -> None:
        with suppress(Exception):
            self.provider.shutdown()
