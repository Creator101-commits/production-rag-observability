"""Pydantic schemas for the FastAPI service endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str = "ready"
    chunks_indexed: int


class QueryRequest(BaseModel):
    """Query input request payload."""

    query: str = Field(..., min_length=1, description="Question about NIST AI RMF")
    request_id: str | None = Field(default=None, description="Optional client request ID")


class CitationItem(BaseModel):
    """Verified citation item with document provenance."""

    citation_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    text_snippet: str


class QueryResponse(BaseModel):
    """Full question answering response with provenance and observability metadata."""

    request_id: str
    trace_id: str
    status: str  # "answered", "abstained", "partial"
    answer: str
    citations: list[CitationItem]
    claims: list[str]
    warnings: list[str]
    latency_ms: float


class FeedbackRequest(BaseModel):
    """User feedback payload for a specific request ID."""

    request_id: str
    rating: int = Field(..., ge=1, le=5, description="1 to 5 satisfaction rating")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    """Feedback acknowledgment response."""

    status: str = "recorded"
    request_id: str
