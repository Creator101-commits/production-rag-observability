"""Content redaction policy for safe telemetry and log export."""

from __future__ import annotations

import re

# Patterns for sensitive entity redaction
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_API_KEY_PATTERN = re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,})\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_sensitive_text(text: str) -> str:
    """Redact personal identifiable information and secrets from telemetry strings."""
    redacted = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = _PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    redacted = _SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
    return redacted
