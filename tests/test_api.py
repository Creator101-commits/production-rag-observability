"""Tests for FastAPI service endpoints using TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_system.api.app import create_app

client = TestClient(create_app())


def test_api_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_api_ready() -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["chunks_indexed"] > 0


def test_api_query_answerable() -> None:
    payload = {"query": "What are the core functions of NIST AI RMF?"}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("answered", "abstained")
    assert "answer" in data
    assert "citations" in data
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_api_query_unanswerable() -> None:
    payload = {"query": "What is the secret recipe for Coca-Cola in 2026?"}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "abstained"
    assert "cannot answer" in data["answer"].lower()


def test_api_query_empty_bad_request() -> None:
    payload = {"query": "   "}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 400


def test_api_feedback() -> None:
    payload = {"request_id": "req-12345", "rating": 5, "comment": "Accurate citations!"}
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert data["request_id"] == "req-12345"
