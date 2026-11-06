"""Tests for production RAG CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from rag_system.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_query() -> None:
    result = runner.invoke(app, ["query", "What is GOVERN in NIST AI RMF?"])
    assert result.exit_code == 0
    assert "RAG Answer" in result.output
    assert "Latency" in result.output


def test_cli_benchmark() -> None:
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0
    assert "Production RAG Evaluation Summary" in result.output
    assert "Retrieval Recall@5" in result.output


def test_cli_report(tmp_path: Path) -> None:
    out = tmp_path / "rag_report.md"
    result = runner.invoke(app, ["report", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Production RAG with Observability Benchmark Report" in content


def test_cli_report_stdout() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "Production RAG with Observability Benchmark Report" in result.output
