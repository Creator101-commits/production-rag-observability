"""Command-line interface for production RAG service."""

from __future__ import annotations

from pathlib import Path

import typer

from rag_system import __version__
from rag_system.citations.validator import CitationValidator
from rag_system.config import RAGConfig
from rag_system.evaluation.dataset import load_evaluation_dataset
from rag_system.evaluation.runner import BenchmarkRunner
from rag_system.generation.base import GenerationRequest
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.ingestion.chunk import chunk_document
from rag_system.ingestion.corpus import load_nist_ai_rmf_corpus
from rag_system.reranking.cross_encoder import CrossEncoderReranker
from rag_system.retrieval.hybrid import HybridRetriever

app = typer.Typer(
    help="Production RAG with Observability CLI.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def query(
    text: str = typer.Argument(..., help="Question about NIST AI RMF."),
) -> None:
    """Query the technical document corpus with citation verification."""
    config = RAGConfig()
    retriever = HybridRetriever(rrf_k=config.rrf_k)
    reranker = CrossEncoderReranker()
    generator = DeterministicFakeProvider(
        default_abstain_threshold=config.evidence_score_threshold
    )

    docs = load_nist_ai_rmf_corpus()
    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                chunk_size_tokens=config.chunk_size_tokens,
                chunk_overlap_tokens=config.chunk_overlap_tokens,
            )
        )
    retriever.index(chunks)

    candidates = retriever.retrieve(text, top_k=config.retrieval_candidate_depth)
    top_evidence, rerank_ms = reranker.rerank(text, candidates, top_k=config.rerank_top_k)

    gen_req = GenerationRequest(
        query=text,
        evidence_list=top_evidence,
        evidence_score_threshold=config.evidence_score_threshold,
    )
    gen_resp = generator.generate(gen_req)
    val_res = CitationValidator.validate(gen_resp.answer, top_evidence)

    typer.echo(f"\n--- RAG Answer ({gen_resp.status}) ---")
    typer.echo(gen_resp.answer)

    if val_res.resolved_citations:
        typer.echo("\n--- Citations ---")
        for c in val_res.resolved_citations:
            typer.echo(f"- [{c.citation_id}] {c.document_title} (p. {c.page_start})")

    if val_res.warnings:
        typer.echo("\n--- Validation Warnings ---")
        for w in val_res.warnings:
            typer.echo(f"! {w}")

    typer.echo(f"\nRerank Latency: {rerank_ms:.2f} ms | Gen Latency: {gen_resp.latency_ms:.2f} ms")


@app.command()
def benchmark() -> None:
    """Run offline evaluation over NIST AI RMF question dataset."""
    questions = load_evaluation_dataset()
    runner = BenchmarkRunner()
    metrics = runner.run_evaluation(questions)

    typer.echo("\n--- Production RAG Evaluation Summary ---")
    typer.echo(f"Retrieval Recall@5:      {metrics.retrieval_recall_at_5 * 100:.1f}%")
    typer.echo(f"Retrieval MRR:           {metrics.retrieval_mrr:.4f}")
    typer.echo(f"Citation Validity Rate:  {metrics.citation_validity_rate * 100:.1f}%")
    typer.echo(f"Abstention Precision:    {metrics.abstention_precision * 100:.1f}%")
    typer.echo(f"Abstention Recall:       {metrics.abstention_recall * 100:.1f}%")
    typer.echo(f"False Answer Rate:       {metrics.false_answer_rate * 100:.1f}%")
    typer.echo(f"False Abstention Rate:   {metrics.false_abstention_rate * 100:.1f}%")
    typer.echo(f"Mean Latency:            {metrics.mean_latency_ms:.2f} ms")
    typer.echo(f"P95 Latency:             {metrics.p95_latency_ms:.2f} ms")
    typer.echo(f"Total Est. Cost:         ${metrics.total_cost_usd:.6f}")


@app.command()
def report(
    output: Path | None = typer.Option(None, "--output", "-o", help="Path to write report."),  # noqa: B008
) -> None:
    """Generate markdown evaluation and architecture report."""
    questions = load_evaluation_dataset()
    runner = BenchmarkRunner()
    metrics = runner.run_evaluation(questions)

    doc_text = (
        "# Production RAG with Observability Benchmark Report\n\n"
        "## Executive Summary\n\n"
        "Evaluation results across answerable, synthesis, adversarial, and unanswerable queries "
        "on the versioned NIST AI Risk Management Framework 1.0 corpus.\n\n"
        "## Retrieval and Generation Metrics\n\n"
        "| Metric | Result |\n"
        "| :--- | :--- |\n"
        f"| `Retrieval Recall@5` | {metrics.retrieval_recall_at_5 * 100:.1f}% |\n"
        f"| `Retrieval MRR` | {metrics.retrieval_mrr:.4f} |\n"
        f"| `Citation Validity` | {metrics.citation_validity_rate * 100:.1f}% |\n"
        f"| `Abstention Precision` | {metrics.abstention_precision * 100:.1f}% |\n"
        f"| `Abstention Recall` | {metrics.abstention_recall * 100:.1f}% |\n"
        f"| `False Answer Rate` | {metrics.false_answer_rate * 100:.1f}% |\n"
        f"| `False Abstention Rate` | {metrics.false_abstention_rate * 100:.1f}% |\n"
        f"| `Mean Latency` | {metrics.mean_latency_ms:.2f} ms |\n"
        f"| `P95 Latency` | {metrics.p95_latency_ms:.2f} ms |\n"
        f"| `Estimated Total Cost` | ${metrics.total_cost_usd:.6f} |\n\n"
        "## Validation and Observability\n\n"
        "- All citations resolve to retrieved chunk IDs with validated source provenance.\n"
        "- Hallucinated citation IDs trigger structured warnings and code-level abstention.\n"
        "- Telemetry traces record spans, latency, token consumption, and redacted attributes.\n"
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(doc_text, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(doc_text)


if __name__ == "__main__":
    app()
