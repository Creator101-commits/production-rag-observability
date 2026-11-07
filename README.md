![Banner](./image.png)

# Production RAG Observability

> A citation-grounded question-answering service with observable retrieval, reranking, and generation stages.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/production-rag-observability?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/production-rag-observability)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/production-rag-observability)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/production-rag-observability)
![GitHub](https://img.shields.io/github/license/Creator101-commits/production-rag-observability)

An offline FastAPI RAG prototype using hand-written NIST-themed teaching paraphrases. It uses BM25 plus deterministic hashed lexical vectors, RRF, lexical coverage reranking, and a deterministic extractive fake generator. It does not ingest official PDFs, run a neural cross-encoder or LLM, or export OpenTelemetry. The fixture page and section labels are local teaching labels, not verified NIST citations. Production deployment, semantic support checking, and hosted generation remain planned.

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [CLI Reference](#cli-reference)
- [Architecture and Observability](#architecture-and-observability)
- [Development and Testing](#development-and-testing)
- [License](#license)

## Features

- **Approximate Token Chunking**: Preserves fixture document and page boundaries with stable SHA-256 chunk identifiers and navigation links.
- **Hybrid Retrieval (RRF)**: Merges Okapi BM25 lexical scores and hashed lexical vectors with cosine similarity with Reciprocal Rank Fusion.
- **Lexical Coverage Reranking**: Re-scores candidate pools with token coverage and bigram alignment; the legacy class name does not load a model.
- **Citation Verification**: Parses bracketed citation identifiers in generated text, verifying every citation against retrieved passages and rejecting phantom references.
- **Code-Level Abstention Policy**: Enforces minimum evidence confidence thresholds, returning structured abstentions on unanswerable or out-of-scope prompts.
- **In-Memory Tracing**: Captures per-stage span timings, token consumption, illustrative configured token-cost estimates (no provider billing), and redacted attributes.

## Quickstart

Run the offline benchmark evaluation across answerable and adversarial prompts:

```bash
uv run rag-service benchmark
```

Query the corpus directly from the CLI:

```bash
uv run rag-service query "What are the core functions of NIST AI RMF?"
```

## Installation

Ensure Python 3.12+ and `uv` are installed:

```bash
git clone https://github.com/Creator101-commits/production-rag-observability.git
cd production-rag-observability
uv sync
```

## API Endpoints

Launch the HTTP service:

```bash
uv run uvicorn rag_system.api.app:create_app --factory --port 8000
```

### 1. Health and Readiness Probes
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 2. Query Answering
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the characteristics of trustworthy AI?"}'
```

Response format:
```json
{
  "request_id": "c16d5573-...",
  "trace_id": "90eb6441-...",
  "status": "answered",
  "answer": "Section 2: Characteristics of Trustworthy AI Systems... [nist_ai_rmf_1_0_p12_c0_09a47dd6]",
  "citations": [
    {
      "citation_id": "nist_ai_rmf_1_0_p12_c0_09a47dd6",
      "document_id": "nist_ai_rmf_1_0",
      "document_title": "NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
      "page_start": 12,
      "page_end": 12,
      "text_snippet": "Section 2: Characteristics of Trustworthy AI Systems..."
    }
  ],
  "claims": ["..."],
  "warnings": [],
  "latency_ms": 14.82
}
```

### 3. User Feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "c16d5573-...", "rating": 5, "comment": "Precise citation."}'
```

## CLI Reference

```bash
# Display package version
uv run rag-service version

# Run benchmark evaluation suite
uv run rag-service benchmark

# Query the corpus
uv run rag-service query "How does GOVERN cultivate risk culture?"

# Generate markdown benchmark report
uv run rag-service report --output artifacts/rag_benchmark_report.md
```

## Architecture and Observability

The figure at the top is generated from actual fixture evaluation. The implemented path is approximate-token chunking, lexical/hash retrieval, RRF, lexical reranking, deterministic generation, citation-ID enforcement, and a structured API response. Timing spans are held in memory and capped at 1,000 completed traces.

## Development and Testing

```bash
# Run pytest with branch coverage
uv run pytest

# Run Ruff linter
uv run ruff check .

# Run static type checker
uv run mypy src
```

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.

## Evaluation limits

`benchmark` uses a small teaching question suite; `image.png` evaluates its seven test questions only. Citation-ID membership verifies references, not semantic entailment or answer correctness. Answers with unknown or missing required citations abstain. Retrieval metrics are document-level, so they do not prove passage relevance. Token counts are approximations and cost estimates are hypothetical. Fixture prose is inspired by [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework); official page-level ingestion and source hashing remain planned.

## Reproducing the figure

`image.png` contains only Matplotlib graphs. Run `PYTHONPATH=src python scripts/render_figure.py` in a Python environment containing the project dependencies and Matplotlib. The plotting environment is separate from the core runtime when Matplotlib is absent. `scripts/figure-data.json` stores the plotted measurements or analytical inputs. Captions distinguish real packaged datasets, synthetic fixtures, analytical calculations, and local timings. No GPU performance is inferred from CPU demonstrations.
