![Banner](./image.png)

# Production RAG Observability

> A citation-grounded question-answering service with observable retrieval, reranking, and generation stages.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/production-rag-observability?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/production-rag-observability)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/production-rag-observability)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/production-rag-observability)
![GitHub](https://img.shields.io/github/license/Creator101-commits/production-rag-observability)

A production-grade Retrieval-Augmented Generation (RAG) microservice grounded on version-pinned NIST AI Risk Management Framework 1.0 (NIST AI 100-1) technical documents. Features token-aware provenance chunking, hybrid BM25 and dense Reciprocal Rank Fusion, cross-encoder reranking, claim-level citation validation, an evidence-based code abstention policy, and stage-level OpenTelemetry-compatible tracing with automatic PII redaction.

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

- **Token-Aware Provenance Chunking**: Preserves document, page, and section boundaries with stable SHA-256 chunk identifiers and navigation links.
- **Hybrid Retrieval (RRF)**: Merges Okapi BM25 lexical scores and dense semantic cosine embeddings with Reciprocal Rank Fusion.
- **Cross-Encoder Reranking**: Re-scores candidate pools with joint query-document phrase alignment and lexical coverage.
- **Citation Verification**: Parses bracketed citation identifiers in generated text, verifying every citation against retrieved passages and rejecting phantom references.
- **Code-Level Abstention Policy**: Enforces minimum evidence confidence thresholds, returning structured abstentions on unanswerable or out-of-scope prompts.
- **Full-Lifecycle Tracing**: Captures per-stage span timings, token consumption, estimated dollar cost, and redacted attributes.

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

```text
Versioned NIST PDFs
       |
       v
Chunking (token targets + page provenance)
       |
       +------------+------------+
       |                         |
       v                         v
   BM25 Index               Dense Index
       |                         |
       +------------+------------+
                    |
               RRF Fusion
                    |
              Cross-Encoder
                    |
            Evidence Threshold
             /              \
       (>= threshold)   (< threshold)
           /                  \
   Fake Provider           Abstain Policy
          |                    |
   Citation Validator          |
          \                    /
        Structured Response + Trace
```

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
