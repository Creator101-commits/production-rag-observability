![Banner](./image.png)

# Production RAG Observability

> A citation-grounded question-answering service over the official NIST AI Risk Management Framework documents, with retrieval, reranking, abstention, and OpenTelemetry observability.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/production-rag-observability?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/production-rag-observability)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/production-rag-observability)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/production-rag-observability)
![GitHub](https://img.shields.io/github/license/Creator101-commits/production-rag-observability)

A FastAPI RAG service that ingests the two official NIST AI RMF PDFs (AI RMF 1.0 and the Generative AI Profile), chunks them with page-aware token chunking, retrieves with BM25 + BGE dense + weighted RRF, optionally reranks with a BGE cross-encoder, and generates structured JSON answers whose claims cite only retrieved evidence. A code-level abstention threshold plus the model's own abstain flag control when the service refuses to answer. Every stage emits real OpenTelemetry spans with redacted attributes. A deterministic fake provider keeps tests, CI, and the CPU Docker image model-free.

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [CLI Reference](#cli-reference)
- [Architecture and Observability](#architecture-and-observability)
- [Methodology](#methodology)
- [Results](#results)
- [Findings](#findings)
- [Reproduction](#reproduction)
- [Development and Testing](#development-and-testing)
- [Evaluation limits](#evaluation-limits)
- [Reproducing the figure](#reproducing-the-figure)
- [License](#license)

## Features

- **Pinned corpus ingestion**: downloads the two official NIST PDFs, verifies SHA-256, extracts per-page text with pypdf, normalizes (NFKC, whitespace collapse, dehyphenation), and records a provenance manifest with page counts and failed pages.
- **Page-aware token chunking**: stable chunk IDs hashed from source SHA, offsets, and config; `page_start`/`page_end`, prev/next links, and token counts preserved.
- **Hybrid retrieval**: Okapi BM25 plus `BAAI/bge-base-en-v1.5` dense embeddings fused with weighted reciprocal rank fusion.
- **Optional cross-encoder reranking**: `BAAI/bge-reranker-base` at a configurable candidate depth, chosen on dev only.
- **Structured generation**: `Qwen/Qwen2.5-3B-Instruct` emits a JSON contract (`answer`, `claims` with `c1..cN` labels, `abstain`, `reason`); code maps labels to chunk IDs and rejects unknown labels.
- **Code-level abstention**: a threshold on the top post-retrieval score, tuned on dev only, plus the model's own abstain flag.
- **Real OpenTelemetry**: spans for request, retrieval, fusion, rerank, threshold, generation, citation validation, serialization, and NLI; redacted attributes (query SHA-256 + length, chunk IDs, scores, token counts, model IDs); in-memory, JSONL, and optional OTLP exporters.
- **Deterministic fake provider**: default for tests, CI, and Docker; no model downloads.

## Quickstart

Offline service (fixture corpus, fake provider):

```bash
uv sync
uv run uvicorn rag_system.api.app:create_app --factory --port 8000
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the core functions of NIST AI RMF?"}'
```

Run the offline CLI benchmark over the fixture eval set:

```bash
uv run rag-service benchmark
uv run rag-service query "How does GOVERN cultivate risk culture?"
```

## Installation

Python 3.11-3.13 and `uv`:

```bash
git clone https://github.com/Creator101-commits/production-rag-observability.git
cd production-rag-observability
uv sync                 # core only, no model downloads
uv sync --extra neural  # adds torch/transformers/sentence-transformers
uv sync --extra otlp    # adds OpenTelemetry OTLP exporter
```

## API Endpoints

### 1. Health and Readiness

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 2. Query Answering

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is residual risk?"}'
```

Response:

```json
{
  "request_id": "...",
  "trace_id": "...",
  "status": "answered",
  "answer": "...",
  "citations": [{"citation_id": "...", "document_id": "...", "page_start": 0, "page_end": 0, "text_snippet": "..."}],
  "claims": ["..."],
  "warnings": [],
  "latency_ms": 12.3
}
```

### 3. User Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "...", "rating": 5, "comment": "Accurate citations."}'
```

## CLI Reference

```bash
uv run rag-service version
uv run rag-service query "What are the core functions of NIST AI RMF?"
uv run rag-service benchmark
uv run rag-service report --output artifacts/rag_benchmark_report.md
```

## Architecture and Observability

```text
NIST PDFs -> sha256 verify -> pypdf extract -> normalize -> token chunks
   -> BM25 index + BGE dense index -> weighted RRF -> optional CE rerank
   -> evidence threshold -> Qwen structured JSON -> label->chunk resolve
   -> citation validation -> response

Every stage -> OTel span -> redacted attributes -> in-memory / JSONL / OTLP
```

Spans emitted per query: `request`, `retrieval`, `fusion` (hybrid only), `rerank`, `threshold`, `generation`, `citation_validation`, `serialization`, and `nli_scoring` (benchmark only). Attributes are redacted by default: query SHA-256 and length, chunk IDs, scores, token counts, model IDs, and outcome flags. Raw query or document text is never exported. The service keeps working if an exporter fails.

## Methodology

- **Corpus**: `nist_ai_100_1` (48 pages) and `nist_ai_600_1` (64 pages), sha256-pinned; page numbers are 1-based physical PDF indices.
- **Eval set**: `data/eval/nist_qa_v1.jsonl`, 70 questions (dev 24, test 46) authored by the maintainer from the PDF text; frozen ground truth.
- **Chunking**: token targets {256, 512, 768} at 15% overlap; size chosen on dev by Hit@5.
- **Retrieval**: BM25, BGE dense, and RRF hybrids tuned on dev by nDCG@10.
- **Rerank**: {none, BGE cross-encoder@20} chosen on dev.
- **Abstention**: threshold on the top post-retrieval score tuned on dev (maximize answerable-answer rate subject to abstaining on all dev unanswerables; falls back to max F1 and says so).
- **Test**: frozen config evaluated once; retrieval-only ablations for every system; bootstrap 95% CIs (seed 42, 1000 resamples).

## Results

Kaggle kernel v12, git SHA `7105039`, Tesla T4. Frozen system: `hybrid-rrf-k10-w1-1` + `cross_encoder` rerank@20, 512-token chunks at 15% overlap, `qwen2.5-3b-fp16` with 5 evidence chunks, abstain threshold 0.00034 (effectively inactive). This is the truncation-correction run: bounded JSON schema plus `max_new_tokens=768`, with cap-truncated parse failures abstaining as `truncated`.

### Retrieval ablations (test, retrieval-only)

| System | nDCG@10 | Hit@5 | MRR@10 |
| :--- | ---: | ---: | ---: |
| bm25 | 0.602 | 0.946 | 0.727 |
| dense (BGE) | 0.517 | 0.811 | 0.689 |
| hybrid-rrf-k10-w1-1 (frozen retrieval) | 0.607 | 0.919 | 0.753 |
| hybrid-rrf-k10-w2-1 | 0.625 | 0.919 | 0.759 |
| hybrid-rrf-k10-w1-1 + rerank-cross_encoder | 0.617 | 0.973 | 0.883 |
| hybrid-rrf-k60-w2-1 | 0.610 | 0.919 | 0.744 |

### End-to-end test metrics (frozen config, 46 questions)

| Metric | Value |
| :--- | :--- |
| Required-point recall | 0.869 [0.761, 0.950] |
| All points covered | 0.811 [0.676, 0.919] |
| Abstain precision / recall / F1 | 0.900 / 1.000 / 0.947 |
| False-answer rate (unanswerable) | 0.000 |
| False-abstain rate (answerable) | 0.027 |
| Answerable answer rate | 0.973 |
| Citation label validity | 1.000 [1.000, 1.000] |
| Citation page precision | 0.799 [0.671, 0.917] |
| Citation hit rate | 0.861 [0.722, 0.972] |
| NLI claim support (estimate) | 0.661 |
| Abstention sources | truncated 5, model_abstain 4, citation_validation 1 |
| Parse failures / token-cap hits | 5 / 5 (truncation rate 0.109) |
| GPU-seconds | 7256.9 |

### Generator x context ablation (test)

| Candidate | Req-point recall | False-abstain | False-answer | Cap hits | Truncation |
| :--- | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-3b-fp16 k3 | 0.842 [0.721, 0.946] | 0.027 | 0.000 | 2 | 0.043 |
| qwen2.5-3b-fp16 k5 (frozen) | 0.869 [0.761, 0.950] | 0.027 | 0.000 | 5 | 0.109 |
| qwen2.5-7b-nf4 k3 | 0.910 [0.811, 0.991] | 0.000 | 0.111 | 0 | 0.000 |
| qwen2.5-7b-nf4 k5 | 0.405 [0.243, 0.568] | 0.541 | 0.000 | 0 | 0.000 |

### Per-stage latency (from exported spans, ms)

| Stage | P50 | P95 |
| :--- | ---: | ---: |
| retrieval | 13.1 | 14.3 |
| fusion | 0.1 | 0.1 |
| rerank | 670.6 | 698.0 |
| generation | 20255.1 | 65305.4 |
| nli_scoring | 511.0 | 1470.9 |
| request | 683.6 | 711.0 |

## Findings

- **RQ1 (lexical vs dense)**: BM25 (nDCG@10 0.602) beats dense BGE (0.517); the best hybrid reaches 0.625. Lexical retrieval is strong on this corpus's precise terminology.
- **RQ2 (chunking)**: 512 tokens won on dev Hit@5 (0.895 vs 0.842 at 256 and 768).
- **RQ3 (hybrid + rerank)**: the frozen `hybrid-rrf-k10-w1-1` plus BGE cross-encoder reaches test Hit@5 0.973 and MRR@10 0.883 versus 0.919 / 0.753 without reranking, so reranking stayed in the frozen config.
- **RQ4 (abstention)**: the score threshold is effectively inactive (0.00034); abstention is driven by the model (4 model_abstain), cap truncation (5 truncated), and citation validation (1). Abstain precision 0.900, recall 1.000, F1 0.947; false-answer rate 0.000 and false-abstain rate 0.027 on test.
- **RQ5 (stage cost)**: generation dominates recorded stage time (P50 about 20 s per call, P95 about 65 s); reranking adds about 0.67 s; retrieval and fusion are negligible. GPU-seconds 7256.9 for the full benchmark.
- **The v12 truncation correction worked**: bounded schema fields (answer/claims/citations/reason length and item caps) plus `max_new_tokens=768` eliminated non-truncation parse failures entirely; the only 5 parse failures (10.9% of records) are token-cap hits that now abstain as `truncated` instead of emitting malformed output. Required-point recall is 0.869 [0.761, 0.950] and abstain F1 0.947.
- **Honest selection caveat**: dev selection chose `qwen2.5-3b-fp16` k_ctx=5 (dev recall 1.0, abstain F1 1.0). On test, `qwen2.5-7b-nf4` k_ctx=3 scored higher recall (0.910 vs 0.869) but also a nonzero false-answer rate (0.111 vs 0.000); the frozen config trades a little recall for perfect unanswerable abstention. `qwen2.5-7b-nf4` k_ctx=5 collapsed (recall 0.405, false-abstain 0.541).

## Reproduction

```bash
./scripts/kaggle_run.sh   # archives HEAD, versions src dataset, pushes kernel, polls, downloads to artifacts/kaggle/<ts>/, guards manifest git_sha == HEAD
```

The kernel ingests the PDFs, chunks all three sizes, tunes on dev, freezes, and evaluates test once. Outputs: `metrics.json`, `manifest.json`, `test_questions.jsonl`, `dev_questions.jsonl`, `traces.jsonl`, `report.md` under `benchmark/`; curated verbatim into `results/`.

## Development and Testing

```bash
uv sync --locked --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
docker build -t rag-system .   # CPU image, fake provider + fixture index
```

## Evaluation limits

- Revision note: kernel v11 (git 69bf7ef) completed a frozen test read before the v12 truncation correction; dev evidence showed every parse failure hit the 384-token cap. v12 (git 7105039) is the correction run and is the published result. v11 headline test numbers, disclosed for transparency: required-point recall 0.595, abstain F1 0.533, false-abstain rate 0.351, 13/46 records cap-truncated. v12: required-point recall 0.869, abstain F1 0.947, false-abstain rate 0.027, 5/46 cap-truncated (all abstained as `truncated`).
- The eval set is 70 questions authored by a single person; treat it as a smoke benchmark, not a broad capability measure.
- Required-point scoring is lexical after normalization (NFKC, lowercase, whitespace collapse, hyphen-insensitive); paraphrases outside the acceptable-alternative lists count as misses.
- NLI claim support is a model-based estimate (`cross-encoder/nli-deberta-v3-base`, entailment argmax), not verified entailment.
- The CPU Docker image uses the fake provider and fixture corpus; real model quality is only measured on Kaggle.

## Reproducing the figure

`image.png` is rendered from `scripts/figure-data.json` (generated from `results/metrics.json` + `manifest.json`). Run `uv run --with matplotlib python scripts/render_figure.py` in an environment with Matplotlib; the plotting dependency is separate from the core runtime.

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
