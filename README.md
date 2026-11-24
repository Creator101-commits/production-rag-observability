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

Kaggle kernel v13, git SHA `94ac5e4`, Tesla T4. Frozen system: `hybrid-rrf-k10-w1-1` + `cross_encoder` rerank@20, 512-token chunks at 15% overlap, `qwen2.5-3b-fp16` with 5 evidence chunks, abstain threshold 0.00034 (effectively inactive). This is the schema-bound correction run: `reason` is now an enforced `anyOf` string-or-null and a post-generation guard counts records violating the output bounds; `max_new_tokens=768` and cap-truncated parse failures still abstain as `truncated`.

### Retrieval ablations (test, retrieval-only)

| System | nDCG@10 | Hit@5 | MRR@10 |
| :--- | ---: | ---: | ---: |
| bm25 | 0.602 | 0.946 | 0.727 |
| dense (BGE) | 0.517 | 0.811 | 0.689 |
| hybrid-rrf-k10-w1-1 (frozen retrieval) | 0.607 | 0.919 | 0.753 |
| hybrid-rrf-k10-w2-1 | 0.625 | 0.919 | 0.759 |
| hybrid-rrf-k10-w1-1 + rerank-cross_encoder | 0.617 | 0.973 | 0.883 |
| hybrid-rrf-k60-w2-1 | 0.610 | 0.919 | 0.744 |

### End-to-end test metrics (frozen config, 46 questions: 37 answerable, 9 unanswerable)

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
| NLI claim support (estimate) | 0.661 (37/56 claims; whole-chunk 0.125) |
| Abstention sources | model_abstain 9, citation_validation 1 |
| Parse failures / token-cap hits | 0 / 0 (truncation rate 0.000) |
| Schema-bound violations | 0 on dev, 0 on test |
| GPU-seconds | 6700.4 |

### Generator x context ablation (test)

| Candidate | Req-point recall | False-abstain | False-answer | Parse fail | Cap hits | Truncation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-3b-fp16 k3 | 0.842 [0.721, 0.946] | 0.027 | 0.000 | 1 | 0 | 0.000 |
| qwen2.5-3b-fp16 k5 (frozen) | 0.869 [0.761, 0.950] | 0.027 | 0.000 | 0 | 0 | 0.000 |
| qwen2.5-7b-nf4 k3 | 0.910 [0.811, 0.991] | 0.000 | 0.111 | 0 | 0 | 0.000 |
| qwen2.5-7b-nf4 k5 | 0.405 [0.243, 0.568] | 0.541 | 0.000 | 0 | 0 | 0.000 |

### Per-stage latency (from exported spans, ms)

`request` spans cover only the context-build retrieval sweep (70 questions through retrieval, fusion, and rerank); they exclude generation. `generation` spans are one per `provider.generate` call: 280 calls = 70 questions x 4 generator candidates. The frozen candidate averages 2,870 prompt and 128 completion tokens per call.

| Stage | P50 | P95 |
| :--- | ---: | ---: |
| retrieval | 13.1 | 15.0 |
| fusion | 0.1 | 0.1 |
| rerank | 664.9 | 677.0 |
| generation (per call) | 18918.1 | 46551.2 |
| nli_scoring | 496.7 | 1481.7 |
| request (retrieval sweep only) | 678.6 | 691.6 |

### What the figure shows

- (a) Retrieval ablation, test: all 13 systems on nDCG@10 and Hit@5.
  BM25 (0.602) beats dense BGE (0.517); the best nDCG@10 is hybrid
  k10 w2:1 at 0.625, while the frozen end-to-end row (k10 w1:1 plus
  cross-encoder rerank, amber) reaches the best Hit@5, 0.973, and MRR
  0.883.
- (b) Stage latency, test spans on a log axis: generation dominates by
  orders of magnitude (median about 19.0 s per call, pooled over 280
  calls, tail to about 80 s); the cross-encoder rerank adds about
  0.67 s and NLI scoring about 0.50 s median with a tail to about
  2.1 s; retrieval (13 ms) and fusion (0.1 ms) are negligible.
- (c) End-to-end quality of the frozen system: required-point recall
  0.869 [0.761, 0.950], all-points-covered 0.811, abstain F1 0.947,
  citation label validity a perfect 1.000, citation page precision
  0.799, citation hit rate 0.861, and the weakest number, NLI claim
  support, an estimated 0.661.
- (d) Generator x context, test: the frozen 3B k=5 candidate reaches
  recall 0.869 with false-abstain 0.027; 7B k=3 posts a slightly higher
  recall (0.910) but the 7B k=5 candidate collapses entirely (recall
  0.405, false-abstain 0.541).

The headline: the frozen hybrid retrieval plus cross-encoder rerank
and 3B k=5 generator answers 87% of required evidence points with
perfect citation validity, and generation is essentially the entire
serving cost.

## Findings

- **RQ1 (lexical vs dense)**: BM25 (nDCG@10 0.602, Hit@5 0.946) beats dense BGE (0.517, 0.811) on test; the best hybrid reaches 0.625 nDCG@10. Dense actually won on dev, so the ordering is not stable across splits; with 37 answerable questions and no CIs on these ablations, small differences should not be over-read.
- **RQ2 (chunking)**: 512 tokens won on dev Hit@5 (0.895 vs 0.842 at 256 and 768).
- **RQ3 (hybrid + rerank)**: the frozen `hybrid-rrf-k10-w1-1` plus BGE cross-encoder reaches test Hit@5 0.973 and MRR@10 0.883 versus 0.919 / 0.753 without reranking, so reranking stayed in the frozen config.
- **RQ4 (abstention)**: the score threshold is effectively inactive (0.00034); abstention is driven by the model (9 model_abstain) and citation validation (1). Abstain precision 0.900, recall 1.000, F1 0.947; false-answer rate 0.000 and false-abstain rate 0.027 on test.
- **RQ5 (stage cost)**: generation dominates recorded stage time (P50 about 18.9 s and P95 about 46.6 s per `provider.generate` call, 280 calls across 4 candidates x 70 questions); reranking adds about 0.66 s; retrieval and fusion are negligible. The frozen candidate averages 2,870 prompt and 128 completion tokens per call. GPU-seconds 6700.4 for the full benchmark on a Tesla T4.
- **The v13 schema-bound fix worked**: `reason` is now an enforced `anyOf` string-or-null, and the post-generation guard counted zero bound violations on dev and test across all four candidates; every candidate also has zero token-cap hits (v12's 5 test cap hits were run-on `reason` strings). Frozen-candidate headline metrics are unchanged from v12: required-point recall 0.869 [0.761, 0.950], abstain F1 0.947.
- **Honest selection caveat**: dev selection chose `qwen2.5-3b-fp16` k_ctx=5 (dev recall 1.000, all 5 dev unanswerables abstained, dev false-abstain 0.000). On test, `qwen2.5-7b-nf4` k_ctx=3 scored higher recall (0.910 [0.811, 0.991] vs 0.869 [0.761, 0.950]); the intervals overlap on 37 answerable questions and there was no re-selection, but 7b-nf4 k3 also shows a nonzero false-answer rate (0.111 vs 0.000). `qwen2.5-7b-nf4` k_ctx=5 collapsed (recall 0.405, false-abstain 0.541): its dev-calibrated threshold is 0.970 because dev question q063 was the one unanswerable the model answered (top retrieval score 0.962), and the all-dev-unanswerables-abstain constraint forced the threshold above it; that threshold then abstained 8 further answerable dev questions (dev false-abstain 0.474) and generalized poorly to test.

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

- Revision note: kernel v11 (git 69bf7ef) completed a frozen test read before the v12 truncation correction; dev evidence showed every parse failure hit the 384-token cap. v12 (git 7105039) raised `max_new_tokens` to 768 and bounded most schema fields; its `reason` field still used an unenforced type-union form, which dev evidence showed produced run-on strings past the cap. v13 corrects `reason` to an enforced `anyOf` schema and adds a post-generation bound-violation count; on the v13 run (git 94ac5e4) the guard reported 0 violations on dev and test and every candidate had 0/70 cap hits, while headline metrics stayed at required-point recall 0.869 and abstain F1 0.947. v11 headline test numbers, disclosed for transparency: required-point recall 0.595, abstain F1 0.533, false-abstain rate 0.351, 13/46 records cap-truncated. v12: required-point recall 0.869, abstain F1 0.947, false-abstain rate 0.027, 5/46 cap-truncated (all abstained as `truncated`).
- The eval set is 70 questions authored by a single person; treat it as a smoke benchmark, not a broad capability measure.
- Required-point scoring is lexical after normalization (NFKC, lowercase, whitespace collapse, hyphen-insensitive); paraphrases outside the acceptable-alternative lists count as misses.
- NLI claim support is a model-based estimate (`cross-encoder/nli-deberta-v3-base`, entailment argmax), not verified entailment.
- The CPU Docker image uses the fake provider and fixture corpus; real model quality is only measured on Kaggle.

## Reproducing the figure

`image.png` is rendered only from `results/` (`metrics.json`,
`manifest.json`, `traces.jsonl`) via `scripts/render_figure.py`. Run
`uv run --with matplotlib python scripts/render_figure.py` in an
environment with Matplotlib; the plotting dependency is separate from
the core runtime.

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.
