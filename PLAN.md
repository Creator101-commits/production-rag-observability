# Build Plan: Production RAG with Observability

## Goal

Build a citation-grounded question-answering service over a version-pinned public technical-document corpus and make every retrieval and generation stage observable. The system will be judged with a frozen evaluation set containing answerable, unanswerable, and adversarial questions, with explicit measurements for retrieval, citation support, groundedness, abstention, latency, token use, cost, and failures.

## Scope

### Included in v1

- Acquire a controlled collection of public NIST AI Risk Management Framework documents and record exact source versions.
- Extract text from PDFs while preserving document, page, heading, and chunk provenance.
- Compare a small set of chunk sizes and overlaps using retrieval evaluation rather than selecting them by intuition.
- Build BM25, dense, and hybrid reciprocal-rank-fusion retrieval.
- Rerank the top candidates with a cross-encoder.
- Generate structured answers containing claims, citation identifiers, and an abstention state.
- Resolve citations to exact retrieved chunks and source pages.
- Reject citations that were not present in the model context.
- Add an evidence threshold and code-level abstention policy.
- Expose `/health`, `/ready`, `/query`, and `/feedback` FastAPI endpoints.
- Provide a deterministic fake generation provider so tests and CI require no paid API.
- Support one configured production LLM provider behind an interface.
- Build a versioned evaluation set with expected answerability and evidence references.
- Measure retrieval recall, citation precision/recall, answer groundedness, answer usefulness, abstention precision/recall, latency, tokens, and cost.
- Trace ingestion, retrieval, reranking, generation, validation, and response serialization with Langfuse and local-safe telemetry.
- Add redaction rules, timeouts, retries, structured errors, and request correlation IDs.
- Package the service with Docker and run tests plus a small offline evaluation in CI.

### Explicitly outside v1

- Open-domain web search.
- User-uploaded private documents.
- Legal, medical, financial, or compliance advice.
- Fine-tuning an embedding, reranking, or generation model.
- Multi-agent workflows.
- Autonomous actions based on generated answers.
- User authentication, billing, multi-tenancy, and production cloud deployment.
- Guaranteed truthfulness based on prompting.
- Logging raw secret or sensitive content to a third-party service by default.
- Using an LLM-as-judge score as the only quality measure.

## Research questions

1. How much do hybrid retrieval and reranking improve evidence recall over BM25 and dense-only baselines?
2. Which chunking configuration best balances evidence recall, context precision, latency, and cost?
3. How often do generated citations resolve to retrieved evidence and actually support the associated claims?
4. Can an evidence-based abstention policy reduce unsupported answers without rejecting too many answerable questions?
5. Which pipeline stage dominates P50 and P95 latency and estimated cost?

## Stack

- **Language:** Python 3.12
- **Packaging:** `uv`, `pyproject.toml`, and locked dependencies
- **Service:** FastAPI, Uvicorn, Pydantic, and HTTPX
- **Extraction:** pypdf plus explicit failure reporting for image-only pages
- **Tokenization and embeddings:** sentence-transformers and PyTorch
- **Retrieval:** BM25S, exact vector search for fixtures, and FAISS CPU for the corpus
- **Reranking:** sentence-transformers cross-encoder
- **Generation:** provider protocol with one hosted adapter and a deterministic test double
- **Evaluation:** direct retrieval/citation metrics, RAGAS as a secondary evaluator, pandas, NumPy, and scikit-learn
- **Observability:** Langfuse plus structured Python logging and OpenTelemetry-compatible context
- **Storage:** JSONL and Parquet for metadata/evaluation; local index artifacts for v1
- **Quality and delivery:** pytest, Ruff, mypy, Docker, and GitHub Actions

## Architecture

```text
Versioned public PDFs
        |
        v
download -> hash -> parse -> clean -> chunk -> provenance manifest
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                         BM25 index                vector index
                              |                         |
                              +------------+------------+
                                           |
                                      RRF fusion
                                           |
                                      cross-encoder
                                           |
                                           v
Client -> FastAPI -> request validation -> evidence threshold
                                           |
                                +----------+----------+
                                |                     |
                             abstain             generation
                                                      |
                                             citation validator
                                                      |
                                                      v
                                           structured response

Every stage -> trace/span -> redacted metrics, latency, tokens, errors, cost
```

### Components

#### `src/rag_system/ingestion/`

Downloads version-pinned documents, verifies checksums, extracts page text, detects extraction failures, normalizes conservatively, locates headings where possible, creates chunks, and writes a provenance manifest. Each chunk has a stable ID derived from source hash and chunking configuration.

#### `src/rag_system/retrieval/`

Builds and loads BM25 and vector indexes, encodes queries, produces ranked candidates, performs reciprocal-rank fusion, and records stage-specific scores and ranks. Retrieval returns structured evidence objects rather than raw strings.

#### `src/rag_system/reranking/`

Scores query-evidence pairs with a pinned cross-encoder, retains original retrieval provenance, and emits the final context order. Batch size, maximum sequence length, model revision, and candidate depth are configuration values.

#### `src/rag_system/generation/`

Defines an asynchronous provider interface and constructs a structured generation request from only the approved evidence. The output schema includes answer text, claim-level citation IDs, and an abstention reason. Provider-specific retry and timeout behavior remains behind the adapter.

#### `src/rag_system/citations/`

Parses generated citation IDs, rejects unknown references, maps valid citations to document/page/chunk metadata, checks claim coverage, and creates the public source list. Validation failures produce a controlled abstention or explicitly marked partial response according to configuration.

#### `src/rag_system/evaluation/`

Runs retrieval-only and end-to-end evaluations against a versioned dataset. It calculates deterministic metrics directly and may invoke judge models for secondary groundedness/usefulness signals with cached, versioned judge inputs and outputs.

#### `src/rag_system/observability/`

Creates request-level traces and stage-level spans, applies content-redaction policy, records latency, counts, model identifiers, token use, estimated cost, cache behavior, evidence IDs, validation outcomes, retries, and errors. The application remains functional when the tracing backend is unavailable.

#### `src/rag_system/api/`

Defines request/response schemas, dependency wiring, health/readiness checks, query handling, feedback capture, request IDs, and structured error mapping. It contains no retrieval or prompt logic.

## Data

### Document corpus

Use a small, version-pinned collection from the **NIST AI Risk Management Framework family**, such as the core framework, playbook or supporting guidance, and a generative-AI profile when the source and revision are clearly documented. The corpus is public, technically structured, citation-friendly, and narrow enough to support manual evidence verification.

### Source manifest

For each document record:

- `document_id`
- title and short name
- source URL
- publication or revision date
- acquisition timestamp
- media type and byte size
- SHA-256 hash
- page count
- extraction status and failed pages
- applicable usage or attribution notes

### Chunk schema

- `chunk_id`
- `document_id`
- source title and URL
- page start and end
- section heading when detected
- chunk text
- token count
- previous and next chunk IDs
- parser version
- chunking configuration hash
- source-document hash

### Chunking experiments

- Start with token targets near 256, 512, and 768 plus conservative overlap.
- Preserve page and section boundaries when practical.
- Never allow a chunk to lose its source-page mapping.
- Choose the v1 configuration using development retrieval metrics and context cost, not final answer examples.

### Evaluation dataset

Create a versioned JSONL dataset with at least these fields:

- `question_id`
- `question`
- `answerable`: boolean
- expected document/page/chunk evidence IDs
- reference answer or required answer points
- question category and difficulty
- provenance of the question authoring process
- reviewer status

The initial target is 60–100 reviewed questions: direct lookup, synthesis across chunks, terminology distinctions, numeric details, out-of-scope questions, false-premise questions, and questions whose answer is absent. Split the set into development and frozen test groups before prompt and threshold tuning.

### Storage

- Raw PDFs: `data/raw/`, ignored by Git
- Extracted pages and chunks: Parquet under `data/processed/`, ignored by Git
- Small evaluation JSONL: version controlled when licensing permits
- Indexes and caches: `indexes/`, ignored by Git
- Evaluation summaries: version-controlled tables and figures when compact; detailed traces remain external

## Retrieval and generation protocol

### Retrieval baselines

- BM25-only
- dense-only
- unweighted RRF hybrid
- tuned hybrid only if development evidence supports it
- hybrid plus cross-encoder reranking

### Context construction

- Enforce a maximum context token budget.
- Prefer diversified evidence when near-duplicate chunks occupy the top ranks.
- Include stable chunk IDs and page metadata in the model context.
- Never add source text from outside the retrieved and approved evidence list.

### Answer contract

The internal response object contains:

- `status`: answered, abstained, or partial
- `answer`
- claim-level citation IDs
- resolved source metadata
- evidence IDs used
- machine-readable validation warnings
- request ID and non-sensitive timing summary

### Abstention

- Retrieval score thresholds are calibrated on development data.
- Answerability may use retrieval features, evidence coverage, and validator results.
- The generation prompt instructs abstention, but code owns the final policy.
- Unsupported citation IDs cause validation failure.
- Test false-positive abstentions and false-negative unsupported answers separately.

## Evaluation design

### Retrieval metrics

- Evidence Recall@5 and @10
- MRR@10
- NDCG@10 when graded judgments exist
- context precision or fraction of selected chunks marked relevant

### Citation metrics

- citation identifier validity
- citation precision: cited chunks that support a claim
- citation recall: required evidence represented in citations
- claim coverage: factual claims with at least one supporting citation
- page-resolution accuracy

### Answer metrics

- required-point coverage against reviewed references
- groundedness or faithfulness, with human-reviewed checks on a fixed subset
- answer usefulness or relevance
- exact-match or numeric-match checks where applicable
- unsupported-claim rate

### Abstention metrics

- precision and recall for the abstain decision
- false-answer rate on unanswerable questions
- false-abstention rate on answerable questions
- selective accuracy as coverage changes

### Operational metrics

- P50, P95, and P99 latency by stage and end to end
- token counts and estimated generation cost
- requests, failure rate, retry rate, timeout rate, and tracing-export failure rate
- index build time, index size, and embedding cost

### Evaluation safeguards

- Direct deterministic metrics are primary where available.
- Judge-model prompts, model revisions, temperatures, and outputs are versioned.
- No single LLM judge score blocks a release by itself.
- CI uses a small deterministic regression set; a larger offline suite runs before releases.
- Threshold changes require a recorded development-set comparison and cannot be tuned on the frozen test set.

## Observability design

### Trace structure

One trace per request with spans for:

1. request validation
2. query preprocessing
3. BM25 retrieval
4. dense retrieval
5. rank fusion
6. reranking
7. context selection
8. generation
9. citation validation
10. response serialization

### Recorded attributes

- request and trace IDs
- configuration and corpus version
- retrieved chunk IDs, ranks, and numeric scores
- selected evidence IDs and token counts
- model identifiers and revisions
- stage latencies
- retries, timeout classes, and errors
- generation input/output token counts and estimated cost
- answer status and validation outcomes

Raw content is excluded by default. A local development flag may record content only after explicit redaction and must never be enabled in CI or a shared environment.

## Milestones

### 1. Reproducible service skeleton

Create package layout, locked dependencies, settings, FastAPI application factory, CLI, structured errors, tests, linting, typing, Dockerfile, and CI. Ship working health/readiness endpoints and a deterministic fake provider.

### 2. Document acquisition and extraction

Implement versioned downloads, hashes, PDF extraction, page-level validation, provenance, and manifests. Ship a corpus report showing document versions, pages, extraction failures, and reproducible raw-to-page processing.

### 3. Chunking and retrieval benchmark

Implement token-aware chunking, stable IDs, BM25 and dense indexes, and a development evidence set. Compare predefined chunk configurations using retrieval metrics and choose one without generation.

### 4. Hybrid retrieval and reranking

Implement RRF, cross-encoder reranking, candidate provenance, index manifests, and cache invalidation. Ship quality-versus-latency results and tests ensuring reranking cannot introduce unknown chunks.

### 5. Structured generation and citation validation

Implement the provider protocol, context builder, structured response schema, citation parser/resolver, and validation policy. Ship deterministic integration tests for valid answers, invented citations, malformed output, missing evidence, timeouts, and provider errors.

### 6. Evaluation-set construction

Create and review the development and frozen test questions, expected evidence, answer points, and answerability labels. Ship an evaluation data card explaining authorship, review, coverage, limitations, and leakage controls.

### 7. Abstention calibration

Implement evidence features, score thresholds, selective-accuracy curves, and validator-driven abstention. Tune on development data only. Ship the chosen threshold policy and errors by answerability category.

### 8. End-to-end observability

Add request traces, pipeline spans, redaction, stage metrics, token/cost estimation, and graceful tracing-backend failure. Ship local dashboards or documented Langfuse views for latency, failure, abstention, token use, and retrieval diagnostics.

### 9. Frozen evaluation and resilience testing

Freeze corpus, indexes, prompt, provider revision, thresholds, and evaluation configuration. Run the full test suite plus malformed PDFs, empty queries, retrieval misses, slow providers, invalid citations, and tracing outages. Ship results with uncertainty and failure examples.

### 10. Final research artifact

Write the report with corpus, architecture, retrieval ablations, generation and citation evaluation, abstention trade-offs, observability findings, latency/cost analysis, security and privacy considerations, limitations, and reproduction instructions. Verify Docker and local workflows from clean environments before tagging v1.

## Open questions

1. **Exact corpus:** Freeze the NIST document list and revisions before authoring the final evaluation set.
2. **Hosted model:** Select one provider/model based on structured-output support, terms, cost ceiling, and reproducible version identification.
3. **Local fallback:** Decide whether v1 includes a real local generator or only a deterministic test provider plus hosted production adapter.
4. **Vector index:** Confirm FAISS compatibility on target platforms or use another local index with reproducible exact-search checks.
5. **Embedding and reranking models:** Pin models only after testing license, size, context length, scientific-document behavior, and CPU latency.
6. **Evaluation authorship:** Determine how many questions receive independent review and how disagreements over supporting evidence are resolved.
7. **Abstention threshold:** Choose the primary optimization objective, such as minimizing unsupported answers subject to a minimum answer coverage.
8. **Trace backend:** Decide between Langfuse Cloud and self-hosting. Default telemetry must remain content-redacted in either case.
9. **Cost calculation:** Define whether costs come from provider-reported usage, a versioned price table, or both. Price tables are time-sensitive and must carry a date.
10. **Release gate:** Establish separate deterministic CI thresholds and larger offline release thresholds after baseline variance is measured.
