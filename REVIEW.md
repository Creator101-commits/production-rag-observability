# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, benchmark
orchestration, observability exporters, project claims, Kaggle scripts,
and README figures.

## Verified locally

- pytest, ruff, and mypy pass on the CPU-safe suite; neural paths are
  exercised through injected encoder, scorer, NLI, and provider fakes.
- `uv sync --locked --all-groups` resolves cleanly.
- The full `run_full_benchmark` orchestrator was smoke-tested locally
  with fake components on the real eval file and fixture corpus; it
  writes metrics, manifest, per-question JSONL, traces, and report.
- OpenTelemetry spans carry redacted attributes only; exporter failures
  are swallowed and counted.

## Kaggle

The GPU benchmark kernel (`kaggle/run_kaggle.py`) downloads and
sha256-verifies the pinned NIST PDFs, resolves and records HF revisions,
runs dev tuning, freezes the config, and evaluates test once. Results
are curated verbatim into `results/`; the manifest `git_sha` is guarded
by `scripts/kaggle_run.sh`.

## Limits

- The eval set is 70 questions authored by a single person.
- Required-point scoring is lexical after normalization.
- NLI claim support is a model-based estimate, not verified entailment.
- The CPU Docker image uses the fake provider and fixture corpus; real
  model quality is only measured on Kaggle.
- Docker was not available in the local environment; the CI docker job
  covers the build and smoke test.

## Protocol revision: v12 generation truncation correction

Kaggle kernel v11 (git 69bf7ef) completed with a frozen test read. Reviewing its records showed every parse_failure in every generator candidate had completion_tokens == 384 == max_new_tokens: the model wrote run-on answer/claim strings and the JSON was cut mid-string. Dev evidence alone showed the same pattern (4/24 dev parse failures for the 3B k3, 3B k5, and 7B k3 candidates), which justified the correction. v12 tightens the output schema (answer maxLength 300, at most 4 claims of maxLength 300 with 1-3 citations each, reason maxLength 160), raises max_new_tokens to 768, shortens the prompted answer/claim budgets, and adds per-record hit_token_cap plus an aggregate truncation_rate; cap-truncated parse failures are classified "truncated" rather than "parse_failure". The v11 test read stands disclosed: required_point_recall 0.595, abstain F1 0.533, false_abstain_rate 0.351, 13/46 records cap-truncated.

## Verified on Kaggle: v12 run

- Smoke kernel `production-rag-observability-smoke` on git SHA `7105039`: delegated to the main kernel with `RAG_SMOKE=1` (3B fp16 candidate, 15 dev questions, k_ctx=5); manifest `git_sha` matched; constrained decoding enforced the schema bounds (raw answer exactly 300 chars at the cap); all 15 dev records parsed with zero parse failures and zero cap hits; gpu_seconds 210.8.
- Full kernel `production-rag-observability-benchmark` on git SHA `71050399f2eb62ab0db48326fa90928df9ed78ad`: frozen config `hybrid-rrf-k10-w1-1` + `cross_encoder`@20, 512-token chunks, `qwen2.5-3b-fp16` k_ctx=5, threshold 0.000343 (effectively inactive); gpu_seconds 7256.9. Test: required_point_recall 0.869 [0.761, 0.950], abstain F1 0.947, false_answer_rate 0.000, false_abstain_rate 0.027, 5/46 token-cap hits all abstained as `truncated`, zero non-truncation parse failures. Outputs curated verbatim into `results/`.

## Protocol revision: v13 reason-field bound correction

Review of the v12 run showed the `reason` schema field used
`{"type": ["string", "null"], "maxLength": 160}`. lm-format-enforcer
0.11.3 does not enforce `maxLength` on type-union schemas (verified by
feeding characters to `JsonSchemaParser`); the `anyOf` branch form is
enforced. All 5 v12 test cap hits and the dev cap hits (3/24 for the
3B k5 candidate) were run-on `reason` strings, so the earlier claim
that every bounded field was enforced was wrong for `reason`. Dev
evidence alone justified the fix; the v12 test read stands disclosed
(required_point_recall 0.869, abstain F1 0.947, 5/46 truncated). v13
rewrites `reason` as `anyOf` string-or-null, adds character-level
parser tests against the real `OUTPUT_SCHEMA`, and adds a
post-generation guard that counts records violating schema bounds in
metrics (the smoke kernel fails if any appear).

## Verified on Kaggle: v13 run

- Smoke kernel `production-rag-observability-smoke`: the first v13
  smoke (git SHA `9dea00bccf64a0075ba2703fed0b5a6d02f92e02`, an
  earlier unpushed revision of the same commits) failed by design
  with `schema_bound_violations {dev: 13, test: 0}`
  because the guard counted the citation-resolved stored `answer`
  (answer text plus claim texts and `[chunk_id]` markers) against the
  300-char schema bound; the raw provider `answer` was exactly at the
  cap and `reason` violations were already 0. The guard was narrowed
  to `raw_output` provider fields (stored claims still checked) and a
  regression test added; smoke v2 on git SHA
  `94ac5e4c80f5d8845b6f00c7496532db691f19e4` then completed clean.
- Full kernel `production-rag-observability-benchmark` on git SHA
  `94ac5e4c80f5d8845b6f00c7496532db691f19e4`: frozen config
  `hybrid-rrf-k10-w1-1` + `cross_encoder`@20, 512-token chunks,
  `qwen2.5-3b-fp16` k_ctx=5, threshold 0.000343 (effectively
  inactive); gpu_seconds 6700.4 on a Tesla T4. `schema_bound_violations`
  is `{dev: 0, test: 0}` (recounted from the downloaded candidate
  records with `count_schema_bound_violations`: 0 across all 8
  dev/test candidate files); token-cap hits 0 and truncation rate 0.0
  for every candidate on both splits; the frozen candidate has 0 parse
  failures. Test (46 questions: 37 answerable, 9 unanswerable):
  required_point_recall 0.869 [0.761, 0.950], abstain F1 0.947,
  false_answer_rate 0.000, false_abstain_rate 0.027; abstentions are
  model_abstain 9 and citation_validation 1. Outputs curated verbatim
  into `results/`.
