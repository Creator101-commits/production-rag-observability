# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Enforced fail-closed citation ID checks, fixed approximate-token chunk limits, limited trace retention, excluded raw queries from validation-span attributes, and removed boilerplate-only evidence matches.

## Verified locally

- 35 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

The service uses hand-written paraphrases, hashed lexical vectors, lexical reranking, and deterministic generation. It does not ingest official PDFs or validate semantic entailment.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.
