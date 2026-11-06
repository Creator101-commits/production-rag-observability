# Production RAG Observability

A citation-grounded question-answering service over versioned NIST AI Risk Management Framework documents with retrieval, abstention, and stage-level observability.

## Build commands

```bash
# Planned after Milestone 1; not runnable yet.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12; Docker is part of the planned delivery workflow.
- CI must use the deterministic fake generation provider; hosted providers and telemetry are optional integrations.

## Key conventions

- Every chunk keeps source, page, section, and corpus-version provenance; generation may use only approved retrieved evidence.
- Code owns citation validation and abstention; unknown citation IDs are rejected.
- Telemetry is redacted by default. Never log raw document content, secrets, or sensitive queries in shared environments.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
