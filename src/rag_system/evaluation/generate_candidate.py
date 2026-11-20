"""Stage-B worker: generate records for one provider spec in isolation.

Run as ``python -m rag_system.evaluation.generate_candidate`` from the
benchmark orchestrator so each generator candidate gets its own process
and can be killed/reaped without taking down the parent. Reads a
serialized contexts file (approved evidence per question) and a provider
spec, writes per-(k_ctx, split) record JSONL plus an OTel JSONL trace
file for the generation spans.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from rag_system.citations.validator import CitationValidator
from rag_system.generation.base import GenerationRequest
from rag_system.generation.resolve import resolve_generation_response
from rag_system.observability.tracing import JsonlSpanExporter, OTelTracer
from rag_system.retrieval.schema import Evidence


def _build_provider(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind", "hf")
    if kind == "fake":
        from rag_system.generation.provider import (  # noqa: PLC0415
            DeterministicFakeProvider,
        )

        return DeterministicFakeProvider()
    if kind == "hf":
        from rag_system.generation.hf_local import HFLocalProvider  # noqa: PLC0415

        return HFLocalProvider(
            model_id=spec["model_id"],
            revision=spec.get("revision"),
            max_new_tokens=int(spec.get("max_new_tokens", 384)),
            seed=int(spec.get("seed", 42)),
            quantization=spec.get("quantization"),
        )
    raise ValueError(f"unknown provider kind: {kind}")


def _evidence_from_dict(raw: dict[str, Any], rank: int) -> Evidence:
    return Evidence(
        chunk_id=raw["chunk_id"],
        document_id=raw.get("document_id", ""),
        document_title=raw.get("document_title", ""),
        page_start=int(raw.get("page_start", 0)),
        page_end=int(raw.get("page_end", 0)),
        text=raw["text"],
        score=float(raw.get("score", 0.0)),
        rank=rank,
        retriever_name=raw.get("retriever_name", "hybrid"),
    )


def _status_value(key: str) -> str:
    """Best-effort /proc/self/status or MemAvailable read (Linux only)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "n/a"


def _mem_available_mb() -> str:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "n/a"


def _log_resources(tag: str, done: int, total: int) -> None:
    cuda = "n/a"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            cuda = f"{torch.cuda.max_memory_allocated() / 2**20:.0f}"
    except Exception:
        pass
    print(
        f"[resources] {tag}: {done}/{total} "
        f"rss={_status_value('VmRSS')} mem_available={_mem_available_mb()} "
        f"cuda_max_mb={cuda}",
        flush=True,
    )


def _generate_records(
    provider: Any,
    questions: list[dict[str, Any]],
    k_ctx: int,
    tracer: OTelTracer,
    *,
    temperature: float = 0.0,
    max_context_tokens: int = 2048,
) -> Iterator[dict[str, Any]]:
    """Run generation for every serialized question at one context size."""
    for q in questions:
        evidence = [
            _evidence_from_dict(e, i + 1)
            for i, e in enumerate(q["evidence"][:k_ctx])
        ]
        record: dict[str, Any] = {
            "question_id": q["question_id"],
            "split": q["split"],
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "answerable": bool(q.get("answerable", True)),
            "question": q["question"],
            "top_score": float(q.get("top_score", 0.0)),
            "retrieved_chunk_ids": [e.chunk_id for e in evidence],
        }
        gen_req = GenerationRequest(
            query=q["question"],
            evidence_list=evidence,
            temperature=temperature,
            # Pipeline threshold is authoritative and applied in stage C.
            evidence_score_threshold=-1e9,
            max_context_tokens=max_context_tokens,
        )
        with tracer.span("generation") as gen_span:
            raw_resp = provider.generate(gen_req)
            gen_resp, resolve_warnings = resolve_generation_response(
                raw_resp, evidence
            )
            raw_label_total = sum(
                len(c.citation_labels) for c in raw_resp.claims
            )
            resolved_label_total = sum(
                len(c.chunk_ids) for c in gen_resp.claims
            )
            gen_resp = CitationValidator.enforce(
                gen_resp, evidence, minimum_citations=1
            )
            gen_span.set_attribute("status", gen_resp.status)
            gen_span.set_attribute("prompt_tokens", gen_resp.prompt_tokens)
            gen_span.set_attribute(
                "completion_tokens", gen_resp.completion_tokens
            )
        reason = (gen_resp.abstain_reason or "").lower()
        if gen_resp.status == "abstained":
            if gen_resp.parse_failure:
                abstain_source = "parse_failure"
            elif "citation" in reason or "valid" in reason:
                abstain_source = "citation_validation"
            else:
                abstain_source = "model_abstain"
        else:
            abstain_source = ""
        record.update(
            status=gen_resp.status,
            abstain_source=abstain_source,
            abstain_reason=gen_resp.abstain_reason,
            answer=gen_resp.answer,
            raw_output=raw_resp.raw_output,
            claims=[
                {"text": c.text, "chunk_ids": list(c.chunk_ids)}
                for c in gen_resp.claims
            ],
            citations=list(gen_resp.citations),
            prompt_tokens=gen_resp.prompt_tokens,
            completion_tokens=gen_resp.completion_tokens,
            parse_failure=gen_resp.parse_failure,
            latency_ms=gen_resp.latency_ms,
            citation_label_validity=(
                resolved_label_total / raw_label_total
                if raw_label_total
                else None
            ),
            resolve_warnings=list(resolve_warnings),
        )
        yield record


def run_candidate(
    contexts_path: Path,
    spec_path: Path,
    out_dir: Path,
    *,
    k_ctx: int | None = None,
) -> dict[str, Any]:
    """Generate records for one provider spec; returns a small summary."""
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    name = spec["name"]
    k_ctxs = (
        [k_ctx]
        if k_ctx is not None
        else [int(k) for k in contexts.get("k_ctxs", [3, 5])]
    )
    questions = contexts["questions"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = _build_provider(spec)
    tracer = OTelTracer(
        service_name=f"rag_system-candidate-{name}",
        span_exporters=[JsonlSpanExporter(out_dir / f"traces_{name}.jsonl")],
        enable_otlp=False,
    )
    summary: dict[str, Any] = {"provider": name, "files": []}
    try:
        for k in k_ctxs:
            handles: dict[str, TextIO] = {}
            try:
                for done, record in enumerate(
                    _generate_records(
                        provider,
                        questions,
                        k,
                        tracer,
                        temperature=float(contexts.get("temperature", 0.0)),
                        max_context_tokens=int(
                            contexts.get("max_context_tokens", 2048)
                        ),
                    ),
                    start=1,
                ):
                    split = str(record["split"])
                    if split not in handles:
                        path = out_dir / f"{name}__k{k}__{split}.jsonl"
                        handles[split] = path.open("w", encoding="utf-8")
                        summary["files"].append(path.name)
                    handle = handles[split]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    if done % 5 == 0:
                        _log_resources(f"candidate-{name}-k{k}", done, len(questions))
            finally:
                for handle in handles.values():
                    handle.close()
    finally:
        tracer.force_flush()
        tracer.shutdown()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--k-ctx", type=int, default=None)
    args = parser.parse_args(argv)
    summary = run_candidate(
        args.contexts, args.spec, args.out_dir, k_ctx=args.k_ctx
    )
    print(f"candidate {summary['provider']} wrote {len(summary['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
