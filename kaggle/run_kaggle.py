"""Kaggle entrypoint for the NIST RAG observability benchmark.

Installs ``rag_system`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, downloads and verifies the pinned NIST PDFs,
runs the dev-tuned frozen-config benchmark with real models, and writes
``metrics.json``, ``manifest.json``, per-question JSONL, ``traces.jsonl``,
and ``report.md`` under ``/kaggle/working/benchmark/``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
GENERATOR_3B = "Qwen/Qwen2.5-3B-Instruct"
GENERATOR_7B = "Qwen/Qwen2.5-7B-Instruct"
K_CTX_CANDIDATES = (3, 5)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    """Locate the project root (pyproject.toml + src/rag_system)."""
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "rag_system").is_dir():
            return candidate.parent
    return None


def _install_package() -> Path:
    """Install ``rag_system`` from the preserved source-dataset tree."""
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/rag_system)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            # Pinned pair: lm-format-enforcer's transformers integration
            # needs transformers 4.x (PreTrainedTokenizerBase import path);
            # transformers 5.x breaks constrained decoding.
            "sentence-transformers==3.4.1",
            "transformers==4.46.3",
            "accelerate>=0.34",
            "huggingface-hub>=0.26",
            "lm-format-enforcer>=0.11",
            "bitsandbytes>=0.43",
        ]
    )
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _find_eval_file(proj: Path) -> Path:
    candidate = proj / "data" / "eval" / "nist_qa_v1.jsonl"
    if candidate.is_file():
        return candidate
    matches = list(proj.rglob("nist_qa_v1.jsonl"))
    if matches:
        return matches[0]
    raise FileNotFoundError("nist_qa_v1.jsonl not found in source dataset")


def _revision(model_id: str) -> str:
    """Resolve and pin the current Hub commit SHA for a model."""
    from huggingface_hub import model_info  # noqa: PLC0415

    return model_info(model_id).sha


def main() -> None:
    proj = _install_package()
    sha = _git_sha(proj)
    print(f"git_sha={sha}", flush=True)

    from rag_system.evaluation.benchmark import (  # noqa: PLC0415
        BGE_QUERY_PREFIX,
        Components,
        run_full_benchmark,
    )
    from rag_system.evaluation.dataset import load_eval_questions  # noqa: PLC0415
    from rag_system.evaluation.nli import HFNLICrossEncoder  # noqa: PLC0415
    from rag_system.ingestion.pdf import (  # noqa: PLC0415
        NIST_CORPUS_SPECS,
        ingest_corpus,
    )
    from rag_system.ingestion.tokens import HFTokenizer  # noqa: PLC0415
    from rag_system.reranking.scorer import (  # noqa: PLC0415
        SentenceTransformerPairScorer,
    )
    from rag_system.retrieval.sentence_transformer import (  # noqa: PLC0415
        SentenceTransformerEncoder,
    )

    raw_dir = WORK / "raw"
    docs, corpus_manifest = ingest_corpus(NIST_CORPUS_SPECS, raw_dir, download=True)
    print(
        f"ingested {len(docs)} docs, "
        f"{sum(len(d.pages) for d in docs)} pages",
        flush=True,
    )

    eval_path = _find_eval_file(proj)
    questions = load_eval_questions(eval_path)
    print(f"loaded {len(questions)} eval questions", flush=True)

    dense_rev = _revision(DENSE_MODEL)
    rerank_rev = _revision(RERANKER_MODEL)
    nli_rev = _revision(NLI_MODEL)
    gen3b_rev = _revision(GENERATOR_3B)
    gen7b_rev = _revision(GENERATOR_7B)
    print(
        f"revisions: dense={dense_rev[:12]} rerank={rerank_rev[:12]} "
        f"nli={nli_rev[:12]} gen3b={gen3b_rev[:12]} gen7b={gen7b_rev[:12]}",
        flush=True,
    )

    smoke = bool(os.environ.get("RAG_SMOKE"))
    provider_specs = {
        "qwen2.5-3b-fp16": {
            "kind": "hf",
            "model_id": GENERATOR_3B,
            "revision": gen3b_rev,
            "max_new_tokens": 768,
            "seed": 42,
            "quantization": None,
        },
        "qwen2.5-7b-nf4": {
            "kind": "hf",
            "model_id": GENERATOR_7B,
            "revision": gen7b_rev,
            "max_new_tokens": 768,
            "seed": 42,
            "quantization": "nf4",
        },
    }
    k_ctx_candidates: tuple[int, ...] = K_CTX_CANDIDATES
    if smoke:
        # Leak-proof smoke: only the 3B candidate, k_ctx=5, 15 dev questions.
        provider_specs = {"qwen2.5-3b-fp16": provider_specs["qwen2.5-3b-fp16"]}
        k_ctx_candidates = (5,)
        questions = [q for q in questions if q.split == "dev"][:15]
        print(f"SMOKE MODE: {len(questions)} dev questions, 3B k_ctx=5", flush=True)

    comps = Components(
        tokenizer=HFTokenizer(DENSE_MODEL, revision=dense_rev),
        encoder=SentenceTransformerEncoder(DENSE_MODEL, revision=dense_rev),
        encoder_factory=lambda: SentenceTransformerEncoder(
            DENSE_MODEL, revision=dense_rev
        ),
        query_prefix=BGE_QUERY_PREFIX,
        pair_scorer=SentenceTransformerPairScorer(
            RERANKER_MODEL, revision=rerank_rev
        ),
        pair_scorer_factory=lambda: SentenceTransformerPairScorer(
            RERANKER_MODEL, revision=rerank_rev
        ),
        nli_scorer=HFNLICrossEncoder(NLI_MODEL, revision=nli_rev),
        provider_specs=provider_specs,
        provider_name="hf-local",
        provider_model_id=GENERATOR_3B,
        dense_model_id=DENSE_MODEL,
        dense_revision=dense_rev,
        reranker_model_id=RERANKER_MODEL,
        reranker_revision=rerank_rev,
        nli_model_id=NLI_MODEL,
        nli_revision=nli_rev,
        k_ctx_candidates=k_ctx_candidates,
        seed=42,
    )

    bench_dir = BENCH / "smoke" if smoke else BENCH
    bench_dir.mkdir(parents=True, exist_ok=True)
    result = run_full_benchmark(
        docs,
        corpus_manifest,
        questions,
        comps,
        bench_dir,
        eval_path=eval_path,
        git_sha=sha,
    )

    violations = result["metrics"].get("schema_bound_violations", {})
    if smoke and (violations.get("dev", 0) + violations.get("test", 0)) > 0:
        raise SystemExit(
            f"schema bound violations detected: {violations}; "
            "constrained decoding is not enforcing OUTPUT_SCHEMA"
        )
    if violations:
        print(f"schema_bound_violations: {violations}", flush=True)

    frozen = result["manifest"]["frozen_config"]
    e2e = result["metrics"]["e2e_test"]
    print("=== FROZEN CONFIG ===", flush=True)
    print(json.dumps(frozen, indent=2), flush=True)
    print("=== TEST E2E HEADLINE ===", flush=True)
    for key in (
        "required_point_recall",
        "all_points_covered",
        "abstain_precision",
        "abstain_recall",
        "abstain_f1",
        "false_abstain_rate",
        "citation_label_validity",
        "citation_page_precision",
        "citation_hit_rate",
        "nli_claim_support_rate",
    ):
        print(f"{key}: {e2e.get(key)}", flush=True)
    print(f"gpu_seconds: {result['metrics']['gpu_seconds']:.2f}", flush=True)

    # Sanity: package versions for the record.
    for pkg in ("rag-system", "sentence-transformers", "torch", "transformers"):
        try:
            print(f"{pkg}=={metadata.version(pkg)}", flush=True)
        except metadata.PackageNotFoundError:
            print(f"{pkg}: not installed", flush=True)


if __name__ == "__main__":
    main()
