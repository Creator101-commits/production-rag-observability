"""Evaluation dataset, metrics, benchmark runner, and QA benchmark."""

from rag_system.evaluation.answers import (
    abstention_metrics,
    normalize_for_matching,
    required_point_recall,
)
from rag_system.evaluation.benchmark import (
    Components,
    FrozenConfig,
    RetrievalSpec,
    run_full_benchmark,
)
from rag_system.evaluation.dataset import (
    EvalQuestion,
    EvidenceRef,
    load_eval_questions,
    split_questions,
)
from rag_system.evaluation.metrics import OverallRAGMetrics
from rag_system.evaluation.nli import NLIScorer, claim_supported
from rag_system.evaluation.relevance import (
    RetrievalScores,
    bootstrap_ci,
    chunk_is_relevant,
    retrieval_scores,
)
from rag_system.evaluation.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "Components",
    "EvalQuestion",
    "EvidenceRef",
    "FrozenConfig",
    "NLIScorer",
    "OverallRAGMetrics",
    "RetrievalScores",
    "RetrievalSpec",
    "abstention_metrics",
    "bootstrap_ci",
    "chunk_is_relevant",
    "claim_supported",
    "load_eval_questions",
    "normalize_for_matching",
    "required_point_recall",
    "retrieval_scores",
    "run_full_benchmark",
    "split_questions",
]
