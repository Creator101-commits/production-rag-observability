"""Versioned evaluation question suite across answerable and adversarial categories."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuestion:
    """Benchmark question for RAG evaluation."""

    question_id: str
    question: str
    answerable: bool
    expected_doc_ids: list[str] = field(default_factory=list)
    category: str = "direct_lookup"  # "direct_lookup", "synthesis", "adversarial", "unanswerable"
    split: str = "test"  # "dev", "test"


def load_evaluation_dataset() -> list[EvalQuestion]:
    """Load benchmark questions covering direct lookup, synthesis, and unanswerable prompts."""
    return [
        EvalQuestion(
            question_id="q_01",
            question="What are the seven characteristics of trustworthy AI systems in NIST AI RMF?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_1_0"],
            category="direct_lookup",
            split="test",
        ),
        EvalQuestion(
            question_id="q_02",
            question="What is the primary role of the GOVERN core function?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_1_0"],
            category="direct_lookup",
            split="test",
        ),
        EvalQuestion(
            question_id="q_03",
            question="How does the MAP function frame and categorize AI system risks?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_1_0"],
            category="direct_lookup",
            split="test",
        ),
        EvalQuestion(
            question_id="q_04",
            question="What activities are included in the MEASURE core function for TEVV?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_1_0", "nist_ai_rmf_playbook"],
            category="synthesis",
            split="test",
        ),
        EvalQuestion(
            question_id="q_05",
            question="What are the unique risks introduced by generative AI foundation models?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_playbook"],
            category="direct_lookup",
            split="test",
        ),
        EvalQuestion(
            question_id="q_06",
            question="How should independent teams conduct TEVV to prevent confirmation bias?",
            answerable=True,
            expected_doc_ids=["nist_ai_rmf_playbook"],
            category="direct_lookup",
            split="dev",
        ),
        EvalQuestion(
            question_id="q_07",
            question="What is the stock price target of NVIDIA according to NIST AI RMF?",
            answerable=False,
            expected_doc_ids=[],
            category="unanswerable",
            split="test",
        ),
        EvalQuestion(
            question_id="q_08",
            question="Explain the warp-drive propulsion mechanics detailed in NIST AI 100-1.",
            answerable=False,
            expected_doc_ids=[],
            category="adversarial",
            split="test",
        ),
        EvalQuestion(
            question_id="q_09",
            question="What are the official IRS tax bracket rates for 2026 mentioned in the text?",
            answerable=False,
            expected_doc_ids=[],
            category="unanswerable",
            split="dev",
        ),
    ]
