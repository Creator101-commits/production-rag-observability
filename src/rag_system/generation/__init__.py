"""Generation providers, protocols, prompts, and citation resolution."""

from rag_system.generation.base import (
    Claim,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
)
from rag_system.generation.hf_local import HFLocalProvider, parse_structured_output
from rag_system.generation.prompts import (
    build_context_block,
    build_labeled_context,
    build_rag_prompt,
    build_structured_prompt,
)
from rag_system.generation.provider import DeterministicFakeProvider
from rag_system.generation.resolve import resolve_generation_response

__all__ = [
    "Claim",
    "DeterministicFakeProvider",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "HFLocalProvider",
    "build_context_block",
    "build_labeled_context",
    "build_rag_prompt",
    "build_structured_prompt",
    "parse_structured_output",
    "resolve_generation_response",
]
