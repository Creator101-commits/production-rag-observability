"""Generation providers, protocols, and prompts."""

from rag_system.generation.base import GenerationProvider, GenerationRequest, GenerationResponse
from rag_system.generation.prompts import build_context_block, build_rag_prompt
from rag_system.generation.provider import DeterministicFakeProvider

__all__ = [
    "DeterministicFakeProvider",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "build_context_block",
    "build_rag_prompt",
]
