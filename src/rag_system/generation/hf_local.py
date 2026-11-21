"""Local Hugging Face generation provider (Qwen2.5-3B-Instruct, fp16).

Requires the ``neural`` extra and runs on Kaggle or a GPU machine; the
chat engine is injectable so tests never load a model.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from rag_system.generation.base import (
    Claim,
    GenerationRequest,
    GenerationResponse,
)
from rag_system.generation.prompts import SYSTEM_PROMPT, build_structured_prompt
from rag_system.generation.provider import ABSTAIN_ANSWER

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 768

OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 300},
        "claims": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 300},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": ["text", "citations"],
            },
        },
        "abstain": {"type": "boolean"},
        "reason": {"type": ["string", "null"], "maxLength": 160},
    },
    "required": ["answer", "claims", "abstain", "reason"],
}


class ChatEngine(Protocol):
    """Minimal text-in/text-out interface for a chat model."""

    def generate_text(
        self, prompt: str, system: str = ""
    ) -> tuple[str, int, int]:
        """Return (raw_text, prompt_tokens, completion_tokens)."""
        ...


@dataclass(frozen=True)
class StructuredOutput:
    """Parsed provider JSON contract."""

    answer: str
    claims: tuple[tuple[str, tuple[str, ...]], ...]
    abstain: bool
    reason: str | None


def _extract_json_object(text: str) -> dict[str, object]:
    """Find and parse the first JSON object in model output."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text[text.find("{") :]
    if not candidate.strip():
        raise ValueError("no JSON object in model output")
    obj, _ = json.JSONDecoder().raw_decode(candidate.strip())
    if not isinstance(obj, dict):
        raise ValueError("model output JSON is not an object")
    return obj


def parse_structured_output(text: str) -> StructuredOutput:
    """Parse the JSON contract; raises ValueError on malformed output."""
    obj = _extract_json_object(text)

    answer = obj.get("answer")
    if not isinstance(answer, str):
        raise ValueError("'answer' must be a string")

    abstain = obj.get("abstain")
    if not isinstance(abstain, bool):
        raise ValueError("'abstain' must be a boolean")

    reason = obj.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("'reason' must be a string or null")

    raw_claims = obj.get("claims", [])
    if not isinstance(raw_claims, list):
        raise ValueError("'claims' must be a list")

    claims: list[tuple[str, tuple[str, ...]]] = []
    for entry in raw_claims:
        if not isinstance(entry, dict):
            raise ValueError("claim entries must be objects")
        claim_text = entry.get("text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ValueError("claim 'text' must be a non-empty string")
        raw_citations = entry.get("citations", [])
        if not isinstance(raw_citations, list) or not all(
            isinstance(c, str) for c in raw_citations
        ):
            raise ValueError("claim 'citations' must be a list of strings")
        claims.append((claim_text, tuple(raw_citations)))

    if abstain and claims:
        raise ValueError("abstaining response must not include claims")

    return StructuredOutput(
        answer=answer,
        claims=tuple(claims),
        abstain=abstain,
        reason=reason,
    )


class TransformersChatEngine:
    """Real engine: local HF chat model, greedy constrained decoding."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = 42,
        quantization: str | None = None,
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.model_id = model_id
        self.revision = revision or "main"
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.quantization = quantization

        torch.manual_seed(seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision
        )
        model_kwargs: dict[str, object] = {}
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig  # noqa: PLC0415

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available()
                    else torch.float32
                ),
            )
        else:
            dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model_kwargs["torch_dtype"] = dtype
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto" if torch.cuda.is_available() else None,
            **model_kwargs,
        )
        if not torch.cuda.is_available() and quantization != "nf4":
            self._model = self._model.float()
        self._model.eval()
        self._generation_count = 0
        # Tokenizer data is the expensive part; the enforcer itself is rebuilt
        # per request because its token-prefix cache grows with every prompt.
        from lmformatenforcer.integrations.transformers import (  # noqa: PLC0415
            build_token_enforcer_tokenizer_data,
        )

        self._tokenizer_data = build_token_enforcer_tokenizer_data(
            self._tokenizer
        )

    def generate_text(
        self, prompt: str, system: str = ""
    ) -> tuple[str, int, int]:
        import torch  # noqa: PLC0415

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text_prompt, return_tensors="pt").to(
            self._model.device
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])
        # Fresh parser per request: the enforcer caches state per token
        # prefix, so a shared instance would retain every prompt forever.
        import gc  # noqa: PLC0415

        from lmformatenforcer import JsonSchemaParser  # noqa: PLC0415
        from lmformatenforcer.integrations.transformers import (  # noqa: PLC0415
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(OUTPUT_SCHEMA)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self._tokenizer_data, parser
        )
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        self._generation_count += 1
        if self._generation_count % 10 == 0:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = int(new_tokens.shape[0])
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw, prompt_tokens, completion_tokens


class HFLocalProvider:
    """GenerationProvider backed by a local Hugging Face chat model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = 42,
        quantization: str | None = None,
        engine: ChatEngine | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision or "main"
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.quantization = quantization
        self._engine = engine or TransformersChatEngine(
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
            seed=seed,
            quantization=quantization,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate structured JSON; parse failures become abstentions."""
        start_time = time.perf_counter()
        prompt = build_structured_prompt(
            request.query,
            request.evidence_list,
            max_tokens=request.max_context_tokens,
        )
        try:
            raw, prompt_tokens, completion_tokens = self._engine.generate_text(
                prompt, system=SYSTEM_PROMPT
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResponse(
                answer=ABSTAIN_ANSWER,
                status="abstained",
                abstain=True,
                abstain_reason=f"generation engine error: {type(exc).__name__}",
                latency_ms=latency_ms,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            parsed = parse_structured_output(raw)
        except ValueError:
            return GenerationResponse(
                answer=ABSTAIN_ANSWER,
                status="abstained",
                abstain=True,
                abstain_reason="generation parse failure",
                parse_failure=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                raw_output=raw,
            )

        if parsed.abstain:
            return GenerationResponse(
                answer=parsed.answer or ABSTAIN_ANSWER,
                status="abstained",
                abstain=True,
                abstain_reason=parsed.reason or "model abstained",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                raw_output=raw,
            )

        claims = tuple(
            Claim(text=text, citation_labels=labels)
            for text, labels in parsed.claims
        )
        return GenerationResponse(
            answer=parsed.answer,
            status="answered",
            claims=claims,
            abstain=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw_output=raw,
        )
