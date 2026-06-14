"""Qianwen / Qwen3.7-Max provider via DashScope's OpenAI-compatible endpoint.

We use the `openai` SDK pointed at DashScope's `compatible-mode/v1`. This
gives us:
- structured outputs via `response_format={"type":"json_schema",...}`
- prompt-cache reporting via `usage.prompt_tokens_details.cached_tokens`
  (DashScope reports cached tokens in the same field shape as OpenAI)
- a future migration path to other OpenAI-compatible providers without
  changing call sites.

For M2 this is sync (one paragraph batch at a time). Batch API integration
lives in `providers/qianwen_batch.py` (M9).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from xinda.config import settings
from xinda.providers.base import ModelProvider, TranslationResult

# Hard ceiling for a single LLM request (whole call, not per-read). Generous:
# qwen3.7-max structured extraction ~40s; large translation batches more. A call
# exceeding this is treated as hung and raised so the stage can recover/retry.
_REQUEST_HARD_TIMEOUT_S = 300.0


class QianwenProvider(ModelProvider):
    name = "qianwen"

    # rough RPM/TPM caps from DashScope (model-tier-dependent; tuned conservatively)
    # Dated snapshots inherit limits from their base model alias.
    _RPM_TABLE = {
        "qwen-turbo": 1200,
        "qwen-plus": 600,
        "qwen-max": 120,
        "qwen3-plus": 600,
        "qwen3.5-plus": 600,
        "qwen3.6-plus": 600,
        "qwen3.7-plus": 600,
        "qwen3-max": 120,
        "qwen3.7-max": 120,
    }
    _TPM_TABLE = {
        "qwen-turbo": 1_000_000,
        "qwen-plus": 500_000,
        "qwen-max": 200_000,
        "qwen3-plus": 500_000,
        "qwen3.5-plus": 500_000,
        "qwen3.6-plus": 500_000,
        "qwen3.7-plus": 500_000,
        "qwen3-max": 200_000,
        "qwen3.7-max": 500_000,
    }

    @staticmethod
    def _resolve_base_alias(model_name: str) -> str:
        """Strip dated snapshot suffix (e.g. '-2026-04-20') to look up rate caps."""
        # Find the model family by stripping trailing -YYYY-MM-DD
        import re
        return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_name)

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        rpm: int | None = None,
        tpm: int | None = None,
        max_retries: int = 3,
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        # OpenAI-compatible: base_url defaults to DashScope but can point at any
        # vendor (GPT/Claude/Gemini/DeepSeek/…) via the model registry.
        self._base_url = base_url or settings.dashscope_openai_base_url
        self._rpm_override = rpm
        self._tpm_override = tpm
        # Qwen3 "thinking" toggle (DashScope `enable_thinking`). None = model default.
        # Set False to suppress chain-of-thought for tasks (e.g. translation) where the
        # hidden reasoning just inflates completion tokens (cost) and latency.
        self._enable_thinking = enable_thinking
        effective_key = api_key or settings.dashscope_api_key
        if not effective_key:
            raise RuntimeError(
                f"API key missing for model '{model_name}'. Set the provider's key env var."
            )
        self._client = AsyncOpenAI(
            api_key=effective_key,
            base_url=self._base_url,
            # Without an explicit timeout the SDK can hang indefinitely on a
            # stalled connection, deadlocking any asyncio.gather over many calls
            # (observed: a whole RCS batch frozen at 0% with the process idle).
            timeout=180.0,
            # SDK auto-retries 429/5xx with exponential backoff + Retry-After. The
            # DashScope coding-plan endpoint throws occasional 429s, so its provider is
            # constructed with a higher max_retries (see registry/factory).
            max_retries=max_retries,
        )

    @property
    def rpm(self) -> int:
        if self._rpm_override is not None:
            return self._rpm_override
        return self._RPM_TABLE.get(
            self._resolve_base_alias(self.model_name), 60
        )

    @property
    def tpm(self) -> int:
        if self._tpm_override is not None:
            return self._tpm_override
        return self._TPM_TABLE.get(
            self._resolve_base_alias(self.model_name), 100_000
        )

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> TranslationResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        if self._enable_thinking is not None:
            # DashScope-specific knob; passed through the OpenAI client's extra_body.
            kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}

        # Hard total-timeout: the OpenAI client's `timeout=` is a per-read deadline,
        # so a constrained decoder that trickles a few bytes at a time (observed with
        # qwen3.5-plus on the CN endpoint) never trips it and the request hangs
        # forever. asyncio.wait_for bounds the WHOLE call so the stage can recover.
        t0 = time.monotonic()
        resp: ChatCompletion = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs),
            timeout=_REQUEST_HARD_TIMEOUT_S,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        cached = 0
        prompt_tok = 0
        completion = 0
        if usage is not None:
            prompt_tok = usage.prompt_tokens or 0
            completion = usage.completion_tokens or 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
        fresh = max(0, prompt_tok - cached)

        return TranslationResult(
            text=text,
            cached_prompt_tokens=cached,
            fresh_prompt_tokens=fresh,
            completion_tokens=completion,
            elapsed_ms=elapsed_ms,
            model=self.model_name,
        )
