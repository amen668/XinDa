"""Model provider factory.

Bare model names (e.g. the per-stage `_DEFAULT_MODEL`) route to DashScope/Qwen.
Names present in the multi-vendor `registry` route to their vendor's OpenAI-
compatible endpoint (Qwen/DeepSeek/Kimi/GLM/Doubao/ERNIE/Yi/Hunyuan/StepFun/MiniMax —
all Chinese vendors) using that vendor's API key env var. This is what lets the
multi-LLM × QE benchmark span vendors.
"""

from __future__ import annotations

import os

from xinda.config import settings
from xinda.providers import registry
from xinda.providers.base import ModelProvider
from xinda.providers.qianwen import QianwenProvider


def create_provider(model_name: str, api_key: str | None = None) -> ModelProvider:
    """Construct a provider by model name (registry-aware, OpenAI-compatible)."""
    spec = registry.get_spec(model_name)
    if spec is None:
        # default path: bare name → DashScope/Qwen (uses settings.dashscope_api_key)
        return QianwenProvider(model_name=model_name, api_key=api_key)

    # Precedence: explicit arg → spec inline key (settings/.env) → env var → qwen fallback.
    key = api_key or spec.api_key or os.environ.get(spec.api_key_env)
    if not key and spec.vendor == "qwen":
        key = settings.dashscope_api_key
    if not key:
        raise RuntimeError(
            f"No API key for '{model_name}' ({spec.vendor}); set {spec.api_key_env}."
        )
    return QianwenProvider(
        model_name=spec.api_model,
        api_key=key,
        base_url=spec.base_url,
        rpm=spec.rpm,
        tpm=spec.tpm,
        max_retries=spec.max_retries,
        enable_thinking=spec.enable_thinking,
    )
