"""ModelProvider abstract base + TranslationResult dataclass.

Key design change from v1: providers RETURN data, not mutate shared state.
The old `_parent_service` side-channel that accumulated token counts onto
the service instance is gone — each `generate()` call returns a
`TranslationResult` with its own token counts, and the caller aggregates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationResult:
    """Result of a single LLM call. No shared state."""

    text: str                            # raw response (JSON string for batch translate)
    cached_prompt_tokens: int = 0        # tokens served from prompt cache (90% off)
    fresh_prompt_tokens: int = 0         # tokens charged at full input price
    completion_tokens: int = 0
    elapsed_ms: int = 0
    model: str = ""
    batch_request_id: str | None = None  # set when call went through Batch API

    @property
    def total_prompt_tokens(self) -> int:
        return self.cached_prompt_tokens + self.fresh_prompt_tokens


class ModelProvider(ABC):
    """Base class for translation backends.

    Subclasses implement `generate` (single sync call with optional cache
    + JSON schema). Batch support is layered in a sibling module
    (`providers/qianwen_batch.py`) rather than mixed into the base contract.
    """

    name: str = "base"

    def __init__(self, model_name: str, api_key: str | None = None, **kwargs: Any):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        max_output_tokens: int | None = None,
    ) -> TranslationResult:
        """Single async LLM call. Returns a TranslationResult."""
        raise NotImplementedError

    # ────────────── default rate-limit knobs (subclasses can override) ──────────────

    @property
    def rpm(self) -> int:
        """Requests per minute."""
        return 60

    @property
    def tpm(self) -> int:
        """Tokens per minute."""
        return 100_000

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap token estimator (≈ 0.5 tok/char). Override per-model if needed."""
        return max(1, int(len(text) * 0.5))
