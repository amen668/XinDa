"""Token → cost (CNY) accounting — the quantitative basis for the cost-reduction
claim (low-cost full-text multilingual publishing vs. human / commercial MT).

Prices are DashScope **list prices in CNY per 1,000,000 tokens**, keyed by the
model *base alias* (dated snapshots inherit their base tier). Cached input tokens
are billed at the DashScope Context-Cache rate (a fraction of fresh input).

NOTE: list prices change — VERIFY against the current Model Studio price page
before quoting numbers in the paper. Source:
https://help.aliyun.com/zh/model-studio/models
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float         # CNY per 1M fresh input tokens
    cached_input: float  # CNY per 1M cached input tokens (Context Cache rate)
    output: float        # CNY per 1M output tokens


# CNY per 1M tokens. plus-tier ≈ (0.8 / 0.32 / 2.0), max-tier ≈ (2.4 / 0.96 / 9.6).
PRICE_TABLE: dict[str, Price] = {
    "qwen-turbo":   Price(0.3, 0.12, 0.6),
    "qwen-plus":    Price(0.8, 0.32, 2.0),
    "qwen-max":     Price(2.4, 0.96, 9.6),
    "qwen3-plus":   Price(0.8, 0.32, 2.0),
    "qwen3.5-plus": Price(0.8, 0.32, 2.0),
    "qwen3.6-plus": Price(0.8, 0.32, 2.0),
    # qwen3.7-plus is a HIGHER tier than qwen-plus: 百炼 ~¥3 in / ¥6 out (cache≈1/10);
    # VERIFY on console. NB: we run it on a CODING-PLAN subscription → this per-token
    # price is a list-price PROXY for reporting, NOT the actual (flat) cost paid.
    "qwen3.7-plus": Price(3.0, 0.3, 6.0),
    "qwen3.7-plus-nothink": Price(3.0, 0.3, 6.0),  # same model, thinking off
    "qwen3-max":    Price(2.4, 0.96, 9.6),
    "qwen3.7-max":  Price(2.4, 0.96, 9.6),
    # Non-Qwen DOMESTIC models (the default model + the multi-vendor comparison set).
    # Values are the vendors' published list prices (CNY/1M tokens) — VERIFY against the
    # current price page before quoting in the paper; DashScope-hosted rates may differ.
    # NO foreign models (GPT-4o etc.) by policy — cost is compared only vs human.
    "deepseek-v4-flash": Price(1.0, 0.1, 2.0),  # default; DashScope/百炼 confirmed ¥1 in / ¥2 out (cache≈1/10)
    "deepseek-v3.2": Price(2.0, 0.2, 3.0),   # retired 2026-06 (kept for old result rows)
    "deepseek-v3":   Price(2.0, 0.5, 8.0),
    "deepseek-r1":   Price(4.0, 1.0, 16.0),
    "kimi-k2":       Price(4.0, 1.0, 16.0),  # Moonshot
    "glm-4.6":       Price(2.0, 0.4, 8.0),   # Zhipu
    "doubao-pro":    Price(0.8, 0.16, 2.0),  # ByteDance
}
_DEFAULT_PRICE = Price(0.8, 0.32, 2.0)  # plus-tier fallback for unknown models


def base_alias(model_name: str) -> str:
    """Strip a dated snapshot suffix (e.g. '-2026-04-20') to the base alias."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_name or "")


def price_for(model_name: str) -> Price:
    return PRICE_TABLE.get(base_alias(model_name), _DEFAULT_PRICE)


def cost_cny(
    *,
    fresh_prompt_tok: int,
    cached_prompt_tok: int,
    completion_tok: int,
    model_name: str,
) -> float:
    """CNY cost of one job's token usage under `model_name`'s list price."""
    p = price_for(model_name)
    return (
        (fresh_prompt_tok or 0) / 1_000_000 * p.input
        + (cached_prompt_tok or 0) / 1_000_000 * p.cached_input
        + (completion_tok or 0) / 1_000_000 * p.output
    )
