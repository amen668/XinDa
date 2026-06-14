"""Multi-vendor model registry for the multi-LLM × QE benchmark.

Every entry is an **OpenAI-compatible** chat endpoint, so one generic provider
(`QianwenProvider`, which is just an AsyncOpenAI client with an injectable
base_url) drives them all. To use a model, set its `api_key_env` in the
environment; models whose key is absent are skipped by the benchmark.

This is provider-agnostic on purpose — the QE benchmark compares translation
quality ACROSS **Chinese** vendors (Qwen / DeepSeek / Kimi / GLM / Doubao / ERNIE /
Yi / Hunyuan / StepFun / MiniMax), which is the point: a trustworthy automated QE
should rank them sensibly. (Foreign models are intentionally excluded.)

Endpoints are the vendors' published OpenAI-compatible base URLs; model ids and
rate caps should be confirmed against each vendor's console before a real run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from xinda.config import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str            # our registry key (also the model id sent to the API unless overridden)
    vendor: str
    base_url: str       # OpenAI-compatible endpoint
    api_key_env: str    # env var holding the API key
    model_id: str = ""  # actual API model id (defaults to `key`)
    rpm: int = 60
    tpm: int = 100_000
    notes: str = ""
    api_key: str = ""       # inline key (e.g. from settings/.env); preferred over env var
    max_retries: int = 3    # SDK 429/5xx retries; raise for flaky endpoints (coding plan)
    enable_thinking: bool | None = None  # qwen3 thinking toggle (None = model default)

    @property
    def api_model(self) -> str:
        return self.model_id or self.key

    @property
    def available(self) -> bool:
        return bool(self.api_key or os.environ.get(self.api_key_env))


_DASHSCOPE = settings.dashscope_openai_base_url

# Curated set of strong CHINESE translators. All OpenAI-compatible endpoints.
# (No foreign models — per project policy.) Verify model ids/endpoints per console.
REGISTRY: dict[str, ModelSpec] = {
    # ── 通义千问 / 阿里 DashScope (compatible-mode) ──
    "qwen-max":      ModelSpec("qwen-max", "qwen", _DASHSCOPE, "DASHSCOPE_API_KEY", rpm=120, tpm=500_000),
    "qwen-plus":     ModelSpec("qwen-plus", "qwen", _DASHSCOPE, "DASHSCOPE_API_KEY", rpm=600, tpm=500_000),
    "qwen-turbo":    ModelSpec("qwen-turbo", "qwen", _DASHSCOPE, "DASHSCOPE_API_KEY", rpm=1200, tpm=1_000_000),
    "qwen3.7-max":   ModelSpec("qwen3.7-max", "qwen", _DASHSCOPE, "DASHSCOPE_API_KEY", rpm=120, tpm=500_000,
                               notes="strongest Qwen; free tier may be exhausted (needs paid mode)"),
    # Good model on the DashScope CODING-PLAN endpoint (separate key/quota); throws
    # occasional 429 → higher max_retries. Key comes from settings (.env), not an env var.
    "qwen3.7-plus":  ModelSpec("qwen3.7-plus", "qwen", settings.dashscope_coding_base_url,
                               "DASHSCOPE_CODING_API_KEY", rpm=120, tpm=500_000,
                               api_key=settings.dashscope_coding_api_key, max_retries=8,
                               notes="coding-plan endpoint; occasional 429, retried"),
    # Same model/endpoint, thinking DISABLED — for translation (CoT just inflates
    # completion tokens + latency). model_id stays the real API name.
    "qwen3.7-plus-nothink": ModelSpec("qwen3.7-plus-nothink", "qwen",
                               settings.dashscope_coding_base_url, "DASHSCOPE_CODING_API_KEY",
                               model_id="qwen3.7-plus", rpm=120, tpm=500_000,
                               api_key=settings.dashscope_coding_api_key, max_retries=8,
                               enable_thinking=False, notes="qwen3.7-plus, thinking off"),
    # ── CODING-PLAN 跨厂商（同一 coding 端点 + DASHSCOPE_CODING_API_KEY，无需各厂商单独 key）──
    # 用于"国产多模型 × 契约即保结构"对照表；均关思考（翻译无需 CoT），MiniMax 拒收该参数故用默认。
    "qwen3-max-2026-01-23": ModelSpec("qwen3-max-2026-01-23", "qwen",
                               settings.dashscope_coding_base_url, "DASHSCOPE_CODING_API_KEY",
                               api_key=settings.dashscope_coding_api_key, rpm=120, tpm=500_000,
                               max_retries=8, enable_thinking=False, notes="coding-plan; strongest Qwen"),
    "glm-4.7": ModelSpec("glm-4.7", "zhipu", settings.dashscope_coding_base_url,
                               "DASHSCOPE_CODING_API_KEY", api_key=settings.dashscope_coding_api_key,
                               rpm=120, tpm=500_000, max_retries=8, enable_thinking=False,
                               notes="coding-plan 智谱"),
    "glm-5": ModelSpec("glm-5", "zhipu", settings.dashscope_coding_base_url,
                               "DASHSCOPE_CODING_API_KEY", api_key=settings.dashscope_coding_api_key,
                               rpm=120, tpm=500_000, max_retries=8, enable_thinking=False,
                               notes="coding-plan 智谱；用作抽检独立筛查器（异厂商于翻译用 Qwen）"),
    "kimi-k2.5": ModelSpec("kimi-k2.5", "moonshot", settings.dashscope_coding_base_url,
                               "DASHSCOPE_CODING_API_KEY", api_key=settings.dashscope_coding_api_key,
                               rpm=120, tpm=500_000, max_retries=8, enable_thinking=False,
                               notes="coding-plan 月之暗面"),
    "minimax-m2.5": ModelSpec("minimax-m2.5", "minimax", settings.dashscope_coding_base_url,
                               "DASHSCOPE_CODING_API_KEY", model_id="MiniMax-M2.5",
                               api_key=settings.dashscope_coding_api_key, rpm=120, tpm=500_000,
                               max_retries=8, notes="coding-plan; rejects enable_thinking → default"),
    # ── DeepSeek / 深度求索 ──
    "deepseek-chat":     ModelSpec("deepseek-chat", "deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", rpm=300, tpm=500_000),
    "deepseek-reasoner": ModelSpec("deepseek-reasoner", "deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", rpm=300, tpm=500_000),
    # ── Kimi / 月之暗面 Moonshot ──
    "kimi-k2": ModelSpec("kimi-k2", "moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY",
                         model_id="kimi-k2-0905-preview", rpm=200, tpm=400_000),
    # ── 智谱 GLM ──
    "glm-4.6": ModelSpec("glm-4.6", "zhipu", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", rpm=200, tpm=400_000),
    # ── 字节豆包 Doubao / 火山引擎 Ark ──
    "doubao-pro": ModelSpec("doubao-pro", "bytedance", "https://ark.cn-beijing.volces.com/api/v3", "ARK_API_KEY",
                            model_id="doubao-pro-32k", rpm=300, tpm=800_000,
                            notes="Ark model id may be a custom endpoint id"),
    # ── 百度文心 ERNIE / 千帆 v2 ──
    "ernie-4.5": ModelSpec("ernie-4.5", "baidu", "https://qianfan.baidubce.com/v2", "QIANFAN_API_KEY",
                           model_id="ernie-4.5-turbo-128k", rpm=120, tpm=400_000),
    # ── 零一万物 Yi ──
    "yi-lightning": ModelSpec("yi-lightning", "01ai", "https://api.lingyiwanwu.com/v1", "YI_API_KEY", rpm=120, tpm=400_000),
    # ── 腾讯混元 Hunyuan ──
    "hunyuan": ModelSpec("hunyuan", "tencent", "https://api.hunyuan.cloud.tencent.com/v1", "HUNYUAN_API_KEY",
                         model_id="hunyuan-turbos-latest", rpm=120, tpm=400_000),
    # ── 阶跃星辰 StepFun ──
    "step-2": ModelSpec("step-2", "stepfun", "https://api.stepfun.com/v1", "STEP_API_KEY",
                        model_id="step-2-16k", rpm=120, tpm=400_000),
    # ── MiniMax ──
    "minimax": ModelSpec("minimax", "minimax", "https://api.minimax.chat/v1", "MINIMAX_API_KEY",
                         model_id="abab6.5s-chat", rpm=120, tpm=400_000,
                         notes="verify OpenAI-compat path/model id"),
}

# Default benchmark set: Chinese models spanning quality tiers (all free/cheap to obtain).
DEFAULT_BENCHMARK = ["qwen-turbo", "qwen-plus", "qwen-max", "deepseek-chat", "glm-4.6", "kimi-k2"]


def get_spec(key: str) -> ModelSpec | None:
    return REGISTRY.get(key)


def available_models() -> list[str]:
    """Registry keys whose API key is present in the environment."""
    return [k for k, s in REGISTRY.items() if s.available]
