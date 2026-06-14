"""Centralized settings via Pydantic BaseSettings.

All secrets and external paths are read from environment variables (or a
`.env` file at the repo root). No hardcoded API keys.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ────────── database ──────────
    database_url: str = (
        "postgresql+asyncpg://postgres:root@localhost:5432/arxiv_translation_hub"
    )

    # ────────── DashScope / Qwen API ──────────
    dashscope_api_key: str = ""
    # OpenAI-compatible endpoint (international or Singapore deployment)
    dashscope_openai_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    # DashScope "coding plan" endpoint + key (separate subscription/quota). The good
    # model (qwen3.7-plus) runs here; the cheap model (qwen-plus) stays on the regular
    # endpoint above. Both keys read from .env (never committed).
    dashscope_coding_base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    dashscope_coding_api_key: str = ""
    # Anthropic-compatible endpoint
    dashscope_anthropic_base_url: str = (
        "https://dashscope-intl.aliyuncs.com/api/v2/apps"
    )

    # ────────── default model assignments (from plan v3) ──────────
    # Single-model policy: all stages use one model for reproducibility.
    # NOTE: `qwen3.5-plus` (a prior quota workaround) reliably HANGS on the
    # detailed structured-extraction prompts (FactExtract/Glossary/Verify) on the
    # DashScope CN endpoint — the constrained decoder trickles tokens forever and
    # the read-timeout never fires. `qwen3.7-max` handles them in ~40s. So the
    # default is qwen3.7-max; do not downgrade structured-extraction stages to
    # qwen3.5-plus without re-checking this hang.
    # thinking OFF: on full-text translation, enable_thinking inflated completion
    # tokens 3× (26.5k vs 8.9k/paper) and cost 2.2×, wall-time 3×, for ZERO gain in
    # structure (PPA/MFR/coverage all 100 regardless). Verified 2026-06-09 on PMC13150831.
    _DEFAULT_MODEL: str = "qwen3.7-plus-nothink"  # DashScope coding-plan endpoint (deepseek-v4 dropped 2026-06).
    # Model survey (CN endpoint, 2026-06): qwen3.5-plus HANGS on structured-output calls
    # (json_schema + real prompt) — usable only for plain text; qwen3.7-max handles them
    # but its free tier is exhausted (needs paid mode); qwen-plus / qwen-max / qwen-flash
    # all handle structured output fast with quota. The 300s hard-timeout in
    # qianwen.generate() prevents a deadlock regardless of model.

    model_first_pass: str = _DEFAULT_MODEL
    model_refine: str = _DEFAULT_MODEL
    model_fact_extract: str = _DEFAULT_MODEL
    model_fact_verify: str = _DEFAULT_MODEL
    model_judge_rubric: str = _DEFAULT_MODEL
    model_judge_geval: str = _DEFAULT_MODEL
    model_qa_reader: str = _DEFAULT_MODEL
    model_qa_judge: str = _DEFAULT_MODEL
    model_cross_doc: str = _DEFAULT_MODEL
    model_glossary: str = _DEFAULT_MODEL

    # ────────── pipeline thresholds ──────────
    # xCOMET-XL scores en→zh scientific text on a much lower/wider scale than the
    # 0.75 originally assumed (observed median ≈0.47 on good translations that
    # judges rate 4.85/5). An absolute 0.75 gate flags ~80% of units — chasing a
    # miscalibrated metric. 0.25 targets only the genuinely-broken tail (~p10);
    # the Refine gate then leans on the (fixed) FPS signal as the primary driver.
    xcomet_threshold: float = 0.25
    fps_threshold: float = 0.95
    comet_threshold: float = 0.70  # legacy auxiliary
    max_refine_passes: int = 2

    # ────────── external binaries ──────────
    latexmlc_path: str = "latexmlc"
    latexmlpost_path: str = "latexmlpost"
    magick_path: str = "magick"

    # ────────── workspace ──────────
    workspace_dir: Path = Path("workspace")
    input_dir: Path = Path("static/input")
    downloads_dir: Path = Path("static/downloads")
    # external term-bank files (Jiqizhixin All.md, Microsoft *.tbx) for glossary grounding
    glossary_data_dir: Path = Path("data/glossaries")

    # ────────── runtime ──────────
    max_concurrency: int = 10  # qwen3.7-max RPM=120 has ample headroom; 4→10 ~halves wall-time
    latexml_timeout_sec: int = 1800

    @property
    def has_dashscope_credentials(self) -> bool:
        return bool(self.dashscope_api_key)


settings = Settings()
