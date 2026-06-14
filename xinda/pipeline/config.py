"""PipelineConfig + variant recipes.

A `PipelineConfig` is the user-facing knob set for a single (paper, lang, variant)
run. `config_hash()` produces a 12-char identifier that's stored on
`translation_jobs.config_hash` for ablation bookkeeping (so the same paper×lang
re-run with different toggles produces a distinct job row).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from xinda.config import settings


Language = Literal["zh", "fr", "es"]


@dataclass(frozen=True)
class PipelineConfig:
    """Effective per-job configuration. All thresholds + toggles live here."""

    language: Language
    pass1_model: str
    refine_model: str | None
    variant: str = "full"

    # ─── feature toggles (drive the ablation matrix) ───
    use_glossary: bool = True
    use_context: bool = True
    use_fact_anchor: bool = True
    use_fact_verify: bool = True
    use_cross_doc: bool = True
    use_coherence: bool = True   # whole-doc discourse harmonization pass
    use_external_glossary: bool = True  # ground LLM glossary against term banks
    use_retry: bool = True

    # ─── modern API features (Qwen3.7-Max + DashScope 2026) ───
    use_cache: bool = True       # DashScope Context Cache (90% off cached tokens)
    use_batch: bool = True       # DashScope Batch API (50% off, 24h async)

    # ─── thresholds ───
    xcomet_threshold: float = field(default_factory=lambda: settings.xcomet_threshold)
    fps_threshold: float = field(default_factory=lambda: settings.fps_threshold)
    max_refine_passes: int = field(default_factory=lambda: settings.max_refine_passes)

    def config_hash(self) -> str:
        """12-char stable hash of the effective config (for DB unique key)."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=6).hexdigest()


# ─────────────────────────── variant recipes ───────────────────────────────

def _full(language: Language) -> PipelineConfig:
    return PipelineConfig(
        language=language,
        pass1_model=settings.model_first_pass,
        refine_model=settings.model_refine,  # currently same model as pass1
        variant="full",
    )


def variants_for(language: Language) -> dict[str, PipelineConfig]:
    """All evaluation matrix variants for a given target language.

    10 variants (1 full + 6 ablations + 3 same-model baselines).
    """
    full = _full(language)
    return {
        "full": full,
        # ablations (single feature disabled)
        "no_glossary":   PipelineConfig(**{**asdict(full), "use_glossary": False,   "variant": "no_glossary"}),
        "no_context":    PipelineConfig(**{**asdict(full), "use_context": False,    "variant": "no_context"}),
        "no_fact_anchor":PipelineConfig(**{**asdict(full), "use_fact_anchor": False,"variant": "no_fact_anchor"}),
        "no_fact_verify":PipelineConfig(**{**asdict(full), "use_fact_verify": False,"variant": "no_fact_verify"}),
        "no_cross_doc":  PipelineConfig(**{**asdict(full), "use_cross_doc": False,  "variant": "no_cross_doc"}),
        "no_coherence":  PipelineConfig(**{**asdict(full), "use_coherence": False,  "variant": "no_coherence"}),
        "no_external_glossary": PipelineConfig(**{**asdict(full), "use_external_glossary": False, "variant": "no_external_glossary"}),
        "no_retry":      PipelineConfig(**{**asdict(full), "use_retry": False,      "refine_model": None, "variant": "no_retry"}),
        # bare baselines (all features off, varying first-pass model)
        # Use same dated snapshot as `full` to isolate the framework's contribution
        # from base-model differences.
        "baseline_qwen_turbo": PipelineConfig(
            language=language, pass1_model="qwen-turbo", refine_model=None,
            use_glossary=False, use_context=False, use_fact_anchor=False,
            use_fact_verify=False, use_cross_doc=False, use_coherence=False,
            use_external_glossary=False, use_retry=False,
            variant="baseline_qwen_turbo",
        ),
        "baseline_qwen_plus": PipelineConfig(
            language=language, pass1_model="qwen3.5-plus-2026-04-20", refine_model=None,
            use_glossary=False, use_context=False, use_fact_anchor=False,
            use_fact_verify=False, use_cross_doc=False, use_coherence=False,
            use_external_glossary=False, use_retry=False,
            variant="baseline_qwen_plus",
        ),
        "baseline_qwen37_max": PipelineConfig(
            language=language, pass1_model="qwen3.7-max", refine_model=None,
            use_glossary=False, use_context=False, use_fact_anchor=False,
            use_fact_verify=False, use_cross_doc=False, use_coherence=False,
            use_external_glossary=False, use_retry=False,
            variant="baseline_qwen37_max",
        ),
    }
