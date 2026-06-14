"""FactExtract stage (C1 core): identify verifiable claims per translation_unit.

For each translation_unit's `src_plain`, call qwen3.7-max (or the model
configured in `settings.model_fact_extract`) with a JSON-schema-constrained
prompt to extract 5 categories of claims. Persist to `verifiable_claims`.

Runs ONCE per paper (all variants share the extracted claims). The Extract
stage's idempotency check returns True if claims exist for the paper.
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    ClaimType,
    PipelineStage,
    TranslationUnit,
    VerifiableClaim,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.factory import create_provider
from xinda.translation import fact_anchors
from xinda.translation.prompts import STRICT_JSON_FOOTER
from xinda.translation.rate_limit import RateLimiter

logger = setup_logger(__name__)


# Operates on the ENGLISH source only — target-language independent, so the
# instruction is plain English with no target-language assumption.
FACT_EXTRACT_SYSTEM_PROMPT = (
    "You are a fact-extraction expert for scientific papers. From the given paragraph, "
    "identify every 'verifiable scientific claim', classify it into 5 types, and normalize it.\n\n"
    "The 5 claim types:\n"
    "- numeric: a concrete value (precision, step count, parameter count, metric); keep units. "
    "Includes quantities with numbers (e.g. '128 tokens', '15 billion')\n"
    "- citation: author+year or numeric citation (e.g. [12]); normalized as 'FirstAuthor+Year' "
    "or 'ref:N'\n"
    "- comparison: an 'A better/worse/equal than B [by X]' statement; direction + baseline + delta\n"
    "- method_name: ONLY a **specifically named** algorithm/optimizer/dataset/model/system proper "
    "name (e.g. 'Adam', 'BLEU', 'GPT-4', 'ImageNet', 'Transformer').\n"
    "  Do **NOT** extract generic disciplinary concepts or common noun phrases (e.g. "
    "'reinforcement learning', 'large language models', 'artificial intelligence', "
    "'dialogue agents', 'curated data') — those get translated, they are not fixed proper names.\n"
    "- symbol: ONLY a **genuine math symbol/variable** and its value (e.g. 'α=0.1', 'λ', 'θ', "
    "'x_i', 'β=0.9').\n"
    "  Do **NOT** treat a descriptive English noun phrase (e.g. 'hyper-parameter', 'ranking "
    "score', 'prior probability', 'correction rate') as a symbol — those are ordinary text and "
    "get translated.\n\n"
    "Rules:\n"
    "- When in doubt, leave it out: a wrong type label is worse than a miss\n"
    "- surface must be an exact substring of the paragraph (including punctuation/spaces)\n"
    "- normalized gives the normalized form; metadata provides auxiliary info\n"
    "- Do not extract content inside {{PT_XXX_N}} placeholders\n\n"
    "Output a JSON object with field `claims`, an array of objects each with "
    "type/surface/normalized/metadata fields.\n"
    + STRICT_JSON_FOOTER
)


FACT_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["numeric", "citation", "comparison", "method_name", "symbol"],
                    },
                    "surface": {"type": "string"},
                    "normalized": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["type", "surface", "normalized"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class FactExtract:
    """Stage: LLM-based extraction of verifiable_claims for every unit."""

    name = PipelineStage.fact_extract
    recoverable = True

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.paper_id is None:
            return False
        n = await session.scalar(
            select(func.count(VerifiableClaim.id))
            .join(TranslationUnit, VerifiableClaim.unit_id == TranslationUnit.id)
            .where(TranslationUnit.paper_id == ctx.paper_id)
        ) or 0
        return n > 0

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.paper_id is None:
            raise StageError("FactExtract requires paper_id")

        # Clear partial prior state (defensive; usually is_done returns True)
        prev_ids_q = (
            select(VerifiableClaim.id)
            .join(TranslationUnit, VerifiableClaim.unit_id == TranslationUnit.id)
            .where(TranslationUnit.paper_id == ctx.paper_id)
        )
        prev_ids = [r[0] for r in (await session.execute(prev_ids_q)).all()]
        if prev_ids:
            await session.execute(
                delete(VerifiableClaim).where(VerifiableClaim.id.in_(prev_ids))
            )
            await session.commit()

        units = (
            await session.execute(
                select(TranslationUnit)
                .where(TranslationUnit.paper_id == ctx.paper_id)
                .order_by(TranslationUnit.ord)
            )
        ).scalars().all()
        if not units:
            raise StageError("no translation_units; Extract didn't populate?")

        provider = create_provider(settings.model_fact_extract)
        limiter = RateLimiter(provider.rpm, provider.tpm)

        sem = asyncio.Semaphore(settings.max_concurrency)

        async def worker(u: TranslationUnit) -> list[VerifiableClaim]:
            async with sem:
                return await _extract_for_unit(provider, limiter, u)

        results = await asyncio.gather(*(worker(u) for u in units))

        # Persist
        total = 0
        for batch in results:
            for c in batch:
                session.add(c)
                total += 1
        await session.commit()

        logger.info(
            "fact_extract: %d claims from %d units (paper %d)",
            total, len(units), ctx.paper_id,
        )
        return ctx


async def _extract_for_unit(provider, limiter, unit: TranslationUnit) -> list[VerifiableClaim]:
    if not unit.src_plain or len(unit.src_plain) < 8:
        return []
    prompt = f"<INPUT>{unit.src_plain}</INPUT>"
    est = provider.estimate_tokens(FACT_EXTRACT_SYSTEM_PROMPT) + provider.estimate_tokens(prompt)
    await limiter.reserve(est)

    try:
        tr = await provider.generate(
            prompt=prompt,
            system=FACT_EXTRACT_SYSTEM_PROMPT,
            json_schema=FACT_EXTRACT_SCHEMA,
        )
    except Exception as e:
        logger.warning("fact_extract failed for unit %d: %s", unit.id, e)
        return []

    raw = (tr.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("fact_extract JSON parse failed for unit %d: %s", unit.id, e)
        return []

    items = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []

    out: list[VerifiableClaim] = []
    for it in items:
        try:
            ctype = ClaimType(it["type"])
        except (KeyError, ValueError):
            continue
        surface = it.get("surface") or ""
        normalized = it.get("normalized") or surface
        metadata = it.get("metadata")
        if not isinstance(metadata, dict):  # model sometimes returns a str/None
            metadata = {}
        if not surface or not normalized:
            continue
        # Re-normalize via our regex routines for consistency (overrides LLM's
        # normalization where possible, preserves LLM's where not).
        norm_calc, meta_calc = fact_anchors.normalize(ctype, surface)
        if norm_calc and norm_calc != surface:
            normalized = norm_calc
        if isinstance(meta_calc, dict) and meta_calc:
            metadata = {**metadata, **meta_calc}
        # span lookup (best-effort)
        try:
            start = unit.src_plain.index(surface)
            end = start + len(surface)
        except ValueError:
            start, end = None, None

        out.append(VerifiableClaim(
            unit_id=unit.id,
            claim_type=ctype,
            span_start=start,
            span_end=end,
            surface_form=surface,
            normalized=normalized,
            claim_metadata=metadata,
            extracted_by=provider.model_name,
            confidence=1.0,
        ))
    return out
