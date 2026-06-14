"""CrossDocFactVerify stage: whole-paper consistency check via 1M context.

Feeds the entire translated paper + all extracted claims to qwen3.7-max
in a single call. Detects inconsistencies that per-paragraph FactVerify
cannot catch:
- same citation rendered differently in different paragraphs
- same technical term drifting across sections
- comparison direction contradicting earlier statements

Persists cross_doc_drifts rows that Refine reads as an additional gate.
"""

from __future__ import annotations

import json

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.util import loads_dict
from xinda.db.models import (
    CrossDocDrift,
    PipelineStage,
    Translation,
    TranslationUnit,
    TuStatus,
    VerifiableClaim,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER, language_name

logger = setup_logger(__name__)


def _cross_doc_system(target_name: str) -> str:
    return (
        "You are a cross-paragraph consistency auditor for scientific-paper translation.\n"
        f"You will receive the {target_name} translation of an English paper (in paragraph "
        "order) plus all verifiable claims extracted from it.\n\n"
        "Identify **cross-paragraph inconsistencies**:\n"
        "- citation_inconsistency: the same English citation rendered differently in different "
        "paragraphs\n"
        "- term_drift: the same English term translated differently across paragraphs (should be "
        "consistent)\n"
        "- comparison_contradiction: one paragraph says A>B, another says A<B (logical "
        "contradiction)\n\n"
        "For each inconsistency give the involved unit_id array, the relevant surface-form array, "
        "severity, and a brief description.\n\n"
        "Output a JSON object with field `drifts`, an array of objects.\n"
        + STRICT_JSON_FOOTER
    )

_CROSS_DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "drifts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drift_type": {
                        "type": "string",
                        "enum": [
                            "citation_inconsistency",
                            "term_drift",
                            "comparison_contradiction",
                        ],
                    },
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "surface_forms": {"type": "array", "items": {"type": "string"}},
                    "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                    "description": {"type": "string"},
                },
                "required": ["drift_type", "unit_ids", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["drifts"],
    "additionalProperties": False,
}


class CrossDocFactVerify:
    name = PipelineStage.cross_doc_verify
    recoverable = True

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if not ctx.config.use_cross_doc:
            return True
        if ctx.job_id is None:
            return False
        # Done if we already recorded at least one cross-doc audit pass.
        # Use a sentinel: stage marker on translation_jobs.last_stage.
        # Here, simply check whether drifts table has been written or zero
        # was explicitly recorded — we treat any prior commit as done.
        n = await session.scalar(
            select(CrossDocDrift.id).where(CrossDocDrift.job_id == ctx.job_id).limit(1)
        )
        return n is not None

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if not ctx.config.use_cross_doc:
            logger.info("cross_doc_verify skipped by config")
            return ctx
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("CrossDocFactVerify requires job_id/paper_id")

        # Clear prior drifts
        await session.execute(
            delete(CrossDocDrift).where(CrossDocDrift.job_id == ctx.job_id)
        )
        await session.commit()

        # Build the per-unit translated text in order
        rows = (
            await session.execute(
                select(TranslationUnit, Translation)
                .join(Translation, Translation.unit_id == TranslationUnit.id)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.status != TuStatus.pending,
                    TranslationUnit.paper_id == ctx.paper_id,
                )
                .order_by(TranslationUnit.ord, Translation.pass_no.desc())
            )
        ).all()
        latest: dict[int, tuple[TranslationUnit, Translation]] = {}
        for u, t in rows:
            latest.setdefault(u.id, (u, t))
        ordered = sorted(latest.values(), key=lambda x: x[0].ord)

        # Cap to fit ~200k tokens (≈ 800k chars)
        body_lines: list[str] = []
        total_chars = 0
        for u, t in ordered:
            line = f"[unit {u.id}] {t.tgt_plain or ''}"
            if total_chars + len(line) > 600_000:
                break
            body_lines.append(line)
            total_chars += len(line)
        translated_body = "\n\n".join(body_lines)

        # Claims summary (just the surface forms — full corpus is too big otherwise)
        claims = (
            await session.execute(
                select(VerifiableClaim)
                .join(TranslationUnit, VerifiableClaim.unit_id == TranslationUnit.id)
                .where(TranslationUnit.paper_id == ctx.paper_id)
            )
        ).scalars().all()
        claims_summary = json.dumps([
            {"type": c.claim_type.value, "surface": c.surface_form, "unit": c.unit_id}
            for c in claims
        ], ensure_ascii=False)

        user_prompt = (
            f"<TRANSLATED_PAPER>\n{translated_body}\n</TRANSLATED_PAPER>\n\n"
            f"<EXTRACTED_CLAIMS>\n{claims_summary}\n</EXTRACTED_CLAIMS>\n\n"
            "Identify the cross-paragraph inconsistencies and output JSON."
        )

        provider = create_provider(settings.model_cross_doc)
        try:
            tr = await provider.generate(
                prompt=user_prompt,
                system=_cross_doc_system(language_name(ctx.config.language)),
                json_schema=_CROSS_DOC_SCHEMA,
            )
        except Exception as e:
            logger.warning("cross_doc_verify failed for job %d: %s", ctx.job_id, e)
            return ctx

        drifts = _parse(tr.text)
        for d in drifts:
            session.add(CrossDocDrift(
                job_id=ctx.job_id,
                drift_type=d.get("drift_type"),
                unit_ids=d.get("unit_ids") or [],
                surface_forms=d.get("surface_forms") or [],
                severity=d.get("severity"),
                description=d.get("description"),
                detected_by=provider.model_name,
            ))
        await session.commit()
        logger.info("cross_doc_verify: %d drifts for job %d", len(drifts), ctx.job_id)
        return ctx


def _parse(text: str) -> list[dict]:
    obj = loads_dict(text)
    drifts = obj.get("drifts") if obj else None
    return drifts if isinstance(drifts, list) else []
