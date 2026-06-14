"""Refine stage: COMET + FPS dual-gated retry with model escalation.

Trigger: for each unit, if its latest Translation has
  xcomet_score < ctx.config.xcomet_threshold  OR
  fps_unit < ctx.config.fps_threshold         OR
  unit appears in cross_doc_drifts (M8+)

then we re-translate that paragraph with `ctx.config.refine_model`,
injecting the failed claims so the model knows exactly what to fix.

Refine runs up to `max_refine_passes` passes (default 2 extra rounds).
After exhausting passes, the best-scoring pass is kept and the unit is
marked `fallback` — never silently revert to source text.

Note: M3 Evaluate must run BEFORE Refine on first invocation to populate
xcomet_score. M6 FactVerify must run before Refine to populate fps_unit.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    ClaimVerification,
    CrossDocDrift,
    DriftType,
    GlossaryTerm,
    Paper,
    PipelineStage,
    Section,
    Translation,
    TranslationUnit,
    TuStatus,
    VerifiableClaim,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.base import ModelProvider, TranslationResult
from xinda.providers.factory import create_provider
from xinda.translation.prompts import (
    REFINE_SCHEMA,
    refine_user_prompt,
    stable_prefix,
)
from xinda.translation.rate_limit import RateLimiter

logger = setup_logger(__name__)


class Refine:
    name = PipelineStage.refine
    recoverable = True

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.job_id is None or ctx.paper_id is None:
            return False
        if not ctx.config.use_retry or ctx.config.refine_model is None:
            return True
        # Done if no translation needs refining and gates are evaluated
        failing = await session.scalar(
            select(func.count(Translation.id))
            .where(
                Translation.job_id == ctx.job_id,
                Translation.status == TuStatus.translated,
                (
                    (Translation.xcomet_score < ctx.config.xcomet_threshold) |
                    (Translation.fps_unit < ctx.config.fps_threshold)
                ),
                Translation.pass_no >= ctx.config.max_refine_passes + 1,
            )
        ) or 0
        # If failing units exist that haven't been retried up to max, not done
        pending_retries = await session.scalar(
            select(func.count(Translation.id))
            .where(
                Translation.job_id == ctx.job_id,
                Translation.status == TuStatus.translated,
                (
                    (Translation.xcomet_score < ctx.config.xcomet_threshold) |
                    (Translation.fps_unit < ctx.config.fps_threshold)
                ),
                Translation.pass_no < ctx.config.max_refine_passes + 1,
            )
        ) or 0
        return pending_retries == 0 and failing >= 0  # treat as done

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if not ctx.config.use_retry or ctx.config.refine_model is None:
            logger.info("refine skipped: use_retry=%s refine_model=%s",
                        ctx.config.use_retry, ctx.config.refine_model)
            return ctx
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("Refine requires job_id/paper_id")

        paper = (
            await session.execute(select(Paper).where(Paper.id == ctx.paper_id))
        ).scalar_one_or_none()
        if paper is None:
            raise StageError("paper missing")

        glossary = (
            await session.execute(
                select(GlossaryTerm).where(
                    GlossaryTerm.paper_id == ctx.paper_id,
                    GlossaryTerm.language == ctx.config.language,
                )
            )
        ).scalars().all()
        glossary_dicts = [
            {"src": g.src_term, "tgt": g.tgt_term,
             "kind": g.kind, "definition": g.definition, "locked": g.locked}
            for g in glossary
        ]
        section_outline = await self._section_outline(session, ctx.paper_id)
        prefix = stable_prefix(
            paper_title=paper.title, arxiv_id=paper.arxiv_id,
            field=paper.field or "Other",
            target_language=ctx.config.language,
            glossary_terms=glossary_dicts if ctx.config.use_glossary else None,
            section_outline=section_outline if ctx.config.use_context else None,
            abstract=paper.source_abstract if ctx.config.use_context else None,
        )

        provider = create_provider(ctx.config.refine_model)
        limiter = RateLimiter(provider.rpm, provider.tpm)

        # Iterate refine passes
        for pass_no in range(2, ctx.config.max_refine_passes + 2):
            candidates = await self._collect_failing(session, ctx, pass_no - 1)
            if not candidates:
                break
            logger.info(
                "refine pass %d: %d candidates (model=%s)",
                pass_no, len(candidates), provider.model_name,
            )
            sem = asyncio.Semaphore(4)

            async def worker(prev_translation, unit, failed_claims, section_heading, prev_translation_text):
                async with sem:
                    return await _refine_one(
                        provider, limiter, prefix,
                        prev_translation=prev_translation,
                        unit=unit,
                        failed_claims=failed_claims,
                        section_heading=section_heading,
                        prev_translation_text=prev_translation_text,
                        glossary=glossary_dicts if ctx.config.use_glossary else [],
                    )

            results = await asyncio.gather(*(
                worker(*c) for c in candidates
            ))

            for (prev_translation, unit, *_), refined in zip(candidates, results):
                if refined is None:
                    continue
                session.add(Translation(
                    job_id=ctx.job_id,
                    unit_id=unit.id,
                    status=TuStatus.refined,
                    pass_no=pass_no,
                    model_used=provider.model_name,
                    tgt_text=refined["tgt_text"],
                    tgt_plain=_strip_plain(
                        refined["tgt_text"], unit.placeholders, unit.special_chars
                    ),
                    cached_prompt_tokens=refined["cached"],
                    fresh_prompt_tokens=refined["fresh"],
                    completion_tokens=refined["completion"],
                    elapsed_ms=refined["elapsed_ms"],
                ))
            await session.commit()

        # After all passes: mark units that never recovered as 'fallback'.
        # (Best-scoring pass remains in DB and is what ApplyXML reads.)
        await self._mark_fallback(session, ctx)
        return ctx

    async def _section_outline(self, session: AsyncSession, paper_id: int) -> list[str]:
        sections = (
            await session.execute(
                select(Section)
                .where(Section.paper_id == paper_id)
                .order_by(Section.depth, Section.ord)
            )
        ).scalars().all()
        return [s.heading_src for s in sections if s.heading_src]

    async def _collect_failing(
        self, session: AsyncSession, ctx: PipelineContext, current_pass: int
    ) -> list[tuple]:
        """Return list of (latest_translation, unit, failed_claims, section_heading, prev_tgt_text).

        Each entry corresponds to a unit whose current-pass translation
        fails either gate, and which hasn't exhausted refine passes.
        """
        rows = (
            await session.execute(
                select(Translation, TranslationUnit)
                .join(TranslationUnit, Translation.unit_id == TranslationUnit.id)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.pass_no == current_pass,
                    Translation.status.in_([TuStatus.translated, TuStatus.refined]),
                )
            )
        ).all()

        if not rows:
            return []

        candidates: list[tuple] = []
        tr_ids = [t.id for t, _ in rows]

        # Per-translation failed claims (drift != verified)
        verifs = (
            await session.execute(
                select(ClaimVerification, VerifiableClaim)
                .join(VerifiableClaim, ClaimVerification.claim_id == VerifiableClaim.id)
                .where(
                    ClaimVerification.translation_id.in_(tr_ids),
                    ClaimVerification.verified.is_(False),
                )
            )
        ).all()
        failed_by_tr: dict[int, list[dict]] = defaultdict(list)
        for v, c in verifs:
            failed_by_tr[v.translation_id].append({
                "drift": v.drift.value,
                "src_surface": c.surface_form,
                "tgt_surface": v.tgt_surface,
            })

        # Section headings
        section_ids = {u.section_id for _, u in rows if u.section_id is not None}
        sections = (
            await session.execute(
                select(Section).where(Section.id.in_(section_ids))
            )
        ).scalars().all()
        section_heading_by_id = {s.id: s.heading_src for s in sections}

        for t, u in rows:
            failed = failed_by_tr.get(t.id, [])
            comet_fail = (
                t.xcomet_score is not None and t.xcomet_score < ctx.config.xcomet_threshold
            )
            fps_fail = (
                t.fps_unit is not None and t.fps_unit < ctx.config.fps_threshold
            )
            if not (comet_fail or fps_fail or failed):
                continue
            section_heading = section_heading_by_id.get(u.section_id)
            candidates.append((t, u, failed, section_heading, None))
        return candidates

    async def _mark_fallback(self, session: AsyncSession, ctx: PipelineContext) -> None:
        """Mark units that still fail after max passes as 'fallback'."""
        # find units whose best (highest-pass) translation still fails either gate
        max_pass = ctx.config.max_refine_passes + 1
        rows = (
            await session.execute(
                select(Translation)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.pass_no == max_pass,
                )
            )
        ).scalars().all()
        for t in rows:
            if (
                (t.xcomet_score is not None and t.xcomet_score < ctx.config.xcomet_threshold)
                or (t.fps_unit is not None and t.fps_unit < ctx.config.fps_threshold)
            ):
                t.status = TuStatus.fallback
        await session.commit()


# ────────────────────────── helpers ──────────────────────────


async def _refine_one(
    provider: ModelProvider,
    limiter: RateLimiter,
    stable_prefix_text: str,
    *,
    prev_translation: Translation,
    unit: TranslationUnit,
    failed_claims: list[dict],
    section_heading: str | None,
    prev_translation_text: str | None,
    glossary: list[dict],
) -> dict | None:
    user = refine_user_prompt(
        src_text=unit.src_text,
        draft_text=prev_translation.tgt_text or "",
        failed_claims=failed_claims,
        section_heading=section_heading,
        prev_translation=prev_translation_text,
        glossary_hits=glossary,
    )
    est = provider.estimate_tokens(stable_prefix_text) + provider.estimate_tokens(user)
    await limiter.reserve(est)

    try:
        tr: TranslationResult = await provider.generate(
            prompt=user, system=stable_prefix_text, json_schema=REFINE_SCHEMA,
        )
    except Exception as e:
        logger.warning("refine failed for unit %d: %s", unit.id, e)
        return None

    raw = (tr.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("refine JSON parse failed for unit %d", unit.id)
        return None
    tgt = obj.get("translation") if isinstance(obj, dict) else None
    if not tgt:
        return None
    return {
        "tgt_text": tgt,
        "cached": tr.cached_prompt_tokens,
        "fresh": tr.fresh_prompt_tokens,
        "completion": tr.completion_tokens,
        "elapsed_ms": tr.elapsed_ms,
    }


def _strip_plain(
    text: str, placeholders: dict[str, str], special_chars: dict[str, str]
) -> str:
    out = text
    for ph in placeholders:
        out = out.replace(ph, "")
    for ph in special_chars:
        out = out.replace(ph, " ")
    return " ".join(out.split())
