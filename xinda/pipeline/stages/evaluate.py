"""Evaluate stage: per-unit COMET + xCOMET + paper-level PPA/MFR + persist.

Runs after Render. Reads the latest Translation per unit, scores against
the src_plain, and writes:
- per-unit: translations.comet_score, translations.xcomet_score
- per-job: evaluation_runs row with means + ordered PPA/MFR
"""

from __future__ import annotations

import statistics

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    EvaluationRun,
    PipelineStage,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.config import settings
from xinda.evaluation import comet as comet_mod
from xinda.evaluation import cost
from xinda.evaluation import metrics as metrics_mod
from xinda.evaluation import xcomet as xcomet_mod
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError

logger = setup_logger(__name__)


class Evaluate:
    """Stage: per-unit COMET-Kiwi + xCOMET; per-job PPA/MFR; persist all."""

    name = PipelineStage.evaluate
    recoverable = True   # eval failure shouldn't kill the whole job

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.job_id is None:
            return False
        row = (
            await session.execute(
                select(EvaluationRun).where(EvaluationRun.job_id == ctx.job_id)
            )
        ).scalar_one_or_none()
        return row is not None

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("Evaluate requires job_id/paper_id")
        if ctx.xml_src_path is None or ctx.xml_tgt_path is None:
            raise StageError("Evaluate requires xml paths from Render stage")

        # Load (unit, latest translation) pairs
        rows = (
            await session.execute(
                select(TranslationUnit, Translation)
                .join(Translation, Translation.unit_id == TranslationUnit.id)
                .where(
                    TranslationUnit.paper_id == ctx.paper_id,
                    Translation.job_id == ctx.job_id,
                    Translation.status != TuStatus.pending,
                )
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).all()

        latest: dict[int, tuple[TranslationUnit, Translation]] = {}
        for u, t in rows:
            latest.setdefault(t.unit_id, (u, t))

        if not latest:
            raise StageError("no translations to evaluate")

        # Build pairs in stable unit order
        ordered_units = sorted(latest.values(), key=lambda x: x[0].ord)
        pairs = [(u.src_plain, t.tgt_plain or "") for u, t in ordered_units]

        # Run COMET + xCOMET (threadpool — these load big models). The neural
        # QE models require torch + gated HF checkpoints; if unavailable, degrade
        # gracefully (this stage is recoverable) and still produce the PPA/MFR +
        # token-accounting evaluation_runs row.
        try:
            comet_scores = await run_in_threadpool(comet_mod.score_pairs, pairs)
        except Exception as e:  # noqa: BLE001
            logger.warning("COMET scoring unavailable (%s); skipping neural QE", e)
            comet_scores = [None] * len(pairs)
        try:
            xcomet_results = await run_in_threadpool(xcomet_mod.score_pairs, pairs)
            xcomet_scores = [r["score"] for r in xcomet_results]
        except Exception as e:  # noqa: BLE001
            logger.warning("xCOMET scoring unavailable (%s); skipping neural QE", e)
            xcomet_scores = [None] * len(pairs)

        # Persist per-unit scores
        for (u, t), c, x in zip(ordered_units, comet_scores, xcomet_scores):
            t.comet_score = float(c) if c is not None else None
            t.xcomet_score = float(x) if x is not None else None
        await session.commit()

        # Paper-level PPA / MFR
        ppa_mfr = await run_in_threadpool(
            metrics_mod.compute, str(ctx.xml_src_path), str(ctx.xml_tgt_path)
        )

        comet_valid = [c for c in comet_scores if c is not None]
        xcomet_valid = [x for x in xcomet_scores if x is not None]

        # Aggregate token totals
        total_cached = sum(t.cached_prompt_tokens or 0 for _, t in ordered_units)
        total_fresh = sum(t.fresh_prompt_tokens or 0 for _, t in ordered_units)
        total_completion = sum(t.completion_tokens or 0 for _, t in ordered_units)

        # Pass-no counters
        pass1_units = sum(1 for _, t in ordered_units if t.pass_no == 1)
        refined_units = sum(1 for _, t in ordered_units if t.pass_no > 1)
        fallback_units = sum(1 for _, t in ordered_units if t.status == TuStatus.fallback)

        eval_row = EvaluationRun(
            job_id=ctx.job_id,
            comet_mean=statistics.mean(comet_valid) if comet_valid else None,
            comet_median=statistics.median(comet_valid) if comet_valid else None,
            comet_p10=_percentile(comet_valid, 10) if comet_valid else None,
            xcomet_mean=statistics.mean(xcomet_valid) if xcomet_valid else None,
            ppa=ppa_mfr["ppa"],
            ppa_ordered=ppa_mfr["ppa_ordered"],
            mfr=ppa_mfr["mfr"],
            mfr_ordered=ppa_mfr["mfr_ordered"],
            pass1_units=pass1_units,
            refined_units=refined_units,
            fallback_units=fallback_units,
            total_units=len(ordered_units),
            total_prompt_tok=total_fresh + total_cached,
            total_cached_tok=total_cached,
            total_completion_tok=total_completion,
            cost_cny=cost.cost_cny(
                fresh_prompt_tok=total_fresh,
                cached_prompt_tok=total_cached,
                completion_tok=total_completion,
                model_name=settings.model_first_pass,
            ),
        )
        session.add(eval_row)
        await session.commit()

        logger.info(
            "evaluate: COMET μ=%.3f xCOMET μ=%.3f PPA=%.1f MFR=%.1f (job %s)",
            eval_row.comet_mean or 0, eval_row.xcomet_mean or 0,
            eval_row.ppa or 0, eval_row.mfr or 0, ctx.job_id,
        )
        return ctx


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
