"""Stage Protocol + Orchestrator with idempotency-based resumability.

Each stage implements `is_done(ctx, session)` (consults DB) and `run(ctx,
session)` (does the work). Orchestrator walks STAGES, skipping completed
ones and updating `translation_jobs.last_stage` after each success. A crash
mid-stage leaves the job with the previous `last_stage`; re-running with
the same job_id resumes from where it left off.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.engine import async_session
from xinda.db.models import (
    JobStatus,
    PipelineStage,
    TranslationJob,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext

logger = setup_logger(__name__)


class StageError(Exception):
    """A stage failed in an unrecoverable way; orchestrator should stop."""


@runtime_checkable
class Stage(Protocol):
    """Pipeline stage interface."""

    name: PipelineStage
    recoverable: bool

    async def is_done(self, ctx: PipelineContext, session: AsyncSession) -> bool:
        """True if this stage's outputs already exist for ctx (idempotency)."""
        ...

    async def run(self, ctx: PipelineContext, session: AsyncSession) -> PipelineContext:
        """Execute the stage, returning the (possibly mutated) context."""
        ...


class Orchestrator:
    """Walks a stage list, handling idempotency, resumability, and errors."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run all stages in order. Each stage gets its own DB session."""
        for stage in self.stages:
            async with async_session() as session:
                try:
                    if await stage.is_done(ctx, session):
                        logger.info(
                            "skip stage %s (already done) for job %s",
                            stage.name.value, ctx.job_id,
                        )
                        continue
                    await self._mark_running(ctx, stage.name, session)
                    t0 = time.monotonic()
                    ctx = await stage.run(ctx, session)
                    dt = time.monotonic() - t0
                    logger.info(
                        "stage %s done in %.1fs for job %s",
                        stage.name.value, dt, ctx.job_id,
                    )
                    await self._mark_done(ctx, stage.name, session)
                except StageError as e:
                    logger.error(
                        "stage %s failed for job %s: %s",
                        stage.name.value, ctx.job_id, e,
                    )
                    await self._mark_failed(ctx, stage.name, str(e), session)
                    if not getattr(stage, "recoverable", False):
                        raise
                except Exception as e:
                    logger.exception(
                        "stage %s crashed for job %s",
                        stage.name.value, ctx.job_id,
                    )
                    await self._mark_failed(ctx, stage.name, str(e), session)
                    raise
        # mark whole job success only if we reached the end without re-raising
        async with async_session() as session:
            await self._mark_job_success(ctx, session)
        return ctx

    # ──────────────── DB bookkeeping helpers ────────────────

    async def _get_job(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> TranslationJob | None:
        if ctx.job_id is None:
            return None
        stmt = select(TranslationJob).where(TranslationJob.id == ctx.job_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _mark_running(
        self,
        ctx: PipelineContext,
        stage: PipelineStage,
        session: AsyncSession,
    ) -> None:
        job = await self._get_job(ctx, session)
        if job is None:
            return
        job.status = JobStatus.running
        job.last_stage = stage
        if job.start_time is None:
            job.start_time = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()

    async def _mark_done(
        self,
        ctx: PipelineContext,
        stage: PipelineStage,
        session: AsyncSession,
    ) -> None:
        job = await self._get_job(ctx, session)
        if job is None:
            return
        job.last_stage = stage
        await session.commit()

    async def _mark_failed(
        self,
        ctx: PipelineContext,
        stage: PipelineStage,
        error_msg: str,
        session: AsyncSession,
    ) -> None:
        job = await self._get_job(ctx, session)
        if job is None:
            return
        job.status = JobStatus.failed
        job.error_msg = f"[{stage.value}] {error_msg}"
        job.end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()

    async def _mark_job_success(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> None:
        job = await self._get_job(ctx, session)
        if job is None:
            return
        job.status = JobStatus.success
        job.end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.commit()
