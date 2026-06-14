"""Resume an existing (failed) translation_job by id, reusing completed stages.

    python -m xinda.cli._resume <job_id> [lang]

The orchestrator's per-stage is_done checks skip already-done work (acquire,
convert, extract, fact_extract, glossary, translate) and resume at the first
incomplete stage. Used to continue a job that crashed mid-pipeline without
re-paying the expensive early stages.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import JobStatus, Paper, TranslationJob
from xinda.evaluation.benchmark import full_stage_list
from xinda.pipeline.config import variants_for
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import Orchestrator


async def amain(job_id: int, lang: str) -> None:
    async with async_session() as session:
        job = (
            await session.execute(select(TranslationJob).where(TranslationJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise SystemExit(f"job {job_id} not found")
        paper = (
            await session.execute(select(Paper).where(Paper.id == job.paper_id))
        ).scalar_one()
        # reset failed status so the run can complete
        job.status = JobStatus.running
        job.error_msg = None
        await session.commit()
        arxiv_id, paper_id, workspace = paper.arxiv_id, paper.id, Path(job.output_dir)

    cfg = variants_for(lang)["full"]
    ctx = PipelineContext(
        arxiv_id=arxiv_id, paper_id=paper_id, job_id=job_id,
        workspace=workspace, config=cfg,
    )
    await Orchestrator(stages=full_stage_list()).run(ctx)
    print(f"resume complete for job {job_id}")


if __name__ == "__main__":
    jid = int(sys.argv[1])
    language = sys.argv[2] if len(sys.argv) > 2 else "zh"
    asyncio.run(amain(jid, language))
