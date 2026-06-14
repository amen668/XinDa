"""Re-run ONLY FactExtract + FactVerify for an existing job (one paper).

Used after improving the fact-extraction prompt / verifier so the claim set and
claim_verifications are regenerated without re-paying the other 10 stages.

    python -m xinda.cli.refacts <job_id> [lang]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import Paper, TranslationJob
from xinda.pipeline.config import variants_for
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.stages.fact_extract import FactExtract
from xinda.pipeline.stages.fact_verify import FactVerify


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
        ctx = PipelineContext(
            arxiv_id=paper.arxiv_id, paper_id=paper.id, job_id=job_id,
            workspace=Path(job.output_dir), config=variants_for(lang)["full"],
        )

    async with async_session() as session:
        print("re-extracting claims (FactExtract)…")
        await FactExtract().run(ctx, session)
    async with async_session() as session:
        print("re-verifying claims (FactVerify)…")
        await FactVerify().run(ctx, session)
    print(f"refacts complete for job {job_id}")


if __name__ == "__main__":
    jid = int(sys.argv[1])
    language = sys.argv[2] if len(sys.argv) > 2 else "zh"
    asyncio.run(amain(jid, language))
