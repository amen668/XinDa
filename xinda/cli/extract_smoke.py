"""M1 smoke test: run Acquire → Convert → Extract on one paper, verify counts.

Usage:
    python -m xinda.cli.extract_smoke 2503.15129

Requires:
    - DATABASE_URL env var pointing at an arxiv_translation_hub PG instance
      with the v3 schema applied (psql -f arxiv_translation_hub.sql).
    - LATEXMLC_PATH on PATH (or via env override).
    - Existing source under static/input/{arxiv_id}/ (or arxiv_id reachable
      via arxiv.org/e-print).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import (
    JobStatus,
    Paper,
    RunVariant,
    Section,
    TranslationJob,
    TranslationUnit,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.config import variants_for
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import Orchestrator
from xinda.pipeline.stages.acquire import Acquire
from xinda.pipeline.stages.convert import Convert
from xinda.pipeline.stages.extract import Extract

logger = setup_logger(__name__)


async def run(arxiv_id: str, language: str = "zh") -> None:
    cfg = variants_for(language)["full"]  # type: ignore[arg-type]

    workspace = (
        settings.workspace_dir / arxiv_id / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    async with async_session() as session:
        # create a stub job up-front so orchestrator can update last_stage
        paper = (
            await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        ).scalar_one_or_none()
        if paper is None:
            paper = Paper(arxiv_id=arxiv_id, title="<pending>")
            session.add(paper)
            await session.commit()
            await session.refresh(paper)

        job = TranslationJob(
            paper_id=paper.id,
            language=language,
            provider="qianwen",
            model_name=cfg.pass1_model,
            refine_model=cfg.refine_model,
            variant=RunVariant(cfg.variant),
            config_hash=cfg.config_hash(),
            status=JobStatus.pending,
            start_time=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        paper_id, job_id = paper.id, job.id

    ctx = PipelineContext(
        arxiv_id=arxiv_id,
        paper_id=paper_id,
        job_id=job_id,
        workspace=workspace,
        config=cfg,
    )

    orch = Orchestrator(stages=[Acquire(), Convert(), Extract()])
    await orch.run(ctx)

    # Verify counts
    async with async_session() as session:
        n_sections = (
            await session.execute(
                select(func.count(Section.id)).where(Section.paper_id == ctx.paper_id)
            )
        ).scalar_one()
        n_units = (
            await session.execute(
                select(func.count(TranslationUnit.id)).where(
                    TranslationUnit.paper_id == ctx.paper_id
                )
            )
        ).scalar_one()

    print()
    print(f"=== M1 smoke test result for {arxiv_id} ===")
    print(f"  paper_id    = {ctx.paper_id}")
    print(f"  job_id      = {ctx.job_id}")
    print(f"  workspace   = {ctx.workspace}")
    print(f"  xml         = {ctx.xml_src_path}")
    print(f"  sections    = {n_sections}")
    print(f"  trans_units = {n_units}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m xinda.cli.extract_smoke <arxiv_id> [lang]")
        sys.exit(1)
    arxiv_id = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "zh"
    asyncio.run(run(arxiv_id, lang))


if __name__ == "__main__":
    main()
