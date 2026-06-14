"""M2 smoke test: full Acquire → … → Render pipeline on one paper.

Usage:
    python -m xinda.cli.translate_smoke 2503.15129 zh

Requires:
    - v3 DB schema applied
    - DASHSCOPE_API_KEY env var
    - LaTeXML on PATH (or LATEXMLC_PATH/LATEXMLPOST_PATH env overrides)
    - source paper under static/input/{arxiv_id}/ (or reachable from arxiv)
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
    RenderedFile,
    RunVariant,
    Translation,
    TranslationJob,
    TranslationUnit,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.config import variants_for
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import Orchestrator
from xinda.pipeline.stages.acquire import Acquire
from xinda.pipeline.stages.apply import ApplyXML
from xinda.pipeline.stages.coherence import Coherence
from xinda.pipeline.stages.convert import Convert
from xinda.pipeline.stages.cross_doc_verify import CrossDocFactVerify
from xinda.pipeline.stages.evaluate import Evaluate
from xinda.pipeline.stages.extract import Extract
from xinda.pipeline.stages.fact_extract import FactExtract
from xinda.pipeline.stages.fact_verify import FactVerify
from xinda.pipeline.stages.glossary import GlossaryBuild
from xinda.pipeline.stages.refine import Refine
from xinda.pipeline.stages.render import Render
from xinda.pipeline.stages.translate import FirstPassTranslate

logger = setup_logger(__name__)


async def run(arxiv_id: str, language: str = "zh") -> None:
    cfg = variants_for(language)["full"]  # type: ignore[arg-type]

    workspace = (
        settings.workspace_dir / arxiv_id / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    async with async_session() as session:
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
            output_dir=str(workspace),
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

    orch = Orchestrator(stages=[
        Acquire(),
        Convert(),
        Extract(),
        FactExtract(),
        GlossaryBuild(),
        FirstPassTranslate(),
        FactVerify(),
        CrossDocFactVerify(),
        Refine(),
        Coherence(),
        ApplyXML(),
        Render(),
        Evaluate(),
    ])
    await orch.run(ctx)

    async with async_session() as session:
        n_units = await session.scalar(
            select(func.count(TranslationUnit.id))
            .where(TranslationUnit.paper_id == paper_id)
        )
        n_trans = await session.scalar(
            select(func.count(Translation.id))
            .where(Translation.job_id == job_id)
        )
        files = (
            await session.execute(
                select(RenderedFile).where(RenderedFile.job_id == job_id)
            )
        ).scalars().all()

    print()
    print(f"=== M2 smoke test for {arxiv_id} ({language}) ===")
    print(f"  paper_id       = {paper_id}")
    print(f"  job_id         = {job_id}")
    print(f"  workspace      = {workspace}")
    print(f"  trans_units    = {n_units}")
    print(f"  translations   = {n_trans}")
    print()
    print("  rendered files:")
    for f in files:
        ok = "✓" if Path(f.storage_path).exists() else "✗"
        size = f"{f.size_bytes:,}" if f.size_bytes else "?"
        print(f"    [{ok}] {f.kind:<15} {size:>10} bytes  {f.storage_path}")
    print()
    if ctx.html_bilingual_path:
        print(f"  open: file://{ctx.html_bilingual_path.resolve()}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m xinda.cli.translate_smoke <arxiv_id> [lang]")
        sys.exit(1)
    arxiv_id = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "zh"
    asyncio.run(run(arxiv_id, lang))


if __name__ == "__main__":
    main()
