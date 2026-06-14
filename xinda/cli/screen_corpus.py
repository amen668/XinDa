"""Convertibility pre-screen: run Acquire→Convert→Extract over a paper list and
record which papers LaTeXML can actually convert (and how many units), so the
full translate batch only runs on convertible papers. NO model calls.

Some papers (big experimental-collaboration classes like ATLAS's atlasdoc.cls +
heavy expl3) crash LaTeXML — Convert's no-`--includestyles` fallback rescues the
\\usepackage-only failures, but main-class failures still fail here. This CLI
flags them so they can be dropped/replaced before the costly translate batch.

    python -m xinda.cli.screen_corpus corpus/paper_ids.txt \
        --ok corpus/ok_ids.txt --report corpus/screen.csv

Run it in the `app` container (needs latexmlc). Resumable: a paper already
extracted (units in DB) is counted OK without re-running.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import (
    JobStatus,
    Paper,
    RunVariant,
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


async def _get_or_create_job(session, arxiv_id: str, cfg, lang: str):
    """Reuse an existing (paper, lang, variant, config_hash) job or create one —
    avoids the translation_jobs unique-key violation on re-runs."""
    paper = (
        await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
    ).scalar_one_or_none()
    if paper is None:
        paper = Paper(arxiv_id=arxiv_id, title="<pending>")
        session.add(paper)
        await session.commit()
        await session.refresh(paper)
    job = (
        await session.execute(
            select(TranslationJob).where(
                TranslationJob.paper_id == paper.id,
                TranslationJob.language == lang,
                TranslationJob.config_hash == cfg.config_hash(),
            )
        )
    ).scalar_one_or_none()
    if job is None:
        job = TranslationJob(
            paper_id=paper.id, language=lang, provider="qianwen",
            model_name=cfg.pass1_model, refine_model=cfg.refine_model,
            variant=RunVariant(cfg.variant), config_hash=cfg.config_hash(),
            status=JobStatus.pending,
            start_time=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
    return paper.id, job.id


async def screen_one(arxiv_id: str, lang: str) -> dict:
    cfg = variants_for(lang)["full"]  # type: ignore[arg-type]
    workspace = settings.workspace_dir / arxiv_id / datetime.now().strftime("%Y%m%d_%H%M%S")
    async with async_session() as session:
        paper_id, job_id = await _get_or_create_job(session, arxiv_id, cfg, lang)
        already = (
            await session.execute(
                select(func.count(TranslationUnit.id)).where(
                    TranslationUnit.paper_id == paper_id
                )
            )
        ).scalar_one()
    if already > 0:
        return {"arxiv_id": arxiv_id, "status": "OK", "units": already, "error": "(cached)"}

    ctx = PipelineContext(arxiv_id=arxiv_id, paper_id=paper_id, job_id=job_id,
                          workspace=workspace, config=cfg)
    try:
        await Orchestrator(stages=[Acquire(), Convert(), Extract()]).run(ctx)
    except Exception as e:  # noqa: BLE001
        stage = "convert" if "latexmlc" in str(e) else (
            "acquire" if "arxiv" in str(e).lower() else "extract")
        return {"arxiv_id": arxiv_id, "status": "FAIL", "units": 0,
                "error": f"{stage}: {str(e)[:100]}"}
    async with async_session() as session:
        n = (
            await session.execute(
                select(func.count(TranslationUnit.id)).where(
                    TranslationUnit.paper_id == paper_id
                )
            )
        ).scalar_one()
    return {"arxiv_id": arxiv_id, "status": "OK" if n > 0 else "FAIL",
            "units": n, "error": "" if n > 0 else "no units extracted"}


async def amain(args: argparse.Namespace) -> None:
    ids = [
        ln.strip() for ln in Path(args.paper_ids).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    rows: list[dict] = []
    for i, aid in enumerate(ids, 1):
        logger.info("screening %d/%d  %s", i, len(ids), aid)
        r = await screen_one(aid, args.lang)
        logger.info("  → %s units=%d %s", r["status"], r["units"], r["error"])
        rows.append(r)

    ok = [r["arxiv_id"] for r in rows if r["status"] == "OK"]
    Path(args.ok).write_text("\n".join(ok) + ("\n" if ok else ""), encoding="utf-8")
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "status", "units", "error"])
        for r in rows:
            w.writerow([r["arxiv_id"], r["status"], r["units"], r["error"]])

    nfail = len(rows) - len(ok)
    print("\n" + "=" * 60)
    print(f"screened {len(rows)}: OK={len(ok)}  FAIL={nfail}")
    for r in rows:
        if r["status"] == "FAIL":
            print(f"  FAIL  {r['arxiv_id']:12} {r['error']}")
    print("=" * 60)
    print(f"convertible ids → {args.ok}")
    print(f"full report     → {args.report}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_ids")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--ok", default="corpus/ok_ids.txt")
    ap.add_argument("--report", default="corpus/screen.csv")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
