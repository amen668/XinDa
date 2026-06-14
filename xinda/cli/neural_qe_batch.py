"""Batch neural-QE: score many jobs in ONE process so the COMET/xCOMET model
checkpoints load only once (each is a multi-GB load — running per-job `neural_qe`
reloads them every time).

Runs in the GPU `qe` container.

    python -m xinda.cli.neural_qe_batch 14 20 35 43 54 60 61 107 115
    python -m xinda.cli.neural_qe_batch --papers corpus/trial10_ids.txt --lang zh --variant full
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from xinda.cli.neural_qe import amain
from xinda.db.engine import async_session
from xinda.db.models import JobStatus, Paper, RunVariant, TranslationJob
from xinda.pipeline.config import variants_for


async def _jobs_for_papers(papers_file: str, lang: str, variant: str) -> list[int]:
    from pathlib import Path

    ids = [
        ln.strip() for ln in Path(papers_file).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    cfg = variants_for(lang)[variant]  # type: ignore[index]
    out: list[int] = []
    async with async_session() as s:
        for aid in ids:
            p = (
                await s.execute(select(Paper).where(Paper.arxiv_id == aid))
            ).scalar_one_or_none()
            if p is None:
                continue
            j = (
                await s.execute(
                    select(TranslationJob).where(
                        TranslationJob.paper_id == p.id,
                        TranslationJob.language == lang,
                        TranslationJob.variant == RunVariant(variant),
                        TranslationJob.config_hash == cfg.config_hash(),
                        TranslationJob.status == JobStatus.success,
                    )
                )
            ).scalar_one_or_none()
            if j is not None:
                out.append(j.id)
    return out


async def run(job_ids: list[int]) -> None:
    for jid in job_ids:
        try:
            await amain(jid)
        except SystemExit as e:
            print(f"job {jid}: SKIP ({e})")
        except Exception as e:  # noqa: BLE001
            print(f"job {jid}: ERROR {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_ids", nargs="*", type=int, help="explicit job ids")
    ap.add_argument("--papers", help="file of arxiv_ids; resolve to jobs via lang/variant")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--variant", default="full")
    args = ap.parse_args()

    job_ids = list(args.job_ids)
    if args.papers:
        job_ids += asyncio.run(_jobs_for_papers(args.papers, args.lang, args.variant))
    if not job_ids:
        raise SystemExit("no jobs: pass job ids or --papers")
    # dedupe, preserve order
    seen: set[int] = set()
    job_ids = [j for j in job_ids if not (j in seen or seen.add(j))]
    print(f"batch neural-QE over {len(job_ids)} jobs: {job_ids}")
    asyncio.run(run(job_ids))


if __name__ == "__main__":
    main()
