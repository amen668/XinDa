"""One-off: expand aggregated RUBRIC-MQM judgment rows into per-run rows.

Early runs stored a single rubric_mqm EvalJudgment per sample (the median of 3
runs) with the 3 raw runs in `raw_response` (a JSON list). meta-eval's
self-consistency check needs the per-run spread, so this expands each aggregate
row into one row per run (run_no=1,2,3) and deletes the original.

Idempotent: rows whose raw_response is already a dict (per-run) are skipped.

    python -m xinda.cli.backfill_judge_runs
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import EvalJudgment


async def amain() -> None:
    async with async_session() as session:
        rows = (
            await session.execute(
                select(EvalJudgment).where(EvalJudgment.protocol == "rubric_mqm")
            )
        ).scalars().all()

        # Collect + delete the aggregate rows FIRST (and flush), so the new
        # per-run rows don't collide with the original on any (sample, run_no)
        # unique constraint still pending in the same flush.
        pending: list[tuple[int, str, list[dict]]] = []
        for j in rows:
            raw = j.raw_response
            if not isinstance(raw, list):
                continue  # already per-run (dict) or empty
            runs = [r for r in raw if isinstance(r, dict) and r.get("rubric_score") is not None]
            if len(runs) < 2:
                continue
            pending.append((j.sample_id, j.judge_model, runs))
            await session.delete(j)
        await session.flush()

        expanded = 0
        for sample_id, judge_model, runs in pending:
            for ri, run in enumerate(runs, start=1):
                sc = run.get("scores") or {}
                session.add(EvalJudgment(
                    sample_id=sample_id, judge_model=judge_model,
                    protocol="rubric_mqm", run_no=ri,
                    rubric_score=run.get("rubric_score"),
                    fluency=sc.get("fluency"),
                    terminology=sc.get("terminology"),
                    structure=sc.get("structure"),
                    raw_response=run,
                ))
            expanded += 1
        await session.commit()
        print(f"backfilled {expanded} aggregate rubric rows → per-run rows")


if __name__ == "__main__":
    asyncio.run(amain())
