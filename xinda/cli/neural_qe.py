"""Backfill COMET-Kiwi + xCOMET neural QE scores for an existing job.

Runs in the GPU `qe` container (torch + unbabel-comet). Loads the latest
translation per unit, scores (src, mt) reference-free, and writes:
- per-unit: translations.comet_score / translations.xcomet_score
- per-job:  evaluation_runs.comet_mean/median/p10 + xcomet_mean

This is the same scoring the Evaluate stage attempts, split out so it can run
on the GPU host image without re-running the whole pipeline.

    python -m xinda.cli.neural_qe <job_id>
"""

from __future__ import annotations

import asyncio
import statistics
import sys

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import (
    EvaluationRun,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.evaluation import comet as comet_mod
from xinda.evaluation import xcomet as xcomet_mod


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


async def amain(job_id: int) -> None:
    async with async_session() as session:
        rows = (
            await session.execute(
                select(TranslationUnit, Translation)
                .join(Translation, Translation.unit_id == TranslationUnit.id)
                .where(
                    Translation.job_id == job_id,
                    Translation.status != TuStatus.pending,
                )
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).all()
        latest: dict[int, tuple[TranslationUnit, Translation]] = {}
        for u, t in rows:
            latest.setdefault(t.unit_id, (u, t))
        if not latest:
            raise SystemExit(f"no translations for job {job_id}")

        ordered = sorted(latest.values(), key=lambda x: x[0].ord)
        pairs = [(u.src_plain, t.tgt_plain or "") for u, t in ordered]

        import torch  # noqa: PLC0415
        gpus = 0
        if torch.cuda.is_available():
            # is_available() can be True while the installed torch lacks kernels
            # for this GPU arch (e.g. Blackwell/sm_120 on torch built for ≤sm_90).
            # Probe a real op; fall back to CPU if it raises.
            try:
                _ = (torch.zeros(1, device="cuda") + 1).item()
                gpus = 1
            except Exception as e:  # noqa: BLE001
                print(f"CUDA present but unusable ({type(e).__name__}); using CPU")
        dev = "GPU" if gpus else "CPU"
        print(f"scoring {len(pairs)} units with COMET-Kiwi + xCOMET-XL on {dev}…")
        comet_scores = comet_mod.score_pairs(pairs, gpus=gpus)
        xcomet_scores = [r["score"] for r in xcomet_mod.score_pairs(pairs, gpus=gpus)]

        for (u, t), c, x in zip(ordered, comet_scores, xcomet_scores):
            t.comet_score = float(c) if c is not None else None
            t.xcomet_score = float(x) if x is not None else None
        await session.commit()

        cv = [c for c in comet_scores if c is not None]
        xv = [x for x in xcomet_scores if x is not None]

        run = (
            await session.execute(
                select(EvaluationRun).where(EvaluationRun.job_id == job_id)
            )
        ).scalar_one_or_none()
        if run is None:
            run = EvaluationRun(job_id=job_id)
            session.add(run)
        run.comet_mean = statistics.mean(cv) if cv else None
        run.comet_median = statistics.median(cv) if cv else None
        run.comet_p10 = _percentile(cv, 10) if cv else None
        run.xcomet_mean = statistics.mean(xv) if xv else None
        await session.commit()

        print("─" * 56)
        print(f"job {job_id}  ({len(pairs)} units)")
        print(f"COMET-Kiwi  mean={run.comet_mean:.4f}  median={run.comet_median:.4f}  "
              f"p10={run.comet_p10:.4f}")
        print(f"xCOMET-XL   mean={run.xcomet_mean:.4f}")
        print("─" * 56)


if __name__ == "__main__":
    asyncio.run(amain(int(sys.argv[1])))
