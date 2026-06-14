"""Export a stratified random sample of AUTO-RELEASED units for human fact audit.

Paper §4.5 (major-revision item #2): the gate's "fact-class interceptions cleared"
is measured by the same fidelity verifier that guided refinement (recall ≈85%), so
the residual fact-error rate of the released pool must be established by an
*independent* human audit. This CLI reproduces the exact gate used by
`cli.review_triage` (same latest-pass selection, same `review_gate.assess`),
keeps only units the gate released (flag=False), stratifies by job (proportional
allocation), and writes a UTF-8-BOM CSV with blank judgment columns for two
independent reviewers.

  python -m xinda.cli.export_audit_sample --n 320 --seed 42 \
      --jobs 5,14,20,35,43,54,60,61,107,115 \
      --out results/triage_analysis/audit_sample.csv

The default job list is the 10-paper en→zh cohort reported in Table 5.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
from pathlib import Path

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import (
    ClaimVerification,
    DriftType,
    Paper,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.evaluation import review_gate
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

DEFAULT_JOBS = "5,14,20,35,43,54,60,61,107,115"

JUDGE_COLS = [
    "人工判定(保真/事实错误/不确定)", "错误类型(数值/引用/比较/方法名/符号/其他)", "人工备注",
    "筛查器事实清单(异厂商模型填)", "筛查器判定", "筛查器理由",
    "最终判定", "分歧复核备注",
]


async def _released_units(job_id: int) -> list[dict]:
    """Same selection + gate as cli.review_triage; return released (unflagged) units."""
    async with async_session() as session:
        rows = (
            await session.execute(
                select(Translation, TranslationUnit, Paper.arxiv_id)
                .join(TranslationUnit, Translation.unit_id == TranslationUnit.id)
                .join(Paper, TranslationUnit.paper_id == Paper.id)
                .where(Translation.job_id == job_id,
                       Translation.status != TuStatus.pending)
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).all()
        if not rows:
            logger.warning("job %s: no translations", job_id)
            return []
        latest: dict[int, tuple[Translation, TranslationUnit, str]] = {}
        for t, u, aid in rows:
            latest.setdefault(t.unit_id, (t, u, aid))

        tr_ids = [t.id for t, _, _ in latest.values()]
        drifts: dict[int, list[str]] = {}
        if tr_ids:
            dv = (
                await session.execute(
                    select(ClaimVerification.translation_id, ClaimVerification.drift)
                    .where(ClaimVerification.translation_id.in_(tr_ids),
                           ClaimVerification.drift != DriftType.verified)
                )
            ).all()
            for tid, d in dv:
                drifts.setdefault(tid, []).append(d.value if hasattr(d, "value") else str(d))

    released: list[dict] = []
    for t, u, aid in latest.values():
        sig = review_gate.UnitSignals(
            fps_unit=t.fps_unit,
            xcomet_score=t.xcomet_score,
            comet_score=t.comet_score,
            rubric_score=getattr(t, "rubric_score", None),
            geval_score=getattr(t, "geval_score", None),
            drifts=drifts.get(t.id, []),
            is_fallback=(t.status == TuStatus.fallback),
        )
        if review_gate.assess(sig).flag:
            continue
        released.append({
            "job_id": job_id,
            "arxiv_id": aid,
            "unit_id": u.id,
            "ord": u.ord,
            "kind": u.kind.value if hasattr(u.kind, "value") else str(u.kind),
            "pass_no": t.pass_no,
            "chars_src": len(u.src_plain or ""),
            # 源/译都取含占位符的版本，使 {{PT_…}} 令牌两侧可逐字比对
            "src_text": u.src_text or u.src_plain or "",
            "tgt_text": t.tgt_text or t.tgt_plain or "",
        })
    return released


def _allocate(pools: dict[int, int], n: int) -> dict[int, int]:
    """Proportional allocation per job, largest-remainder rounding, capped at pool."""
    total = sum(pools.values())
    if total <= n:  # audit everything
        return dict(pools)
    raw = {j: n * c / total for j, c in pools.items()}
    alloc = {j: min(int(r), pools[j]) for j, r in raw.items()}
    rema = sorted(pools, key=lambda j: raw[j] - int(raw[j]), reverse=True)
    i = 0
    while sum(alloc.values()) < n:
        j = rema[i % len(rema)]
        if alloc[j] < pools[j]:
            alloc[j] += 1
        i += 1
    return alloc


async def amain(jobs: list[int], n: int, seed: int, out: Path) -> None:
    per_job: dict[int, list[dict]] = {}
    for j in jobs:
        per_job[j] = await _released_units(j)
        logger.info("job %s: %d released units", j, len(per_job[j]))

    pools = {j: len(v) for j, v in per_job.items()}
    total_released = sum(pools.values())
    alloc = _allocate(pools, n)
    rng = random.Random(seed)

    sample: list[dict] = []
    for j in jobs:
        units = sorted(per_job[j], key=lambda r: r["unit_id"])  # deterministic base order
        sample.extend(rng.sample(units, alloc[j]))
    rng.shuffle(sample)  # blind reviewers to per-paper grouping

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_no", "job_id", "arxiv_id", "unit_id", "ord", "kind", "pass_no",
            "chars_src", "src_text", "tgt_text", *JUDGE_COLS]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, row in enumerate(sample, 1):
            w.writerow({"sample_no": i, **row, **{c: "" for c in JUDGE_COLS}})

    print(f"released pool: {total_released} units across {len(jobs)} jobs")
    print("allocation   :", {j: alloc[j] for j in jobs})
    print(f"sampled      : {len(sample)} units  (seed={seed})")
    print(f"wrote        : {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=DEFAULT_JOBS, help="comma-separated job ids")
    ap.add_argument("--n", type=int, default=320, help="total sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("results/triage_analysis/audit_sample.csv"))
    a = ap.parse_args()
    jobs = [int(x) for x in a.jobs.split(",") if x.strip()]
    asyncio.run(amain(jobs, a.n, a.seed, a.out))


if __name__ == "__main__":
    main()
