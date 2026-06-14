"""Selective human-review triage for one job (the cost-reduction mechanism).

Reads the automated QE signals already in the DB (no LLM calls), runs the quality
gate per unit, and reports how many segments a human must actually review — plus
the resulting cost vs. translating/reviewing the whole paper by hand.

  python -m xinda.cli.review_triage <job_id> [--human-rate 400] [--top 15]

`--human-rate` = CNY per 1000 target chars for human translation/review
(public quotes for en→zh sci-translation are ~300–500). The gated cost reviews
only the flagged segments; the rest are auto-accepted.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from xinda.db.engine import async_session
from xinda.db.models import (
    ClaimVerification,
    DriftType,
    EvaluationRun,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.evaluation import review_gate
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)


async def amain(job_id: int, human_rate: float, top: int) -> None:
    async with async_session() as session:
        rows = (
            await session.execute(
                select(Translation, TranslationUnit)
                .join(TranslationUnit, Translation.unit_id == TranslationUnit.id)
                .where(Translation.job_id == job_id,
                       Translation.status != TuStatus.pending)
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).all()
        if not rows:
            raise SystemExit(f"no translations for job {job_id}")
        latest: dict[int, tuple[Translation, TranslationUnit]] = {}
        for t, u in rows:
            latest.setdefault(t.unit_id, (t, u))

        # drift types per translation (non-verified)
        tr_ids = [t.id for t, _ in latest.values()]
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

        ev = (
            await session.execute(select(EvaluationRun).where(EvaluationRun.job_id == job_id))
        ).scalar_one_or_none()

    # run the gate per unit
    assessed: list[tuple[TranslationUnit, Translation, review_gate.GateResult]] = []
    for t, u in latest.values():
        sig = review_gate.UnitSignals(
            fps_unit=t.fps_unit,
            xcomet_score=t.xcomet_score,
            comet_score=t.comet_score,
            rubric_score=getattr(t, "rubric_score", None),
            geval_score=getattr(t, "geval_score", None),
            drifts=drifts.get(t.id, []),
            is_fallback=(t.status == TuStatus.fallback),
        )
        assessed.append((u, t, review_gate.assess(sig)))

    summary = review_gate.summarize([r for _, _, r in assessed])

    # cost story
    total_chars = sum(len(u.src_plain or "") for u, _, _ in assessed)
    flagged_chars = sum(len(u.src_plain or "") for u, _, r in assessed if r.flag)
    full_human = total_chars / 1000 * human_rate
    gated_human = flagged_chars / 1000 * human_rate
    llm_cost = (ev.cost_cny if ev and ev.cost_cny else 0.0)

    print("\n" + "=" * 70)
    print(f"Review triage  job={job_id}")
    print("-" * 70)
    print(f"segments               : {summary.total}")
    print(f"flagged for review     : {summary.flagged}  ({summary.flag_rate*100:.1f}%)")
    print(f"auto-accepted          : {summary.total - summary.flagged}  "
          f"({(1-summary.flag_rate)*100:.1f}%)")
    print("flag reasons           :")
    for reason, c in summary.reason_counts.items():
        print(f"    {reason:24} {c}")
    print("-" * 70)
    print(f"cost @ ¥{human_rate:g}/1000 chars (human):")
    print(f"  full human translation : ¥{full_human:.1f}  ({total_chars} chars)")
    print(f"  our pipeline + gated review: ¥{llm_cost:.2f} (LLM) + ¥{gated_human:.1f} "
          f"(review {flagged_chars} chars) = ¥{llm_cost+gated_human:.1f}")
    if full_human > 0:
        saving = (1 - (llm_cost + gated_human) / full_human) * 100
        print(f"  → cost reduction        : {saving:.1f}%")
    print("-" * 70)
    print(f"top {top} riskiest segments (review queue):")
    ranked = sorted(assessed, key=lambda x: -x[2].risk_score)
    for u, _t, r in ranked[:top]:
        if not r.flag:
            break
        print(f"  unit#{u.ord:<4} risk={r.risk_score:.2f}  {', '.join(r.reasons)}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", type=int)
    ap.add_argument("--human-rate", type=float, default=400.0, help="CNY per 1000 chars")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    asyncio.run(amain(a.job_id, a.human_rate, a.top))


if __name__ == "__main__":
    main()
