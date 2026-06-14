"""Meta-evaluation: prove that LLM-as-judge results are trustworthy.

Four required checks (any failure → document as limitation):
1. Judge-vs-COMET correlation: Pearson r between RUBRIC-MQM rubric_score
   and xCOMET on the same samples. Threshold: r ≥ 0.5.
2. Judge self-consistency: re-run RUBRIC-MQM 3+ times on a 200-sample
   subset; report intra-judge Krippendorff α. Threshold: α ≥ 0.7.
3. Inter-judge agreement: between RUBRIC-MQM and G-Eval on the same
   samples (or two model sizes of the same judge). Threshold: α ≥ 0.5.
4. External calibration: against WMT24 General MT public human DA scores
   on a 30-50 sample subset. Threshold: r ≥ 0.4.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    EvalJudgment,
    EvalSample,
    Translation,
)
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation; None if degenerate."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    from scipy.stats import spearmanr  # noqa: PLC0415
    r, _ = spearmanr(xs, ys)
    if np.isnan(r):
        return None
    return float(r)


def krippendorff_alpha(units: list[list[float]]) -> float | None:
    """`units[i]` = list of ratings for unit i (different raters)."""
    try:
        import krippendorff  # noqa: PLC0415
    except ImportError:
        logger.warning("krippendorff not installed; skipping α")
        return None
    if not units or not any(len(u) >= 2 for u in units):
        return None
    # build the (raters × units) matrix; pad missing with nan
    max_raters = max(len(u) for u in units)
    matrix = []
    for r in range(max_raters):
        row = []
        for u in units:
            row.append(u[r] if r < len(u) else np.nan)
        matrix.append(row)
    try:
        return float(krippendorff.alpha(
            reliability_data=matrix, level_of_measurement="interval",
        ))
    except Exception as e:
        logger.warning("krippendorff alpha failed: %s", e)
        return None


# ────────────────────────── checks ──────────────────────────


async def judge_vs_xcomet_correlation(
    session: AsyncSession, language: str | None = None,
) -> dict:
    """Check 1: r between RUBRIC-MQM rubric_score and xCOMET on shared samples."""
    rubric_q = (
        select(EvalSample.unit_id, EvalJudgment.rubric_score)
        .join(EvalJudgment, EvalJudgment.sample_id == EvalSample.id)
        .where(EvalJudgment.protocol == "rubric_mqm")
    )
    rubric_rows = (await session.execute(rubric_q)).all()
    rubric_by_unit: dict[int, list[float]] = {}
    for unit_id, s in rubric_rows:
        if s is not None:
            rubric_by_unit.setdefault(unit_id, []).append(float(s))

    # xCOMET per unit comes from translations
    tr_rows = (
        await session.execute(
            select(Translation.unit_id, Translation.xcomet_score)
            .where(Translation.xcomet_score.isnot(None))
        )
    ).all()
    xcomet_by_unit: dict[int, list[float]] = {}
    for unit_id, x in tr_rows:
        xcomet_by_unit.setdefault(unit_id, []).append(float(x))

    xs: list[float] = []
    ys: list[float] = []
    for unit_id, rs in rubric_by_unit.items():
        if unit_id not in xcomet_by_unit:
            continue
        xs.append(sum(rs) / len(rs))
        ys.append(sum(xcomet_by_unit[unit_id]) / len(xcomet_by_unit[unit_id]))

    r = pearson(xs, ys)
    rho = spearman(xs, ys)
    return {
        "n": len(xs), "pearson": r, "spearman": rho,
        "passes_threshold": (r is not None and r >= 0.5),
    }


async def judge_self_consistency(session: AsyncSession) -> dict:
    """Check 2: re-runs of same RUBRIC-MQM judge on same samples (run_no > 1)."""
    rows = (
        await session.execute(
            select(EvalJudgment.sample_id, EvalJudgment.rubric_score)
            .where(EvalJudgment.protocol == "rubric_mqm")
            .order_by(EvalJudgment.sample_id, EvalJudgment.run_no)
        )
    ).all()
    by_sample: dict[int, list[float]] = {}
    for sid, s in rows:
        if s is not None:
            by_sample.setdefault(sid, []).append(float(s))
    repeated = [v for v in by_sample.values() if len(v) >= 2]
    if not repeated:
        return {"n": 0, "alpha": None, "passes_threshold": False}
    alpha = krippendorff_alpha(repeated)
    return {
        "n": len(repeated), "alpha": alpha,
        "passes_threshold": (alpha is not None and alpha >= 0.7),
    }


async def inter_judge_agreement(session: AsyncSession) -> dict:
    """Check 3: RUBRIC-MQM rubric_score vs G-Eval mean(4 dims) per sample."""
    rubric_rows = (
        await session.execute(
            select(EvalJudgment.sample_id, EvalJudgment.rubric_score)
            .where(EvalJudgment.protocol == "rubric_mqm")
        )
    ).all()
    rubric_by_sample: dict[int, float] = {}
    for sid, s in rubric_rows:
        if s is not None:
            rubric_by_sample[sid] = float(s)

    geval_rows = (
        await session.execute(
            select(EvalJudgment)
            .where(EvalJudgment.protocol == "g_eval")
        )
    ).scalars().all()
    geval_by_sample: dict[int, float] = {}
    for r in geval_rows:
        vals = [r.fluency, r.adequacy, r.terminology, r.structure]
        vals = [v for v in vals if v is not None]
        if vals:
            geval_by_sample[r.sample_id] = sum(vals) / len(vals)

    units = []
    for sid in rubric_by_sample.keys() & geval_by_sample.keys():
        units.append([rubric_by_sample[sid], geval_by_sample[sid]])
    alpha = krippendorff_alpha(units)
    return {
        "n": len(units), "alpha": alpha,
        "passes_threshold": (alpha is not None and alpha >= 0.5),
    }


def wmt24_calibration(
    judge_scores: list[float], wmt_human_scores: list[float],
) -> dict:
    """Check 4: r between judge scores on WMT24 sample and the published human DA.

    This is a free function (no DB) because the WMT24 evaluation set is
    independent of our pipeline. Loader code for the dataset is the
    caller's responsibility.
    """
    r = pearson(judge_scores, wmt_human_scores)
    rho = spearman(judge_scores, wmt_human_scores)
    return {
        "n": len(judge_scores), "pearson": r, "spearman": rho,
        "passes_threshold": (r is not None and r >= 0.4),
    }


async def full_report(session: AsyncSession) -> dict:
    """Run all four checks; return summary dict suitable for the paper."""
    return {
        "check_1_judge_vs_xcomet": await judge_vs_xcomet_correlation(session),
        "check_2_self_consistency": await judge_self_consistency(session),
        "check_3_inter_judge": await inter_judge_agreement(session),
        "note": (
            "Check 4 (WMT24 calibration) requires external data; run "
            "`python -m xinda.cli.meta_eval --wmt24 <path>`."
        ),
    }
