"""Quality gate for selective human-in-the-loop review (the headline mechanism).

Full-text multilingual publishing is unaffordable if a human must check every
segment, and unsafe if none are checked. The pipeline already produces automated,
reference-free quality signals per unit (FPS / neural-QE / LLM-judge / fact-drift);
this module fuses them into a transparent **risk score + review flag** so a human
only verifies the segments the system is unsure about.

The gate is deliberately **rule-based and explainable** (every flag carries its
reasons) — important for a publishing workflow and for the paper. A unit is flagged
when ANY hard signal fires; the continuous `risk_score` is for ranking the review
queue. All signals are optional (NULL-tolerant): the gate uses whatever the eval
suite has computed for that job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Drift types that are genuine factual errors (must be reviewed if present).
_ERROR_DRIFTS = {
    "numeric_drift", "citation_swap", "comparison_flip",
    "method_drift", "symbol_drift", "missing",
}


@dataclass
class UnitSignals:
    """Per-unit automated quality signals (any may be None if not computed)."""
    fps_unit: float | None = None
    xcomet_score: float | None = None
    comet_score: float | None = None
    rubric_score: float | None = None      # 1-5
    geval_score: float | None = None       # 1-5
    drifts: list[str] = field(default_factory=list)  # non-verified drift types
    is_fallback: bool = False              # translation fell back (failed)


@dataclass
class GateResult:
    flag: bool
    risk_score: float          # 0 (safe) … 1 (risky), for ranking the queue
    reasons: list[str]


def assess(
    sig: UnitSignals,
    *,
    fps_threshold: float = 0.95,
    xcomet_threshold: float = 0.25,
    judge_threshold: float = 3.0,   # 1-5 scale; below this is risky
) -> GateResult:
    """Decide whether a unit needs human review + a risk score for ranking."""
    reasons: list[str] = []
    risks: list[float] = []

    if sig.is_fallback:
        reasons.append("translation_fallback")
        risks.append(1.0)

    if sig.fps_unit is not None and sig.fps_unit < fps_threshold:
        reasons.append(f"fps<{fps_threshold:g} ({sig.fps_unit:.2f})")
        risks.append(1.0 - sig.fps_unit)

    if sig.xcomet_score is not None and sig.xcomet_score < xcomet_threshold:
        reasons.append(f"xcomet<{xcomet_threshold:g} ({sig.xcomet_score:.2f})")
        # normalise distance below threshold into [0,1]
        risks.append(min(1.0, (xcomet_threshold - sig.xcomet_score) / max(xcomet_threshold, 1e-6)))

    for j, name in ((sig.rubric_score, "rubric"), (sig.geval_score, "geval")):
        if j is not None and j < judge_threshold:
            reasons.append(f"{name}<{judge_threshold:g} ({j:.1f})")
            risks.append((judge_threshold - j) / 5.0)

    err = [d for d in sig.drifts if d in _ERROR_DRIFTS]
    if err:
        reasons.append("fact_drift:" + ",".join(sorted(set(err))))
        risks.append(1.0)

    risk_score = max(risks) if risks else 0.0
    return GateResult(flag=bool(reasons), risk_score=risk_score, reasons=reasons)


@dataclass
class TriageSummary:
    total: int
    flagged: int
    flag_rate: float
    reason_counts: dict[str, int]


def summarize(results: list[GateResult]) -> TriageSummary:
    total = len(results)
    flagged = sum(1 for r in results if r.flag)
    counts: dict[str, int] = {}
    for r in results:
        for reason in r.reasons:
            key = reason.split(":")[0].split("<")[0].split(" ")[0]  # coarse bucket
            counts[key] = counts.get(key, 0) + 1
    return TriageSummary(
        total=total,
        flagged=flagged,
        flag_rate=(flagged / total) if total else 0.0,
        reason_counts=dict(sorted(counts.items(), key=lambda x: -x[1])),
    )
