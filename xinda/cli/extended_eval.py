"""Backfill the extended evaluation metrics onto an existing finished job.

The 12-stage pipeline's Evaluate stage only fills COMET/xCOMET + PPA/MFR. The
`evaluation_runs` row already has columns for the rest of the v3 metric suite
(`tcr`, `fps_paper`, `fps_<type>`, `rcs_paper`, `drift_*_count`, `total_claims`).
This CLI computes those and UPDATEs them in place — plus persists the judge
sampling tables (`eval_samples`/`eval_judgments`/`eval_mqm_errors`) and the RCS
tables (`comprehension_qa`/`comprehension_responses`).

    python -m xinda.cli.extended_eval <job_id> [--lang zh]
        [--rcs] [--judges] [--judge-sample N]

Default (no flags) runs only the zero-cost pure-function metrics: FPS aggregation
+ per-type breakdown + drift counts, and TCR. `--rcs` and `--judges` add the
LLM-driven metrics (cost quota).
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.engine import async_session
from xinda.db.models import (
    ClaimVerification,
    ClaimType,
    DriftType,
    EvalJudgment,
    EvalMqmError,
    EvalSample,
    EvaluationRun,
    GlossaryTerm,
    Translation,
    TranslationJob,
    TranslationUnit,
    TuStatus,
    VerifiableClaim,
)
from xinda.evaluation import fps as fps_mod
from xinda.evaluation import judge_geval, judge_rubric, rcs
from xinda.evaluation.tcr import compute_tcr
from xinda.logger_config import setup_logger
from xinda.translation import fact_anchors

logger = setup_logger(__name__)


# claim_type.value -> evaluation_runs.fps_<col>
_FPS_COL = {
    "numeric": "fps_numeric",
    "citation": "fps_citation",
    "comparison": "fps_comparison",
    "method_name": "fps_method",
    "symbol": "fps_symbol",
}
# drift.value -> evaluation_runs.drift_<col>
_DRIFT_COL = {
    "numeric_drift": "drift_numeric_count",
    "citation_swap": "drift_citation_count",
    "comparison_flip": "drift_comparison_count",
    "method_drift": "drift_method_count",
    "symbol_drift": "drift_symbol_count",
    "missing": "drift_missing_count",
}


async def _latest_tgt_by_unit(session: AsyncSession, job_id: int) -> dict[int, Translation]:
    rows = (
        await session.execute(
            select(Translation)
            .where(Translation.job_id == job_id)
            .order_by(Translation.unit_id, Translation.pass_no.desc())
        )
    ).scalars().all()
    latest: dict[int, Translation] = {}
    for t in rows:
        latest.setdefault(t.unit_id, t)
    return latest


async def reverify_anchors(session: AsyncSession, job_id: int) -> dict:
    """Apply the verbatim anchor-preservation override to existing
    claim_verifications WITHOUT re-calling the LLM. Flips false→verified where
    the claim's language-invariant value survives verbatim in the translation."""
    rows = (
        await session.execute(
            select(ClaimVerification, VerifiableClaim, Translation)
            .join(VerifiableClaim, ClaimVerification.claim_id == VerifiableClaim.id)
            .join(Translation, ClaimVerification.translation_id == Translation.id)
            .where(Translation.job_id == job_id)
        )
    ).all()
    flipped = 0
    for cvf, claim, tr in rows:
        if cvf.verified:
            continue
        rec = fact_anchors.ClaimRecord(
            claim_type=claim.claim_type,
            surface_form=claim.surface_form,
            normalized=claim.normalized,
            metadata=claim.claim_metadata or {},
        )
        if fact_anchors.anchor_preserved(rec, tr.tgt_plain or ""):
            cvf.verified = True
            cvf.drift = DriftType.verified
            cvf.drift_magnitude = 0.0
            flipped += 1
    # recompute per-translation fps_unit
    await session.flush()
    per_tr: dict[int, list[bool]] = {}
    for cvf, _claim, _tr in rows:
        per_tr.setdefault(cvf.translation_id, []).append(cvf.verified)
    tmap = {tr.id: tr for _, _, tr in rows}
    for tid, vs in per_tr.items():
        tmap[tid].fps_unit = sum(vs) / len(vs) if vs else 1.0
    await session.commit()
    return {"flipped": flipped, "total": len(rows)}


async def compute_fps(session: AsyncSession, job_id: int) -> dict:
    rows = (
        await session.execute(
            select(VerifiableClaim.claim_type, ClaimVerification.drift)
            .join(ClaimVerification, ClaimVerification.claim_id == VerifiableClaim.id)
            .join(Translation, ClaimVerification.translation_id == Translation.id)
            .where(Translation.job_id == job_id)
        )
    ).all()
    verifs: list[tuple[ClaimType, DriftType]] = [(ct, dr) for ct, dr in rows]
    return fps_mod.fps_from_verifications(verifs)


async def compute_tcr_for_job(
    session: AsyncSession, paper_id: int, language: str, latest: dict[int, Translation]
) -> dict:
    gterms = (
        await session.execute(
            select(GlossaryTerm).where(
                GlossaryTerm.paper_id == paper_id,
                GlossaryTerm.language == language,
            )
        )
    ).scalars().all()
    glossary = [
        {"src": g.src_term, "tgt": g.tgt_term, "kind": g.kind or ""} for g in gterms
    ]
    units = (
        await session.execute(
            select(TranslationUnit)
            .where(TranslationUnit.paper_id == paper_id)
            .order_by(TranslationUnit.ord)
        )
    ).scalars().all()
    src_texts, tgt_texts = [], []
    for u in units:
        t = latest.get(u.id)
        src_texts.append(u.src_plain or "")
        tgt_texts.append((t.tgt_plain if t else "") or "")
    if not glossary:
        return {"tcr": None, "term_count": 0}
    return compute_tcr(glossary, src_texts, tgt_texts)


async def run_judges(
    session: AsyncSession, job_id: int, paper_id: int, sample_n: int
) -> dict:
    """Sample N units, run RUBRIC-MQM (3-run median) + G-Eval on each."""
    # Drop empty samples left by a prior failed/interrupted run, then skip units
    # already judged so a re-run resumes instead of duplicating samples.
    await session.execute(
        delete(EvalSample).where(
            EvalSample.job_id == job_id,
            ~EvalSample.judgments.any(),
        )
    )
    await session.commit()
    already = set(
        (
            await session.execute(
                select(EvalSample.unit_id).where(EvalSample.job_id == job_id)
            )
        ).scalars().all()
    )

    latest = await _latest_tgt_by_unit(session, job_id)
    units = (
        await session.execute(
            select(TranslationUnit)
            .where(TranslationUnit.paper_id == paper_id)
            .order_by(TranslationUnit.ord)
        )
    ).scalars().all()
    # pick units with the longest source text (most to judge), up to sample_n
    judged = [
        u for u in units
        if u.id not in already
        and (t := latest.get(u.id)) is not None
        and t.status != TuStatus.pending
        and (t.tgt_plain or "")
        and len(u.src_plain or "") >= 40
    ]
    judged.sort(key=lambda u: len(u.src_plain or ""), reverse=True)
    judged = judged[:max(0, sample_n - len(already))]

    rubric_scores: list[float] = []
    geval_adeq: list[float] = []
    for u in judged:
        t = latest[u.id]
        rub = await judge_rubric.judge_one(u.src_plain, t.tgt_plain, runs=3)
        gev = await judge_geval.judge_one(u.src_plain, t.tgt_plain)

        sample = EvalSample(job_id=job_id, unit_id=u.id, sampling_kind="longest")
        session.add(sample)
        await session.flush()  # get sample.id

        if rub.get("valid"):
            # Store EACH run as its own judgment row (run_no=1,2,3) so meta-eval's
            # self-consistency check has the per-run spread; check_1/3 average per
            # sample, so multiple rows are fine there too.
            runs = [r for r in (rub.get("raw_runs") or [])
                    if r and r.get("rubric_score") is not None]
            j = None
            for ri, run in enumerate(runs, start=1):
                sc = run.get("scores") or {}
                j = EvalJudgment(
                    sample_id=sample.id, judge_model="qianwen",
                    protocol="rubric_mqm", run_no=ri,
                    rubric_score=run.get("rubric_score"),
                    fluency=sc.get("fluency"),
                    terminology=sc.get("terminology"),
                    structure=sc.get("structure"),
                    raw_response=run,
                )
                session.add(j)
            await session.flush()
            if j is not None:  # attach aggregated MQM errors to the last run row
                for e in (rub.get("errors") or []):
                    session.add(EvalMqmError(
                        judgment_id=j.id, category=e.get("category"),
                        severity=e.get("severity"), span_text=e.get("span_text"),
                        explanation=e.get("explanation"),
                    ))
            if rub.get("rubric_score") is not None:
                rubric_scores.append(rub["rubric_score"])
        if gev.get("valid"):
            session.add(EvalJudgment(
                sample_id=sample.id, judge_model="qianwen",
                protocol="g_eval", run_no=1,
                fluency=gev.get("fluency"), adequacy=gev.get("adequacy"),
                terminology=gev.get("terminology"), structure=gev.get("structure"),
                raw_response=gev.get("raw"),
            ))
            if gev.get("adequacy") is not None:
                geval_adeq.append(gev["adequacy"])
        await session.commit()
        logger.info("judged unit %d (rubric=%s geval_adeq=%s)",
                    u.id, rub.get("rubric_score"), gev.get("adequacy"))

    import statistics
    return {
        "judged_units": len(judged),
        "rubric_median": statistics.median(rubric_scores) if rubric_scores else None,
        "geval_adequacy_mean": statistics.mean(geval_adeq) if geval_adeq else None,
    }


async def amain(job_id: int, language: str, do_rcs: bool, do_judges: bool,
                judge_sample: int, do_reverify: bool) -> None:
    async with async_session() as session:
        job = (
            await session.execute(select(TranslationJob).where(TranslationJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise SystemExit(f"job {job_id} not found")
        paper_id = job.paper_id
        lang = language or job.language

        run = (
            await session.execute(
                select(EvaluationRun).where(EvaluationRun.job_id == job_id)
            )
        ).scalar_one_or_none()
        if run is None:
            run = EvaluationRun(job_id=job_id)
            session.add(run)

        # ---- reverify anchors (free, no LLM): fix cross-lingual false-misses ----
        if do_reverify:
            rv = await reverify_anchors(session, job_id)
            print(f"reverify: flipped {rv['flipped']}/{rv['total']} "
                  f"false-miss → verified (verbatim anchor present)")

        # ---- FPS aggregation (free) ----
        fps_agg = await compute_fps(session, job_id)
        run.fps_paper = fps_agg["fps"]
        run.total_claims = fps_agg["total"]
        # Always set (None when a claim type produced no claims this run) so the
        # exported evaluation_runs row never carries stale per-type values.
        for ctype_val, col in _FPS_COL.items():
            setattr(run, col, fps_agg["per_type"].get(ctype_val))
        for drift_val, col in _DRIFT_COL.items():
            setattr(run, col, fps_agg["drift_counts"].get(drift_val, 0))

        # ---- TCR (free) ----
        latest = await _latest_tgt_by_unit(session, job_id)
        tcr_res = await compute_tcr_for_job(session, paper_id, lang, latest)
        run.tcr = tcr_res.get("tcr")
        await session.commit()

        print("─" * 60)
        print(f"job {job_id}  paper {paper_id}  lang {lang}")
        print(f"FPS_paper        = {_fmt(run.fps_paper)}   (over {run.total_claims} claims)")
        for ctype_val, col in _FPS_COL.items():
            print(f"  fps[{ctype_val:11}] = {_fmt(getattr(run, col))}")
        print("drift counts     = " + ", ".join(
            f"{k.replace('drift_','').replace('_count','')}:{getattr(run, k)}"
            for k in _DRIFT_COL.values()
        ))
        print(f"TCR              = {_fmt(run.tcr)}   (over {tcr_res.get('term_count')} terms)")

        # ---- RCS (LLM) ----
        if do_rcs:
            n_qa = await rcs.generate_qa_for_paper(session, paper_id)
            rcs_res = await rcs.score_translation(session, job_id, paper_id)
            run.rcs_paper = rcs_res.get("rcs_paper")
            await session.commit()
            print(f"RCS_paper        = {_fmt(run.rcs_paper)}   "
                  f"(qa={n_qa}, answered={rcs_res.get('answered')})")

        # ---- Judges (LLM) ----
        if do_judges:
            jr = await run_judges(session, job_id, paper_id, judge_sample)
            print(f"RUBRIC-MQM med   = {_fmt(jr['rubric_median'])}   "
                  f"(1-5 scale, {jr['judged_units']} units)")
            print(f"G-Eval adequacy  = {_fmt(jr['geval_adequacy_mean'])}   (1-5 scale)")
        print("─" * 60)


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", type=int)
    ap.add_argument("--lang", default="")
    ap.add_argument("--rcs", action="store_true", help="run Reader Comprehension Score (LLM)")
    ap.add_argument("--judges", action="store_true", help="run RUBRIC-MQM + G-Eval (LLM)")
    ap.add_argument("--judge-sample", type=int, default=8,
                    help="number of units to judge (default 8)")
    ap.add_argument("--reverify", action="store_true",
                    help="apply verbatim anchor override to existing claim_verifications (no LLM)")
    a = ap.parse_args()
    asyncio.run(amain(a.job_id, a.lang, a.rcs, a.judges, a.judge_sample, a.reverify))


if __name__ == "__main__":
    main()
