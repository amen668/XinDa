"""Matrix runner: papers × languages × variants → CSV + LaTeX tables.

Reads finished translation_jobs + evaluation_runs from DB, pivots into the
standard paper-evaluation tables expected by Tables 1, 2, 3 of the paper.
Also exposes `run_matrix` to schedule the full ablation+baseline sweep.
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import (
    EvaluationRun,
    JobStatus,
    Paper,
    PipelineStage,
    RunVariant,
    TranslationJob,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.config import variants_for
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import Orchestrator
from xinda.pipeline.stages.acquire import Acquire
from xinda.pipeline.stages.apply import ApplyXML
from xinda.pipeline.stages.convert import Convert
from xinda.pipeline.stages.coherence import Coherence
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


def full_stage_list() -> list:
    """The canonical 13-stage pipeline (idempotent; ablation toggles skip).

    Coherence (whole-doc discourse harmonization) sits after Refine and before
    ApplyXML so its harmonized passes are what get written into the XML.
    """
    return [
        Acquire(), Convert(), Extract(),
        FactExtract(), GlossaryBuild(),
        FirstPassTranslate(), FactVerify(), CrossDocFactVerify(),
        Refine(), Coherence(), ApplyXML(), Render(), Evaluate(),
    ]


# ────────────────────────── runner ──────────────────────────


async def run_one(arxiv_id: str, language: str, variant: str) -> int | None:
    """Run a single (paper, lang, variant) job. Returns job_id."""
    cfg = variants_for(language).get(variant)  # type: ignore[arg-type]
    if cfg is None:
        raise ValueError(f"unknown variant: {variant}")

    workspace = (
        settings.workspace_dir
        / arxiv_id
        / f"{language}_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

        # An existing job shares the (paper, lang, variant, config_hash) unique key.
        # Reuse it ONLY if it ran the FULL pipeline (last_stage == evaluate) — a job
        # left at an earlier stage (e.g. a screening run that only did Acquire→Extract
        # also marks status=success) must be RESUMED, not skipped. If we created a new
        # row here it would violate the unique key anyway, so resume the existing one.
        existing = (
            await session.execute(
                select(TranslationJob).where(
                    TranslationJob.paper_id == paper.id,
                    TranslationJob.language == language,
                    TranslationJob.variant == RunVariant(variant),
                    TranslationJob.config_hash == cfg.config_hash(),
                )
            )
        ).scalar_one_or_none()
        if existing is not None and existing.last_stage == PipelineStage.evaluate:
            logger.info(
                "reuse completed job %d for (%s, %s, %s)",
                existing.id, arxiv_id, language, variant,
            )
            return existing.id
        if existing is not None:
            logger.info(
                "resume incomplete job %d (last_stage=%s) for (%s, %s, %s)",
                existing.id, existing.last_stage, arxiv_id, language, variant,
            )
            paper_id, job_id = paper.id, existing.id
        else:
            job = TranslationJob(
                paper_id=paper.id,
                language=language,
                provider="qianwen",
                model_name=cfg.pass1_model,
                refine_model=cfg.refine_model,
                variant=RunVariant(variant),
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
        arxiv_id=arxiv_id, paper_id=paper_id, job_id=job_id,
        workspace=workspace, config=cfg,
    )
    try:
        await Orchestrator(stages=full_stage_list()).run(ctx)
        return job_id
    except Exception as e:
        logger.exception("job %d failed: %s", job_id, e)
        return job_id


async def run_matrix(
    arxiv_ids: Iterable[str],
    languages: Iterable[str],
    variants: Iterable[str],
    *,
    concurrency: int = 2,
) -> list[int]:
    """Schedule a full matrix with paper-level concurrency control."""
    sem = asyncio.Semaphore(concurrency)

    async def worker(arxiv_id: str, lang: str, var: str) -> int | None:
        async with sem:
            return await run_one(arxiv_id, lang, var)

    tasks = [
        worker(a, l, v)
        for a in arxiv_ids for l in languages for v in variants
    ]
    return [j for j in await asyncio.gather(*tasks) if j is not None]


# ────────────────────────── export ──────────────────────────


CSV_COLUMNS = [
    "arxiv_id", "language", "variant", "config_hash",
    "status", "comet_mean", "comet_median", "comet_p10",
    "xcomet_mean", "ppa", "ppa_ordered", "mfr", "mfr_ordered", "tcr",
    "fps_paper", "rcs_paper",
    "pass1_units", "refined_units", "fallback_units", "total_units", "total_claims",
    "total_prompt_tok", "total_cached_tok", "total_completion_tok", "cost_cny",
    "wallclock_sec",
]


async def export_csv(out_path: Path) -> None:
    """Dump one row per (paper, lang, variant) job with its evaluation_run."""
    async with async_session() as session:
        rows = (
            await session.execute(
                select(Paper.arxiv_id, TranslationJob, EvaluationRun)
                .join(TranslationJob, TranslationJob.paper_id == Paper.id)
                .outerjoin(EvaluationRun, EvaluationRun.job_id == TranslationJob.id)
            )
        ).all()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for arxiv_id, job, ev in rows:
            w.writerow({
                "arxiv_id": arxiv_id,
                "language": job.language,
                "variant": job.variant.value,
                "config_hash": job.config_hash,
                "status": job.status.value,
                "comet_mean": ev.comet_mean if ev else None,
                "comet_median": ev.comet_median if ev else None,
                "comet_p10": ev.comet_p10 if ev else None,
                "xcomet_mean": ev.xcomet_mean if ev else None,
                "ppa": ev.ppa if ev else None,
                "ppa_ordered": ev.ppa_ordered if ev else None,
                "mfr": ev.mfr if ev else None,
                "mfr_ordered": ev.mfr_ordered if ev else None,
                "tcr": ev.tcr if ev else None,
                "fps_paper": ev.fps_paper if ev else None,
                "rcs_paper": ev.rcs_paper if ev else None,
                "pass1_units": ev.pass1_units if ev else None,
                "refined_units": ev.refined_units if ev else None,
                "fallback_units": ev.fallback_units if ev else None,
                "total_units": ev.total_units if ev else None,
                "total_claims": ev.total_claims if ev else None,
                "total_prompt_tok": ev.total_prompt_tok if ev else None,
                "total_cached_tok": ev.total_cached_tok if ev else None,
                "total_completion_tok": ev.total_completion_tok if ev else None,
                "cost_cny": ev.cost_cny if ev else None,
                "wallclock_sec": ev.wallclock_sec if ev else None,
            })

    await export_cost_summary(out_path.with_name("cost_summary.csv"))


async def export_cost_summary(out_path: Path) -> None:
    """Per-language and per-variant mean translation cost (the降本 evidence)."""
    async with async_session() as session:
        rows = (
            await session.execute(
                select(TranslationJob.language, TranslationJob.variant, EvaluationRun.cost_cny)
                .join(EvaluationRun, EvaluationRun.job_id == TranslationJob.id)
                .where(EvaluationRun.cost_cny.isnot(None))
            )
        ).all()
    by_lang: dict[str, list[float]] = {}
    by_variant: dict[str, list[float]] = {}
    for lang, variant, c in rows:
        by_lang.setdefault(lang, []).append(c)
        by_variant.setdefault(variant.value, []).append(c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "key", "n_jobs", "mean_cost_cny", "total_cost_cny"])
        for k, cs in sorted(by_lang.items()):
            w.writerow(["language", k, len(cs), round(sum(cs) / len(cs), 6), round(sum(cs), 6)])
        for k, cs in sorted(by_variant.items()):
            w.writerow(["variant", k, len(cs), round(sum(cs) / len(cs), 6), round(sum(cs), 6)])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


async def _variant_lang_groups() -> dict[tuple[str, str], list[EvaluationRun]]:
    """All successful jobs' EvaluationRuns grouped by (variant, language)."""
    async with async_session() as session:
        rows = (
            await session.execute(
                select(TranslationJob.variant, TranslationJob.language, EvaluationRun)
                .join(EvaluationRun, EvaluationRun.job_id == TranslationJob.id)
                .where(TranslationJob.status == JobStatus.success)
            )
        ).all()
    agg: dict[tuple[str, str], list[EvaluationRun]] = {}
    for variant, lang, ev in rows:
        agg.setdefault((variant.value, lang), []).append(ev)
    return agg


def _emit_table(
    agg: dict[tuple[str, str], list[EvaluationRun]],
    languages: list[str],
    caption: str,
    col_headers: list[str],
    cell_fns: list,
) -> str:
    """Render a variant × (lang × metrics) LaTeX table from grouped EvaluationRuns."""
    ncol = len(col_headers)
    n_papers = max((len(v) for v in agg.values()), default=0)
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption} (n$\approx${n_papers} papers per variant).}}",
        r"\begin{tabular}{l" + "c" * (len(languages) * ncol) + "}",
        r"\toprule",
        r"Variant " + " & ".join(
            f"\\multicolumn{{{ncol}}}{{c}}{{{lang}}}" for lang in languages
        ) + r" \\",
        r" & " + " & ".join(" & ".join(col_headers) for _ in languages) + r" \\",
        r"\midrule",
    ]
    for v in sorted({vv for vv, _ in agg.keys()}):
        cells = [v.replace("_", r"\_")]
        for lang in languages:
            evs = agg.get((v, lang), [])
            cells.extend(fn(evs) for fn in cell_fns)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


async def export_latex_main_table(out_path: Path, languages: list[str]) -> None:
    """Headline results: one column per thesis dimension, no redundant metrics.

    Structure (PPA/MFR) · fact fidelity (FPS) · neural QE (COMET-Kiwi) · reader
    comprehension (RCS) · cost (¥/paper). Redundant / hard-to-calibrate metrics
    (xCOMET, ordered variants, TCR) live in the appendix table; LLM judges live in
    the meta-eval. See export_latex_appendix_table."""
    agg = await _variant_lang_groups()
    table = _emit_table(
        agg, languages,
        caption="Main results: per-variant means by thesis dimension",
        col_headers=["PPA", "MFR", "COMET", "FPS", "RCS", r"Cost¥"],
        cell_fns=[
            lambda e: f"{_mean([x.ppa for x in e]):.1f}",
            lambda e: f"{_mean([x.mfr for x in e]):.1f}",
            lambda e: f"{_mean([x.comet_mean for x in e]):.3f}",
            lambda e: f"{_mean([x.fps_paper for x in e]):.3f}",
            lambda e: f"{_mean([x.rcs_paper for x in e]):.3f}",
            lambda e: f"{_mean([x.cost_cny for x in e]):.3f}",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")


async def export_latex_appendix_table(out_path: Path, languages: list[str]) -> None:
    """Robustness / cross-check table: the secondary metrics demoted from the
    headline — xCOMET (a second neural QE, miscalibrated on dense scientific text →
    used to cross-validate COMET/judges in meta-eval, not as a headline), the ordered
    structure variants (Kendall tau; near-saturated once PPA/MFR are content-keyed),
    and TCR (terminology consistency; a near-1.0 no-regression guard)."""
    agg = await _variant_lang_groups()
    table = _emit_table(
        agg, languages,
        caption="Robustness / cross-check metrics (secondary)",
        col_headers=["xCOMET", "PPA-ord", "MFR-ord", "TCR"],
        cell_fns=[
            lambda e: f"{_mean([x.xcomet_mean for x in e]):.3f}",
            lambda e: f"{_mean([x.ppa_ordered for x in e]):.1f}",
            lambda e: f"{_mean([x.mfr_ordered for x in e]):.1f}",
            lambda e: f"{_mean([x.tcr for x in e]):.3f}",
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table, encoding="utf-8")
