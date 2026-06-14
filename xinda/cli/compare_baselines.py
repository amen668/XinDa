"""Differentiation comparison: ours vs naive-LLM vs Google vs abstract-only.

Produces the paper's headline four-dimensional table on ONE paper:
  structure preservation (placeholder/math) · coverage · fidelity (FPS) · cost.

  python -m xinda.cli.compare_baselines 2503.15129 zh [--job N] [--google] [--out results/]

Our pipeline keeps every inline math/citation placeholder; a translator without
the contract corrupts them. The naive-LLM baseline (same model, plain prompt) is
the controlled comparison; Google is best-effort (free endpoint often 429s);
abstract-only contributes the coverage dimension (status quo of journal practice).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from sqlalchemy import select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import (
    EvaluationRun,
    JobStatus,
    Paper,
    Translation,
    TranslationJob,
    TranslationUnit,
)
from xinda.evaluation import baselines, cost
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider

logger = setup_logger(__name__)


async def _load(session, arxiv_id: str, language: str, job_id: int | None):
    paper = (
        await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
    ).scalar_one_or_none()
    if paper is None:
        raise SystemExit(f"paper {arxiv_id} not found; run the pipeline first")
    units = (
        await session.execute(
            select(TranslationUnit)
            .where(TranslationUnit.paper_id == paper.id)
            .order_by(TranslationUnit.ord)
        )
    ).scalars().all()

    if job_id is None:
        job = (
            await session.execute(
                select(TranslationJob)
                .where(
                    TranslationJob.paper_id == paper.id,
                    TranslationJob.language == language,
                    TranslationJob.status == JobStatus.success,
                )
                .order_by(TranslationJob.id.desc())
            )
        ).scalars().first()
    else:
        job = (
            await session.execute(select(TranslationJob).where(TranslationJob.id == job_id))
        ).scalar_one_or_none()
    if job is None:
        raise SystemExit("no successful 'ours' job for this paper/lang; run translate_smoke first")

    trans = (
        await session.execute(
            select(Translation).where(Translation.job_id == job.id)
            .order_by(Translation.unit_id, Translation.pass_no.desc())
        )
    ).scalars().all()
    ours: dict[int, str] = {}
    for t in trans:
        ours.setdefault(t.unit_id, t.tgt_text or "")

    ev = (
        await session.execute(select(EvaluationRun).where(EvaluationRun.job_id == job.id))
    ).scalar_one_or_none()
    return paper, units, job, ours, ev


async def amain(arxiv_id: str, language: str, job_id: int | None, try_google: bool,
                out_dir: str | None) -> None:
    async with async_session() as session:
        paper, units, job, ours_map, ev = await _load(session, arxiv_id, language, job_id)

    rows: list[dict] = []

    # ── ours (from DB) ──
    ours_pres = baselines.aggregate_preservation(
        [(u.src_text, ours_map.get(u.id)) for u in units]
    )
    rows.append({
        "system": "ours (pipeline)",
        "placeholder_rate": ours_pres["placeholder_rate"],
        "math_rate": ours_pres["math_rate"],
        "coverage": 1.0,
        "fps": ev.fps_paper if ev else None,
        "ppa": ev.ppa if ev else None,
        "mfr": ev.mfr if ev else None,
        "cost_cny": ev.cost_cny if ev else None,
    })

    # ── naive LLM (same model, plain prompt — no placeholder contract) ──
    provider = create_provider(settings.model_first_pass)
    sem = asyncio.Semaphore(settings.max_concurrency)

    async def naive_one(u: TranslationUnit):
        async with sem:
            try:
                r = await baselines.naive_translate(provider, u.src_text, language)
                return u, r
            except Exception as e:  # noqa: BLE001
                logger.warning("naive translate failed unit %d: %s", u.id, e)
                return u, None

    naive_res = await asyncio.gather(*(naive_one(u) for u in units))
    naive_pairs = [(u.src_text, (r.text if r else "")) for u, r in naive_res]
    naive_pres = baselines.aggregate_preservation(naive_pairs)
    naive_cost = sum(
        cost.cost_cny(
            fresh_prompt_tok=r.fresh_prompt_tokens, cached_prompt_tok=r.cached_prompt_tokens,
            completion_tok=r.completion_tokens, model_name=provider.model_name,
        ) for _, r in naive_res if r
    )
    rows.append({
        "system": "naive LLM (no contract)",
        "placeholder_rate": naive_pres["placeholder_rate"],
        "math_rate": naive_pres["math_rate"],
        "coverage": 1.0,
        "fps": None, "ppa": None, "mfr": None,
        "cost_cny": naive_cost,
    })

    # ── google (best-effort, free endpoint) ──
    if try_google:
        loop = asyncio.get_event_loop()
        g_pairs: list[tuple[str, str | None]] = []
        unavailable = 0
        for u in units:
            g = await loop.run_in_executor(None, baselines.google_translate, u.src_text, language)
            if g is None:
                unavailable += 1
            g_pairs.append((u.src_text, g))
        if unavailable < len(units):
            g_pres = baselines.aggregate_preservation(g_pairs)
            rows.append({
                "system": f"google ({len(units)-unavailable}/{len(units)} ok)",
                "placeholder_rate": g_pres["placeholder_rate"],
                "math_rate": g_pres["math_rate"],
                "coverage": 1.0, "fps": None, "ppa": None, "mfr": None, "cost_cny": 0.0,
            })
        else:
            logger.warning("google endpoint unavailable for all units (429); skipping")

    # ── abstract-only (coverage dimension) ──
    total_chars = sum(len(u.src_plain or "") for u in units)
    abstract_chars = len(paper.source_abstract or "")
    rows.append({
        "system": "abstract-only (status quo)",
        "placeholder_rate": None, "math_rate": None,
        "coverage": baselines.coverage_fraction(abstract_chars, total_chars),
        "fps": None, "ppa": None, "mfr": None, "cost_cny": None,
    })

    _print_table(arxiv_id, language, job.id, rows)
    if out_dir:
        _write_csv(Path(out_dir) / f"compare_{arxiv_id}_{language}.csv", rows)


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.4f}"


def _print_table(arxiv_id, language, job_id, rows):
    print("\n" + "=" * 92)
    print(f"Differentiation comparison  paper={arxiv_id} lang={language} ours-job={job_id}")
    print("-" * 92)
    print(f"{'system':28}{'placeholder':>12}{'math':>9}{'coverage':>10}{'FPS':>8}{'PPA':>8}{'cost¥':>11}")
    for r in rows:
        print(
            f"{r['system']:28}"
            f"{_fmt(r['placeholder_rate'], True):>12}"
            f"{_fmt(r['math_rate'], True):>9}"
            f"{_fmt(r['coverage'], True):>10}"
            f"{_fmt(r['fps']):>8}"
            f"{(_fmt(r['ppa']) if r['ppa'] is not None else '—'):>8}"
            f"{(f'{r['cost_cny']:.4f}' if r['cost_cny'] is not None else '—'):>11}"
        )
    print("=" * 92)
    print("structure (placeholder/math preservation): ours keeps inline math/citations; "
          "naive corrupts them.\ncoverage: full-text vs abstract-only.")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arxiv_id")
    ap.add_argument("lang")
    ap.add_argument("--job", type=int, default=None, help="ours job_id (default: latest success)")
    ap.add_argument("--google", action="store_true", help="also try the (flaky) Google baseline")
    ap.add_argument("--out", default=None, help="output dir for the comparison CSV")
    a = ap.parse_args()
    asyncio.run(amain(a.arxiv_id, a.lang, a.job, a.google, a.out))


if __name__ == "__main__":
    main()
