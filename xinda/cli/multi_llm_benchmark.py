"""Multi-LLM × QE benchmark — validates that the automated QE has discriminative
power (a weak translator should score low, a strong one high).

Each model in the registry translates the SAME paper's units with the SAME shared
prompt (so quality differences reflect the model, not the prompt). A FIXED judge
model (`settings.model_judge_rubric`) then scores every system — using one judge
for all keeps the comparison unbiased. Per-model translations are persisted so the
GPU metrics (xCOMET via `cli/neural_qe`) and FPS (via FactVerify) can be backfilled.

NOT run by default — set each vendor's API key env var first (see providers/registry).
Models whose key is absent are skipped.

  python -m xinda.cli.multi_llm_benchmark 2503.15129 zh \
      --models qwen-turbo,qwen-plus,qwen-max,deepseek-chat,glm-4.6,kimi-k2 \
      --judge qwen-max --sample 20 --out workspace/_multillm

Output: a per-model table (RUBRIC / placeholder-preservation / tokens / cost) — if
RUBRIC ranks turbo < plus < max (and the strong external models high), the QE
discriminates quality. Then run neural_qe on the persisted sets for xCOMET.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

from sqlalchemy import select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import Paper, TranslationUnit
from xinda.evaluation import baselines, cost, judge_rubric
from xinda.logger_config import setup_logger
from xinda.providers import registry
from xinda.providers.factory import create_provider
from xinda.translation.prompts import language_name

logger = setup_logger(__name__)

_PH = re.compile(r"\{\{[^{}]+?\}\}")


def _shared_system(lang: str) -> str:
    return (
        f"Translate the following English scientific-paper text into {language_name(lang)}. "
        "Keep every placeholder of the form {{PT_XXX_N}}, {{NL}}, {{TAB}}, {{RE}} verbatim. "
        "Use standard academic terminology. Output ONLY the translation."
    )


async def _translate_all(model_key: str, units, lang: str):
    """Translate every unit with `model_key`. Returns dict unit_id -> (tgt, result)."""
    provider = create_provider(model_key)
    system = _shared_system(lang)
    sem = asyncio.Semaphore(settings.max_concurrency)

    async def one(u):
        async with sem:
            try:
                r = await provider.generate(prompt=u.src_text, system=system)
                return u.id, (r.text or "", r)
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] unit %d failed: %s", model_key, u.id, e)
                return u.id, ("", None)

    return dict(await asyncio.gather(*(one(u) for u in units)))


async def _judge_sample(units, out: dict, sample: int) -> float | None:
    """Mean RUBRIC over a sample (fixed judge model, runs=1 to bound cost)."""
    chosen = units[:sample]
    scores: list[float] = []

    async def j(u):
        tgt, _ = out.get(u.id, ("", None))
        if not tgt:
            return None
        tgt_read = _PH.sub("", tgt)  # strip placeholders for readability judging
        res = await judge_rubric.judge_one(u.src_plain, tgt_read, field=None, runs=1)
        return res.get("rubric_score") if res.get("valid") else None

    for s in await asyncio.gather(*(j(u) for u in chosen)):
        if s is not None:
            scores.append(s)
    return statistics.mean(scores) if scores else None


async def amain(arxiv_id: str, lang: str, models: list[str], judge: str,
                sample: int, out_dir: str) -> None:
    # fix the judge model for the whole benchmark (unbiased)
    settings.model_judge_rubric = judge

    async with async_session() as session:
        paper = (
            await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        ).scalar_one_or_none()
        if paper is None:
            raise SystemExit(f"paper {arxiv_id} not found; run extract first")
        units = (
            await session.execute(
                select(TranslationUnit).where(TranslationUnit.paper_id == paper.id)
                .order_by(TranslationUnit.ord)
            )
        ).scalars().all()

    rows = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for mk in models:
        spec = registry.get_spec(mk)
        if spec is not None and not spec.available:
            print(f"skip {mk}: {spec.api_key_env} not set")
            continue
        logger.info("translating %d units with %s", len(units), mk)
        out = await _translate_all(mk, units, lang)
        pres = baselines.aggregate_preservation([(u.src_text, out.get(u.id, ("", None))[0]) for u in units])
        toks = [r for _, r in out.values() if r]
        fresh = sum(r.fresh_prompt_tokens for r in toks)
        cached = sum(r.cached_prompt_tokens for r in toks)
        comp = sum(r.completion_tokens for r in toks)
        c = cost.cost_cny(fresh_prompt_tok=fresh, cached_prompt_tok=cached,
                          completion_tok=comp, model_name=mk)
        rubric = await _judge_sample(units, out, sample)
        rows.append({
            "model": mk, "n_ok": len(toks),
            "rubric": rubric, "placeholder_rate": pres["placeholder_rate"],
            "tokens": fresh + cached + comp, "cost_cny_qwenprice": round(c, 4),
        })
        # persist translations for later xCOMET / FPS backfill
        (out_path / f"{arxiv_id}_{lang}_{mk}.json").write_text(
            json.dumps({str(u.id): out.get(u.id, ("", None))[0] for u in units},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"Multi-LLM × QE benchmark  paper={arxiv_id} lang={lang}  judge={judge}")
    print("-" * 76)
    print(f"{'model':22}{'n':>5}{'RUBRIC(1-5)':>13}{'placeholder':>13}{'tokens':>10}{'¥(qwen-eq)':>12}")
    for r in sorted(rows, key=lambda x: (x["rubric"] or 0), reverse=True):
        rub = f"{r['rubric']:.2f}" if r["rubric"] is not None else "—"
        print(f"{r['model']:22}{r['n_ok']:>5}{rub:>13}"
              f"{r['placeholder_rate']*100:>12.1f}%{r['tokens']:>10}{r['cost_cny_qwenprice']:>12}")
    print("=" * 76)
    print("If RUBRIC ranks weak<strong models sensibly, the automated QE discriminates "
          "quality.\nxCOMET: run neural_qe on the persisted per-model sets in " + out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arxiv_id")
    ap.add_argument("lang")
    ap.add_argument("--models", default=",".join(registry.DEFAULT_BENCHMARK),
                    help="comma-separated registry keys")
    ap.add_argument("--judge", default="qwen-max", help="fixed judge model id")
    ap.add_argument("--sample", type=int, default=20, help="units to RUBRIC-judge")
    ap.add_argument("--out", default="workspace/_multillm")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    asyncio.run(amain(a.arxiv_id, a.lang, models, a.judge, a.sample, a.out))


if __name__ == "__main__":
    main()
