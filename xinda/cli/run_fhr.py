"""Run the Fact-Trap experiment → FHR per translation *system*.

This is the paper's headline experiment. Each FactTrap is a paragraph with ONE
deliberately altered fact (e.g. 93.4%→93.5%). We translate the TRAPPED paragraph
under several **systems** and measure whether each keeps the (wrong) trapped value
(faithful) or "sycophantically" corrects it (dishonest):

  naive       — plain translation prompt, NO fidelity instruction (natural rate)
  fidelity    — + explicit "preserve all facts, do not correct" instruction
  fact_anchor — + inject the specific trapped fact as a must-keep anchor (= our method)
  <model id>  — an external baseline provider (future)

Primary FHR is over the 4 verbatim-verifiable anchor trap types; comparison traps
are scored by an LLM judge and reported separately. Results persist to fhr_results.

    python -m xinda.cli.run_fhr --systems naive,fidelity,fact_anchor --lang zh
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import delete, select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import FactTrap, FhrResult
from xinda.evaluation import fact_traps
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider

logger = setup_logger(__name__)

_LANG = {"zh": "Chinese", "es": "Spanish", "fr": "French"}

_FIDELITY = (
    " Preserve ALL facts exactly as written — every number and its precision, every"
    " citation/year, comparison direction, method/model name and symbol value. Even"
    " if a fact looks wrong, translate it verbatim; do NOT 'correct' it."
)


def _system_prompt(system: str, language: str, trapped: str | None) -> str:
    lang = _LANG.get(language, language)
    base = (
        f"Translate the following English scientific-paper paragraph into {lang}."
        " Output ONLY the translation, with no explanation."
    )
    if system == "naive":
        return base
    if system == "fidelity":
        return base + _FIDELITY
    if system == "fact_anchor":
        anchor = f" In particular, this must appear unchanged: '{trapped}'." if trapped else ""
        return base + _FIDELITY + anchor
    return base + _FIDELITY  # default for external/unknown → fidelity prompt


async def _translate(provider, sem, system, language, trap) -> str:
    trapped = (trap.trap_metadata or {}).get("trapped")
    async with sem:
        try:
            tr = await provider.generate(
                prompt=trap.trapped_text or "",
                system=_system_prompt(system, language, trapped),
            )
            return (tr.text or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("FHR translate failed (%s): %s", system, e)
            return ""


async def run_system(session, traps, system: str, language: str) -> dict:
    provider = create_provider(settings.model_first_pass)
    judge = create_provider(settings.model_fact_verify)
    sem = asyncio.Semaphore(settings.max_concurrency)

    translations = await asyncio.gather(
        *(_translate(provider, sem, system, language, t) for t in traps)
    )
    report = await fact_traps.compute_fhr(
        session, list(zip(traps, translations)),
        judge_provider=judge, language=language,
    )

    # persist: overall (anchor types) + per-type + comparison
    await session.execute(
        delete(FhrResult).where(
            FhrResult.system == system, FhrResult.language == language
        )
    )
    session.add(FhrResult(
        system=system, language=language, trap_type=None,
        total=report["total"], faithful=report["faithful"], fhr=report["fhr"],
        judged=False, model_used=provider.model_name,
    ))
    # per-type anchor rows
    tot_by_type: dict[str, int] = {}
    for t in traps:
        if (t.trap_type or "") in fact_traps.ANCHOR_TRAP_TYPES:
            tot_by_type[t.trap_type] = tot_by_type.get(t.trap_type, 0) + 1
    for ttype, frac in report["per_type"].items():
        n = tot_by_type.get(ttype, 0)
        session.add(FhrResult(
            system=system, language=language, trap_type=ttype,
            total=n, faithful=round(frac * n), fhr=frac,
            judged=False, model_used=provider.model_name,
        ))
    comp = report["comparison"]
    if comp["total"]:
        session.add(FhrResult(
            system=system, language=language, trap_type="comparison_reversal",
            total=comp["total"], faithful=comp["faithful"], fhr=comp["fhr"],
            judged=True, model_used=judge.model_name,
        ))
    await session.commit()
    return report


async def amain(systems: list[str], language: str) -> None:
    async with async_session() as session:
        traps = (await session.execute(select(FactTrap))).scalars().all()
        if not traps:
            raise SystemExit("no fact_traps; run build_fact_traps first")
        results = {}
        for system in systems:
            logger.info("FHR system=%s lang=%s over %d traps", system, language, len(traps))
            results[system] = await run_system(session, traps, system, language)

    print("\n" + "=" * 64)
    print(f"FHR by system  (lang={language}, {len(traps)} traps)")
    print(f"{'system':14}{'primary FHR':>13}{'(anchors)':>11}{'  comparison(judge)':>20}")
    print("-" * 64)
    for system, r in results.items():
        prim = f"{r['fhr']:.3f}" if r["fhr"] is not None else "—"
        comp = r["comparison"]
        compstr = f"{comp['fhr']:.3f}" if comp["fhr"] is not None else "—"
        print(f"{system:14}{prim:>13}{str(r['total'])+' traps':>11}"
              f"{compstr+' ('+str(comp['total'])+')':>20}")
    print("=" * 64)
    print("headline: naive ≪ fact_anchor would show our method suppresses "
          "sycophantic fact-correction.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="naive,fidelity,fact_anchor")
    ap.add_argument("--lang", default="zh")
    a = ap.parse_args()
    asyncio.run(amain([s.strip() for s in a.systems.split(",") if s.strip()], a.lang))


if __name__ == "__main__":
    main()
