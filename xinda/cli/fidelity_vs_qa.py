"""Exp2 — fidelity verification > QA comprehension (the novelty-② demonstration).

A *fluent reversal* (a comparison translated with its direction flipped, e.g. "A > B"
rendered as "B > A") is perfectly readable, so a reading-comprehension (QA) evaluation —
which measures whether a reader can understand the translation — does NOT distinguish it
from a faithful translation. A fidelity check that audits the specific claim DOES.

For each `comparison_reversal` trap we build two translations of the SAME paragraph:
  - good = faithful translation of `original_text`   (direction preserved)
  - bad  = faithful translation of `trapped_text`    (direction flipped — the error)
then score both with BOTH evaluators:
  - QA comprehension (`rcs` prompts): 3 questions from the SOURCE, reader answers from the
    translation, judge grades vs reference. Reported as mean correctness ∈ [0,1].
  - fidelity verifier (`comparison_verify`, focused on the claim): preserved / reversed /
    weakened / dropped.

Headline: QA scores `good` and `bad` almost identically (ΔRCS ≈ 0 → blind to the flip),
while the fidelity verifier flags the `bad` ones it lets the `good` ones pass.

    python -m xinda.cli.fidelity_vs_qa --lang zh --limit 12
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
from pathlib import Path

from sqlalchemy import select

from xinda.config import settings
from xinda.util import loads_dict
from xinda.db.engine import async_session
from xinda.db.models import FactTrap
from xinda.evaluation import comparison_verify
from xinda.evaluation.fact_traps import COMPARISON_TRAP_TYPES
from xinda.evaluation.rcs import (
    _QA_GEN_SCHEMA,
    _QA_GEN_SYSTEM,
    _READER_SCHEMA,
    _SCORER_SCHEMA,
    _SCORER_SYSTEM,
    _reader_system,
)
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import FAITHFUL_TRANSLATE_SYSTEM, language_name

logger = setup_logger(__name__)



def _parse(text: str) -> dict | None:
    return loads_dict(text)


async def _gen(provider, sem, **kw):
    """One throttle-tolerant LLM call (a stray 429 past SDK retries must not abort the run)."""
    async with sem:
        try:
            return await provider.generate(**kw)
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM call failed: %s", e)
            return None


async def _translate(translator, sem, text: str, lang: str) -> str:
    tr = await _gen(translator, sem, prompt=text or "",
                    system=FAITHFUL_TRANSLATE_SYSTEM.format(lang=language_name(lang)))
    return (tr.text or "").strip() if tr else ""


async def _qa_comprehension(qa_gen, reader, scorer, sem, *, source: str,
                            translation: str, lang: str) -> float | None:
    """3 questions from SOURCE, reader answers from TRANSLATION, judge grades. Mean ∈ [0,1]."""
    g = await _gen(qa_gen, sem, prompt=source, system=_QA_GEN_SYSTEM,
                   json_schema=_QA_GEN_SCHEMA)
    obj = _parse(g.text) if g else None
    qas = (obj or {}).get("qas") or []
    if not qas:
        return None
    reader_sys = _reader_system(language_name(lang))
    scores: list[float] = []
    for qa in qas:
        q, ref = qa.get("question", ""), qa.get("reference_answer", "")
        a = await _gen(reader, sem, prompt=f"TRANSLATION:\n{translation}\n\nQUESTION: {q}",
                       system=reader_sys, json_schema=_READER_SCHEMA)
        ans = (_parse(a.text) or {}).get("answer", "") if a else ""
        s = await _gen(scorer, sem, prompt=f"REFERENCE: {ref}\nREADER: {ans}",
                       system=_SCORER_SYSTEM, json_schema=_SCORER_SCHEMA)
        c = (_parse(s.text) or {}).get("correctness") if s else None
        if c is not None:
            scores.append(float(c))
    return statistics.mean(scores) if scores else None


async def amain(lang: str, limit: int, out_dir: str) -> None:
    async with async_session() as session:
        traps = (await session.execute(
            select(FactTrap).where(FactTrap.trap_type.in_(COMPARISON_TRAP_TYPES)).limit(limit)
        )).scalars().all()
    if not traps:
        raise SystemExit("no comparison_reversal traps; run build_fact_traps first")

    translator = create_provider(settings.model_first_pass)
    qa_gen = create_provider(settings.model_qa_reader)
    reader = create_provider(settings.model_qa_reader)
    scorer = create_provider(settings.model_qa_judge)
    verifier = create_provider(settings.model_fact_verify)
    # this demo fans out ~18 calls/trap (translate ×2, QA gen/read/score ×14, verify ×2);
    # the coding endpoint throttles, so keep total in-flight low.
    sem = asyncio.Semaphore(3)
    caught = {comparison_verify.REVERSED, comparison_verify.WEAKENED,
              comparison_verify.DROPPED}

    async def one(trap: FactTrap) -> dict:
        focus = ((trap.trap_metadata or {}).get("claim_surface") or "").strip()
        good_tr, bad_tr = await asyncio.gather(
            _translate(translator, sem, trap.original_text or "", lang),
            _translate(translator, sem, trap.trapped_text or "", lang),
        )
        # QA comprehension on both translations (source = original = ground truth)
        rcs_good, rcs_bad = await asyncio.gather(
            _qa_comprehension(qa_gen, reader, scorer, sem,
                              source=trap.original_text or "", translation=good_tr, lang=lang),
            _qa_comprehension(qa_gen, reader, scorer, sem,
                              source=trap.original_text or "", translation=bad_tr, lang=lang),
        )
        # fidelity verifier: audit the claim against the ORIGINAL (correct) source
        async with sem:
            fid_good, _ = await comparison_verify.verify(
                verifier, source_comparison=trap.original_text or "", target_text=good_tr,
                language=lang, focus=focus)
        async with sem:
            fid_bad, _ = await comparison_verify.verify(
                verifier, source_comparison=trap.original_text or "", target_text=bad_tr,
                language=lang, focus=focus)
        return {"trap": trap.id, "rcs_good": rcs_good, "rcs_bad": rcs_bad,
                "fid_good": fid_good, "fid_bad": fid_bad,
                # the texts make the CSV double as a qualitative-example bank for the paper
                "focus": focus, "source": trap.original_text or "",
                "good_tr": good_tr, "bad_tr": bad_tr}

    rows = await asyncio.gather(*(one(t) for t in traps))

    rg = [r["rcs_good"] for r in rows if r["rcs_good"] is not None]
    rb = [r["rcs_bad"] for r in rows if r["rcs_bad"] is not None]
    fid_bad_caught = sum(1 for r in rows if r["fid_bad"] in caught)
    fid_good_pass = sum(1 for r in rows if r["fid_good"] == comparison_verify.PRESERVED)
    n = len(rows)

    print("\n" + "=" * 64)
    print(f"Exp2  fidelity > QA   (lang={lang}, {n} comparison reversals)")
    print("-" * 64)
    print("QA reading-comprehension (mean correctness 0–1):")
    print(f"  faithful (good)  : {statistics.mean(rg):.3f}" if rg else "  good: n/a")
    print(f"  flipped  (bad)   : {statistics.mean(rb):.3f}" if rb else "  bad : n/a")
    if rg and rb:
        print(f"  Δ (good − bad)   : {statistics.mean(rg) - statistics.mean(rb):+.3f}"
              "   ← small Δ + flipped still 'comprehensible' → QA is a weak fidelity signal")
    print("\nfidelity verifier (audits the specific claim):")
    print(f"  flipped flagged  : {fid_bad_caught}/{n}  ({100*fid_bad_caught/n:.0f}%)  ← catches the error")
    print(f"  faithful passed  : {fid_good_pass}/{n}  ({100*fid_good_pass/n:.0f}%)  ← no false alarm")
    print("=" * 64)

    # persist: per-trap rows (also a qualitative-example bank) + the headline summary, so
    # the experiment is reproducible and the paper can quote real flipped sentences.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"fidelity_vs_qa_{lang}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "lang": lang, "n": n,
        "rcs_good": round(statistics.mean(rg), 3) if rg else None,
        "rcs_bad": round(statistics.mean(rb), 3) if rb else None,
        "rcs_delta": round(statistics.mean(rg) - statistics.mean(rb), 3) if rg and rb else None,
        "fid_flipped_caught": f"{fid_bad_caught}/{n}",
        "fid_faithful_passed": f"{fid_good_pass}/{n}",
    }
    (out / f"summary_{lang}.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {csv_path}  ({n} rows)  +  summary_{lang}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp2: fidelity verification vs QA comprehension")
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out-dir", default="results/exp2_fidelity")
    a = ap.parse_args()
    asyncio.run(amain(a.lang, a.limit, a.out_dir))


if __name__ == "__main__":
    main()
