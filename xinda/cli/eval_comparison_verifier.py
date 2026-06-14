"""Validate the cross-lingual comparison verifier (method contribution).

The verifier's accuracy is measured against **synthetic ground truth** — no human
labelling. Each `comparison_reversal` fact-trap gives a clean paragraph
(`original_text`, original direction) and a trapped one (`trapped_text`, reversed
direction). We faithfully translate both into the target language and ask the
verifier whether each translation preserves the *trapped* comparison:

  - translation of `trapped_text`   → should be PRESERVED  (keeps trapped dir)
  - translation of `original_text`  → should be REVERSED   (keeps original dir,
                                       i.e. the opposite of the trapped reference)

So every trap yields one PRESERVED and one REVERSED labelled case, by
construction. We then report the verifier's precision/recall (PRESERVED = positive
class) plus the 4-way confusion matrix.

    python -m xinda.cli.eval_comparison_verifier --lang zh [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict

from sqlalchemy import select

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import FactTrap
from xinda.evaluation import comparison_verify
from xinda.evaluation.fact_traps import COMPARISON_TRAP_TYPES
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import FAITHFUL_TRANSLATE_SYSTEM, language_name

logger = setup_logger(__name__)

# A faithful-translation prompt so the constructed label is reliable: the
# translation must reflect the input paragraph's stated direction verbatim.

PRESERVED = comparison_verify.PRESERVED
REVERSED = comparison_verify.REVERSED


async def _translate(provider, sem, text: str, lang: str) -> str:
    async with sem:
        try:
            tr = await provider.generate(
                prompt=text or "",
                system=FAITHFUL_TRANSLATE_SYSTEM.format(lang=language_name(lang)),
            )
            return (tr.text or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("translate failed: %s", e)
            return ""


async def _label_case(judge, sem, *, reference: str, target_text: str, lang: str,
                      focus: str | None = None) -> str:
    if not target_text:
        return comparison_verify.WEAKENED
    async with sem:
        verdict, _ = await comparison_verify.verify(
            judge, source_comparison=reference, target_text=target_text, language=lang,
            focus=focus,
        )
    return verdict


async def amain(language: str, limit: int | None) -> None:
    async with async_session() as session:
        q = select(FactTrap).where(FactTrap.trap_type.in_(COMPARISON_TRAP_TYPES))
        if limit:
            q = q.limit(limit)
        traps = (await session.execute(q)).scalars().all()
    if not traps:
        raise SystemExit("no comparison_reversal traps; run build_fact_traps first")

    translator = create_provider(settings.model_first_pass)
    judge = create_provider(settings.model_fact_verify)
    sem = asyncio.Semaphore(settings.max_concurrency)

    # 1) faithfully translate both paragraphs of every trap
    async def both(trap: FactTrap) -> tuple[str, str, str, str]:
        # reference = the full TRAPPED paragraph (the comparison in context); the
        # bare metadata tokens (e.g. "BERT, ours") lack direction, so the verifier
        # can't extract a source direction from them.
        ref = (trap.trapped_text or "").strip()
        # focus = the specific claim under audit, so the verifier extracts THAT comparison
        # on both sides (passages often hold several comparisons → otherwise it picks a
        # different, more salient one on each side and a preserved direction reads as match).
        focus = ((trap.trap_metadata or {}).get("claim_surface") or "").strip()
        tr_trapped, tr_original = await asyncio.gather(
            _translate(translator, sem, trap.trapped_text or "", language),
            _translate(translator, sem, trap.original_text or "", language),
        )
        return ref, tr_trapped, tr_original, focus

    quads = await asyncio.gather(*(both(t) for t in traps))

    # 2) build labelled cases and run the verifier
    cases: list[tuple[str, str]] = []  # (gold_label, predicted_verdict)

    async def judge_case(reference: str, target_text: str, gold: str,
                         focus: str) -> tuple[str, str]:
        pred = await _label_case(judge, sem, reference=reference, target_text=target_text,
                                 lang=language, focus=focus)
        return gold, pred

    jobs = []
    for ref, tr_trapped, tr_original, focus in quads:
        jobs.append(judge_case(ref, tr_trapped, PRESERVED, focus))
        jobs.append(judge_case(ref, tr_original, REVERSED, focus))
    cases = list(await asyncio.gather(*jobs))

    # 3) metrics — PRESERVED is the positive class
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    tp = fp = fn = 0
    correct = 0
    for gold, pred in cases:
        confusion[(gold, pred)] += 1
        if pred == gold:
            correct += 1
        if gold == PRESERVED and pred == PRESERVED:
            tp += 1
        elif gold != PRESERVED and pred == PRESERVED:
            fp += 1
        elif gold == PRESERVED and pred != PRESERVED:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(cases)

    print("\n" + "=" * 60)
    print(f"Comparison verifier validation  (lang={language}, {len(traps)} traps, {n} cases)")
    print("-" * 60)
    print(f"accuracy (4-way)        : {correct}/{n} = {correct / n:.3f}")
    print(f"PRESERVED precision     : {precision:.3f}")
    print(f"PRESERVED recall        : {recall:.3f}")
    print(f"PRESERVED F1            : {f1:.3f}")
    print("\nconfusion (gold → predicted):")
    golds = [PRESERVED, REVERSED]
    preds = [PRESERVED, REVERSED, comparison_verify.DROPPED, comparison_verify.WEAKENED]
    header = "gold\\pred".ljust(12) + "".join(p.ljust(11) for p in preds)
    print(header)
    for g in golds:
        row = g.ljust(12) + "".join(str(confusion.get((g, p), 0)).ljust(11) for p in preds)
        print(row)
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(amain(a.lang, a.limit))


if __name__ == "__main__":
    main()
