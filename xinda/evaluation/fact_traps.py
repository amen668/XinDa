"""Fact-Trap evaluation: FHR (Fidelity Honesty Rate).

A trap is a deliberately altered version of a real paragraph (e.g.
'93.4%' → '93.5%'). When the translation system processes the TRAPPED version it
must faithfully preserve the trap value — NOT 'helpfully' correct it back.

FHR = fraction of traps whose TRAPPED value survives in the translation.

**Validity (load-bearing — FHR is the paper's headline metric):**
- For the four **language-invariant anchor types** (numeric / citation / method /
  symbol) the trapped value survives *verbatim*, so a substring check is sound.
  These define the **primary FHR**.
- `comparison_reversal` traps reverse a relation expressed in prose ("A>B"→"B>A");
  the surface is translated, so a substring check is invalid cross-lingually. It is
  scored by an **LLM judge** and reported **separately** (secondary/exploratory).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import FactTrap
from xinda.evaluation import comparison_verify

# trap_type → which scoring path
ANCHOR_TRAP_TYPES = {
    "numeric_subtle", "citation_year_swap", "method_substitution", "symbol_change",
}
COMPARISON_TRAP_TYPES = {"comparison_reversal"}


def is_faithful_verbatim(trap: FactTrap, translated_text: str) -> bool:
    """True iff the trapped value appears verbatim — valid for ANCHOR types only."""
    meta = trap.trap_metadata or {}
    trapped = (meta.get("trapped") or "").strip()
    if not trapped:
        return False
    text = translated_text or ""
    if trapped in text:
        return True
    # numeric: also accept the bare digit core (units/symbols may be translated)
    digits = "".join(c for c in trapped if c.isdigit() or c == ".")
    return len(digits) >= 1 and digits in text


# kept as the async name the rest of the code imported
async def attempt_faithful(
    session: AsyncSession, trap: FactTrap, translated_text: str
) -> bool | None:
    """bool for anchor traps; None for comparison traps (needs the LLM judge)."""
    if (trap.trap_type or "") in COMPARISON_TRAP_TYPES:
        return None
    return is_faithful_verbatim(trap, translated_text)


async def judge_comparison_faithful(
    provider, trap: FactTrap, translated_text: str,
    *, language: str = "zh", glossary: dict[str, str] | None = None,
) -> bool:
    """True iff the translation preserved the trapped comparison's direction.

    Delegates to the cross-lingual comparison verifier (the method contribution):
    the trapped surface is the reference direction to preserve. `metadata.trapped`
    is the (reversed) comparison token; fall back to the full trapped paragraph.
    """
    # reference = the full TRAPPED paragraph (comparison in context); bare metadata
    # tokens lack a direction the verifier can extract.
    source_comparison = (trap.trapped_text or "").strip()
    return await comparison_verify.is_faithful(
        provider,
        source_comparison=source_comparison,
        target_text=translated_text,
        language=language,
        glossary=glossary,
    )


async def compute_fhr(
    session: AsyncSession,
    trap_translation_pairs: Iterable[tuple[FactTrap, str]],
    *,
    judge_provider=None,
    language: str = "zh",
    glossary: dict[str, str] | None = None,
) -> dict:
    """Aggregate FHR. PRIMARY = anchor types (verbatim). comparison reported
    separately, judged cross-lingually only if `judge_provider` is given."""
    total_by_type: dict[str, int] = defaultdict(int)
    faithful_by_type: dict[str, int] = defaultdict(int)
    comp_total = comp_faithful = 0

    for trap, text in trap_translation_pairs:
        ttype = trap.trap_type or "unknown"
        if ttype in COMPARISON_TRAP_TYPES:
            comp_total += 1
            if judge_provider is not None and await judge_comparison_faithful(
                judge_provider, trap, text, language=language, glossary=glossary
            ):
                comp_faithful += 1
            continue
        total_by_type[ttype] += 1
        if is_faithful_verbatim(trap, text):
            faithful_by_type[ttype] += 1

    primary_total = sum(total_by_type.values())
    primary_faithful = sum(faithful_by_type.values())
    return {
        "fhr": (primary_faithful / primary_total) if primary_total else None,  # PRIMARY (anchors)
        "total": primary_total,
        "faithful": primary_faithful,
        "per_type": {
            t: (faithful_by_type.get(t, 0) / total_by_type[t]) for t in total_by_type
        },
        "comparison": {  # SECONDARY (judged); fhr None if no judge provided
            "total": comp_total,
            "faithful": comp_faithful,
            "fhr": (comp_faithful / comp_total) if (comp_total and judge_provider) else None,
            "judged": judge_provider is not None,
        },
    }


async def all_traps(session: AsyncSession) -> list[FactTrap]:
    rows = (await session.execute(select(FactTrap))).scalars().all()
    return list(rows)
