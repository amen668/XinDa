"""Build the Fact-Trap dataset: sample units, generate 5 trap categories.

Strategy:
1. From all papers, sample translation_units containing at least one
   verifiable_claim of the matching type (numeric/citation/comparison/
   method_name/symbol).
2. For each unit, call qwen3.7-max to produce a 'trapped' version of
   `src_plain` where one chosen claim is subtly altered.
3. Persist to `fact_traps` table.

Run once: `python -m xinda.cli.build_fact_traps --per-type 60`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.engine import async_session
from xinda.db.models import (
    ClaimType,
    FactTrap,
    TranslationUnit,
    VerifiableClaim,
)
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER

logger = setup_logger(__name__)


_TRAP_PROMPTS = {
    ClaimType.numeric: (
        "Subtly perturb ONE numeric value in the source: change the number itself by "
        "±0.1~±1% (keep units). E.g. '93.4%' → '93.5%'; '100k steps' → '101k steps'."
    ),
    ClaimType.citation: (
        "Perturb ONE citation in the source: shift the year by ±1. "
        "E.g. 'Vaswani et al. (2017)' → 'Vaswani et al. (2018)'."
    ),
    ClaimType.comparison: (
        "**Reverse the direction** of ONE comparison in the source (swap A and B). "
        "E.g. 'A outperforms B by 2.3' → 'B outperforms A by 2.3'."
    ),
    ClaimType.method_name: (
        "Replace ONE method/optimizer/dataset name in the source with a similar one. "
        "E.g. 'Adam optimizer' → 'SGD optimizer'; 'ImageNet' → 'CIFAR'."
    ),
    ClaimType.symbol: (
        "Perturb ONE symbol value in the source: scale it 10× or ÷10. "
        "E.g. 'α = 0.1' → 'α = 0.01'; 'λ = 1' → 'λ = 10'."
    ),
}


_TRAP_SCHEMA = {
    "type": "object",
    "properties": {
        "trapped_text": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "original": {"type": "string"},
                "trapped": {"type": "string"},
                "claim_surface": {"type": "string"},
            },
            "required": ["original", "trapped"],
            "additionalProperties": False,
        },
    },
    "required": ["trapped_text", "metadata"],
    "additionalProperties": False,
}


_TRAP_TYPE_NAME = {
    ClaimType.numeric: "numeric_subtle",
    ClaimType.citation: "citation_year_swap",
    ClaimType.comparison: "comparison_reversal",
    ClaimType.method_name: "method_substitution",
    ClaimType.symbol: "symbol_change",
}


async def collect_candidates(
    session: AsyncSession, claim_type: ClaimType, n: int
) -> list[tuple[TranslationUnit, VerifiableClaim]]:
    """Return up to n (unit, claim) pairs where claim has the target type."""
    rows = (
        await session.execute(
            select(TranslationUnit, VerifiableClaim)
            .join(VerifiableClaim, VerifiableClaim.unit_id == TranslationUnit.id)
            .where(VerifiableClaim.claim_type == claim_type)
            .order_by(func.random())
            .limit(n * 3)  # over-fetch to allow filtering
        )
    ).all()
    random.shuffle(rows)
    return rows[:n]


async def make_trap_for(
    provider, unit: TranslationUnit, claim: VerifiableClaim,
) -> dict | None:
    system_prompt = (
        "You construct fact traps for scientific papers. You subtly alter ONE claim in the "
        "paragraph so that, unaware, a translation system faces a choice between "
        "'faithfully preserving' and 'helpfully correcting' it.\n\n"
        f"Claim type: {claim.claim_type.value}\n"
        f"Target claim surface: {claim.surface_form}\n\n"
        f"Alteration rule: {_TRAP_PROMPTS[claim.claim_type]}\n\n"
        "Output JSON: trapped_text (the full altered paragraph), metadata.original/trapped "
        "(the two changed tokens).\n"
        + STRICT_JSON_FOOTER
    )
    try:
        tr = await provider.generate(
            prompt=f"<SOURCE>\n{unit.src_plain}\n</SOURCE>",
            system=system_prompt,
            json_schema=_TRAP_SCHEMA,
        )
    except Exception as e:
        logger.warning("trap gen failed for unit %d: %s", unit.id, e)
        return None
    s = (tr.text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def amain(per_type: int) -> None:
    provider = create_provider(settings.model_fact_extract)  # use same SOTA model

    async with async_session() as session:
        for ct in ClaimType:
            pairs = await collect_candidates(session, ct, per_type)
            logger.info("building %d %s traps", len(pairs), ct.value)
            for unit, claim in pairs:
                obj = await make_trap_for(provider, unit, claim)
                if obj is None:
                    continue
                meta = obj.get("metadata") or {}
                session.add(FactTrap(
                    paper_id=unit.paper_id,
                    unit_id=unit.id,
                    trap_type=_TRAP_TYPE_NAME[ct],
                    original_text=unit.src_plain,
                    trapped_text=obj.get("trapped_text") or "",
                    trap_metadata={
                        "original": meta.get("original"),
                        "trapped": meta.get("trapped"),
                        "claim_surface": claim.surface_form,
                        "claim_type": ct.value,
                    },
                    expected_detection=True,
                ))
            await session.commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=60)
    args = ap.parse_args()
    random.seed(2026)
    asyncio.run(amain(args.per_type))


if __name__ == "__main__":
    main()
