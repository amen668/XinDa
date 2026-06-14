"""FactVerify stage (C2): re-extract claims from target, align with source.

For each Translation row with status=translated|refined, this stage:
1. Re-runs the same LLM extraction prompt against `tgt_plain`
2. Matches target claims to source verifiable_claims rows
3. Per-source-claim runs the type-specific drift check
4. Persists claim_verifications rows + sets translations.fps_unit

Refine stage later reads claim_verifications WHERE drift != verified to
build the failed-claims list for the refinement prompt.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    ClaimType,
    ClaimVerification,
    DriftType,
    PipelineStage,
    Translation,
    TranslationUnit,
    TuStatus,
    VerifiableClaim,
)
from xinda.evaluation import comparison_verify
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.pipeline.stages.fact_extract import (
    FACT_EXTRACT_SCHEMA,
    FACT_EXTRACT_SYSTEM_PROMPT,
)
from xinda.providers.factory import create_provider
from xinda.translation import fact_anchors
from xinda.translation.rate_limit import RateLimiter

logger = setup_logger(__name__)


class FactVerify:
    name = PipelineStage.fact_verify
    recoverable = True

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.job_id is None or ctx.paper_id is None:
            return False
        if not ctx.config.use_fact_verify:
            return True  # ablation
        # Done when every translation has fps_unit populated.
        rows = (
            await session.execute(
                select(Translation.id, Translation.fps_unit)
                .where(Translation.job_id == ctx.job_id, Translation.pass_no == 1)
            )
        ).all()
        if not rows:
            return False
        return all(r[1] is not None for r in rows)

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("FactVerify requires job_id/paper_id")
        if not ctx.config.use_fact_verify:
            logger.info("FactVerify skipped by config")
            return ctx

        # Load latest Translation per unit + that unit's claims
        joined = (
            await session.execute(
                select(Translation, TranslationUnit)
                .join(TranslationUnit, Translation.unit_id == TranslationUnit.id)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.status != TuStatus.pending,
                )
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).all()
        if not joined:
            raise StageError("no translations to verify")

        latest: dict[int, tuple[Translation, TranslationUnit]] = {}
        for t, u in joined:
            latest.setdefault(t.unit_id, (t, u))

        # Bulk-load claims for all involved units
        unit_ids = [u.id for _, u in latest.values()]
        crows = (
            await session.execute(
                select(VerifiableClaim).where(VerifiableClaim.unit_id.in_(unit_ids))
            )
        ).scalars().all()
        claims_by_unit: dict[int, list[VerifiableClaim]] = defaultdict(list)
        for c in crows:
            claims_by_unit[c.unit_id].append(c)

        # Clear any prior verifications for these translations (defensive)
        tr_ids = [t.id for t, _ in latest.values()]
        if tr_ids:
            await session.execute(
                delete(ClaimVerification).where(
                    ClaimVerification.translation_id.in_(tr_ids)
                )
            )
            await session.commit()

        # Glossary target renderings → credit method_name claims translated
        # consistently per the locked glossary as preserved.
        from xinda.db.models import GlossaryTerm
        gterms = (
            await session.execute(
                select(GlossaryTerm).where(
                    GlossaryTerm.paper_id == ctx.paper_id,
                    GlossaryTerm.language == ctx.config.language,
                )
            )
        ).scalars().all()
        glossary_map = {g.src_term.lower(): g.tgt_term for g in gterms if g.tgt_term}

        provider = create_provider(settings.model_fact_verify)
        limiter = RateLimiter(provider.rpm, provider.tpm)

        sem = asyncio.Semaphore(settings.max_concurrency)

        async def worker(t_unit: tuple[Translation, TranslationUnit]):
            t, u = t_unit
            async with sem:
                return await _verify_one(
                    provider, limiter, t, u, claims_by_unit.get(u.id, []),
                    glossary_map, ctx.config.language,
                )

        per_unit = await asyncio.gather(*(worker(p) for p in latest.values()))

        # Persist
        for t_unit, verifications in zip(latest.values(), per_unit):
            t, _u = t_unit
            total = len(verifications)
            verified = sum(1 for v in verifications if v.verified)
            t.fps_unit = (verified / total) if total else 1.0
            for v in verifications:
                session.add(v)
        await session.commit()

        logger.info("fact_verify: processed %d translations", len(latest))
        return ctx


# ────────────────────────── helpers ──────────────────────────


async def _verify_one(
    provider, limiter, translation: Translation, unit: TranslationUnit,
    src_claims: list[VerifiableClaim],
    glossary_map: dict[str, str] | None = None,
    language: str = "zh",
) -> list[ClaimVerification]:
    """Return a list of ClaimVerification rows for one translation."""
    if not src_claims:
        return []
    tgt_plain = translation.tgt_plain or ""
    if not tgt_plain:
        # everything missing
        return [
            ClaimVerification(
                translation_id=translation.id, claim_id=c.id,
                verified=False, drift=DriftType.missing,
                tgt_surface="", tgt_normalized="",
                drift_magnitude=1.0, verifier=provider.model_name,
            ) for c in src_claims
        ]

    # Step 1: re-extract claims from translation
    prompt = f"<INPUT>{tgt_plain}</INPUT>"
    est = provider.estimate_tokens(FACT_EXTRACT_SYSTEM_PROMPT) + provider.estimate_tokens(prompt)
    await limiter.reserve(est)

    try:
        tr = await provider.generate(
            prompt=prompt,
            system=FACT_EXTRACT_SYSTEM_PROMPT,
            json_schema=FACT_EXTRACT_SCHEMA,
        )
    except Exception as e:
        logger.warning("verify extract failed for translation %d: %s", translation.id, e)
        return [
            ClaimVerification(
                translation_id=translation.id, claim_id=c.id,
                verified=False, drift=DriftType.missing,
                drift_magnitude=1.0, verifier=provider.model_name,
            ) for c in src_claims
        ]

    tgt_claim_records = _parse_target_claims(tr.text)

    # Step 2: convert DB rows to in-memory ClaimRecord for matching
    src_records: list[fact_anchors.ClaimRecord] = [
        fact_anchors.ClaimRecord(
            claim_type=c.claim_type,
            surface_form=c.surface_form,
            normalized=c.normalized,
            metadata=c.claim_metadata or {},
        ) for c in src_claims
    ]

    matches = fact_anchors.match_source_to_target(src_records, tgt_claim_records)

    out: list[ClaimVerification] = []
    for i, src in enumerate(src_records):
        # Comparison claims are verified cross-lingually (the English-normalised
        # re-extract matcher is invalid for translated comparisons). The verifier
        # extracts the target comparison tuple and decides direction preservation.
        if src.claim_type == ClaimType.comparison:
            drift, magnitude, verdict = await comparison_verify.verify_drift(
                provider,
                source_comparison=src.surface_form,
                target_text=tgt_plain,
                language=language,
                glossary=glossary_map,
            )
            out.append(ClaimVerification(
                translation_id=translation.id,
                claim_id=src_claims[i].id,
                verified=(drift == DriftType.verified),
                drift=drift,
                tgt_surface="",
                tgt_normalized=verdict,
                drift_magnitude=magnitude,
                verifier=provider.model_name,
            ))
            continue
        tgt = matches.get(i)
        if tgt is None:
            # No re-extracted match — but the anchor may still be present verbatim
            # in the translation (the re-extraction is brittle cross-lingually).
            if fact_anchors.anchor_preserved(src, tgt_plain, glossary_map):
                out.append(ClaimVerification(
                    translation_id=translation.id,
                    claim_id=src_claims[i].id,
                    verified=True, drift=DriftType.verified,
                    tgt_surface="", tgt_normalized=src.normalized,
                    drift_magnitude=0.0, verifier=provider.model_name,
                ))
                continue
            out.append(ClaimVerification(
                translation_id=translation.id,
                claim_id=src_claims[i].id,
                verified=False, drift=DriftType.missing,
                drift_magnitude=1.0, verifier=provider.model_name,
            ))
            continue
        drift, magnitude = fact_anchors.check_drift(src, tgt)
        # Lenient anchor override: a matched-but-flagged claim whose value still
        # survives verbatim in the target is preserved, not drifted.
        if drift != DriftType.verified and fact_anchors.anchor_preserved(src, tgt_plain, glossary_map):
            drift, magnitude = DriftType.verified, 0.0
        out.append(ClaimVerification(
            translation_id=translation.id,
            claim_id=src_claims[i].id,
            verified=(drift == DriftType.verified),
            drift=drift,
            tgt_surface=tgt.surface_form,
            tgt_normalized=tgt.normalized,
            drift_magnitude=magnitude,
            verifier=provider.model_name,
        ))
    return out


def _parse_target_claims(text: str) -> list[fact_anchors.ClaimRecord]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    items = obj.get("claims") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    out: list[fact_anchors.ClaimRecord] = []
    from xinda.db.models import ClaimType
    for it in items:
        try:
            ctype = ClaimType(it["type"])
        except (KeyError, ValueError):
            continue
        surface = it.get("surface") or ""
        normalized = it.get("normalized") or surface
        metadata = it.get("metadata")
        if not isinstance(metadata, dict):  # model sometimes returns a str/None
            metadata = {}
        if not surface:
            continue
        # re-normalize
        norm_calc, meta_calc = fact_anchors.normalize(ctype, surface)
        if norm_calc:
            normalized = norm_calc
        if isinstance(meta_calc, dict) and meta_calc:
            metadata = {**metadata, **meta_calc}
        out.append(fact_anchors.ClaimRecord(
            claim_type=ctype, surface_form=surface,
            normalized=normalized, metadata=metadata,
        ))
    return out
