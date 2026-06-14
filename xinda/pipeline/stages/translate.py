"""FirstPassTranslate stage: translate every translation_unit.

In M2 this was a bare translation. M5 (this file) adds three layers of
context driven by `ctx.config.use_*` toggles:
- glossary: per-paper terminology table built by GlossaryBuild
- context: section heading + previous-paragraph translation
- fact_anchor: 5-category verifiable claims injected as hard constraints

When all three are off (baselines), behavior reverts to M2.
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    ClaimType,
    GlossaryTerm,
    Paper,
    PipelineStage,
    Section,
    Translation,
    TranslationUnit,
    TuStatus,
    VerifiableClaim,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.base import ModelProvider, TranslationResult
from xinda.providers.factory import create_provider
from xinda.translation.batching import batch_all
from xinda.translation.prompts import (
    TRANSLATION_BATCH_SCHEMA,
    stable_prefix,
    variable_suffix,
)
from xinda.translation.rate_limit import RateLimiter
from xinda.util import parse_translation_array

logger = setup_logger(__name__)


class FirstPassTranslate:
    name = PipelineStage.translate
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.job_id is None or ctx.paper_id is None:
            return False
        n_units = await session.scalar(
            select(func.count(TranslationUnit.id))
            .where(TranslationUnit.paper_id == ctx.paper_id)
        ) or 0
        n_trans = await session.scalar(
            select(func.count(Translation.id))
            .where(Translation.job_id == ctx.job_id, Translation.pass_no == 1)
        ) or 0
        return n_units > 0 and n_trans >= n_units

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("FirstPassTranslate requires job_id and paper_id")

        paper = (
            await session.execute(select(Paper).where(Paper.id == ctx.paper_id))
        ).scalar_one_or_none()
        if paper is None:
            raise StageError(f"paper {ctx.paper_id} missing")

        units = (
            await session.execute(
                select(TranslationUnit)
                .where(TranslationUnit.paper_id == ctx.paper_id)
                .order_by(TranslationUnit.ord)
            )
        ).scalars().all()
        if not units:
            raise StageError("no translation_units; Extract didn't run?")

        sections = (
            await session.execute(
                select(Section)
                .where(Section.paper_id == ctx.paper_id)
                .order_by(Section.depth, Section.ord)
            )
        ).scalars().all()
        section_heading_by_id: dict[int, str] = {
            s.id: s.heading_src for s in sections if s.heading_src
        }
        section_outline = [s.heading_src for s in sections if s.heading_src]

        glossary: list[dict] = []
        if ctx.config.use_glossary:
            rows = (
                await session.execute(
                    select(GlossaryTerm)
                    .where(
                        GlossaryTerm.paper_id == ctx.paper_id,
                        GlossaryTerm.language == ctx.config.language,
                    )
                )
            ).scalars().all()
            glossary = [
                {
                    "src": r.src_term, "tgt": r.tgt_term, "kind": r.kind or "",
                    "definition": r.definition or "", "locked": bool(r.locked),
                } for r in rows
            ]

        # claims per unit (for fact-anchor injection)
        claims_by_unit: dict[int, list[VerifiableClaim]] = defaultdict(list)
        if ctx.config.use_fact_anchor:
            unit_ids = [u.id for u in units]
            crows = (
                await session.execute(
                    select(VerifiableClaim).where(VerifiableClaim.unit_id.in_(unit_ids))
                )
            ).scalars().all()
            for c in crows:
                claims_by_unit[c.unit_id].append(c)

        provider = create_provider(ctx.config.pass1_model)
        limiter = RateLimiter(provider.rpm, provider.tpm)

        prefix = stable_prefix(
            paper_title=paper.title, arxiv_id=paper.arxiv_id,
            field=paper.field or "Other",
            target_language=ctx.config.language,
            glossary_terms=glossary if ctx.config.use_glossary else None,
            section_outline=section_outline if ctx.config.use_context else None,
            abstract=paper.source_abstract if ctx.config.use_context else None,
        )

        # Items as plain dicts for batching
        items: list[dict[str, Any]] = []
        for u in units:
            it: dict[str, Any] = {
                "id": u.id, "src_text": u.src_text, "char_count": u.char_count,
                "_unit": u,
            }
            items.append(it)

        batches = batch_all(items, token_fn=provider.estimate_tokens)
        logger.info(
            "first-pass: %d units → %d batches (model=%s lang=%s glossary=%s fact=%s ctx=%s)",
            len(items), len(batches), provider.model_name, ctx.config.language,
            ctx.config.use_glossary, ctx.config.use_fact_anchor, ctx.config.use_context,
        )

        # For previous-paragraph context: build a map of unit_id -> tgt_text from
        # prior batches as they complete. We process batches sequentially within
        # each section to preserve coherence; between sections we parallelize.
        # Simplification: run batches sequentially for now (M5 quality > M5 speed).
        sem = asyncio.Semaphore(settings.max_concurrency)

        results_per_batch: list[list[dict[str, Any]]] = []
        prev_tgt: dict[int, str] = {}  # unit_id -> tgt_text (for "prev" context)
        for bi, batch in enumerate(batches):
            # forward context: the source text of the next batch (read-only) so
            # the model can resolve cataphora / terms introduced only downstream.
            next_source = None
            if ctx.config.use_context and bi + 1 < len(batches):
                next_source = " ".join(
                    it["_unit"].src_plain for it in batches[bi + 1]
                )
            async with sem:
                res = await _translate_batch(
                    provider, limiter, prefix, batch,
                    glossary=glossary,
                    section_heading_by_id=section_heading_by_id,
                    claims_by_unit=claims_by_unit,
                    prev_tgt=prev_tgt,
                    next_source=next_source,
                    config_use_context=ctx.config.use_context,
                    config_use_glossary=ctx.config.use_glossary,
                    config_use_fact_anchor=ctx.config.use_fact_anchor,
                )
                results_per_batch.append(res)
                # update prev_tgt for next batch
                for r in res:
                    if not r.get("fallback"):
                        prev_tgt[r["id"]] = r["tgt_text"]

        # Flatten and persist
        translated_by_id: dict[int, dict[str, Any]] = {}
        for batch_res in results_per_batch:
            for r in batch_res:
                translated_by_id[r["id"]] = r

        missing = [u for u in units if u.id not in translated_by_id]
        if missing:
            logger.warning("first-pass: %d units fell to source fallback", len(missing))
            for u in missing:
                translated_by_id[u.id] = {
                    "id": u.id, "tgt_text": u.src_text,
                    "cached": 0, "fresh": 0, "completion": 0,
                    "elapsed_ms": 0, "fallback": True,
                }

        for u in units:
            r = translated_by_id[u.id]
            tgt_text = r["tgt_text"]
            tgt_plain = _strip_placeholders(tgt_text, u.placeholders, u.special_chars)
            glossary_hits_meta = r.get("glossary_hits")
            session.add(Translation(
                job_id=ctx.job_id, unit_id=u.id,
                status=TuStatus.fallback if r.get("fallback") else TuStatus.translated,
                pass_no=1, model_used=provider.model_name,
                tgt_text=tgt_text, tgt_plain=tgt_plain,
                cached_prompt_tokens=r.get("cached", 0),
                fresh_prompt_tokens=r.get("fresh", 0),
                completion_tokens=r.get("completion", 0),
                elapsed_ms=r.get("elapsed_ms", 0),
                glossary_hits=glossary_hits_meta,
            ))
        await session.commit()
        return ctx


# ────────────────────────── helpers ──────────────────────────


async def _translate_batch(
    provider: ModelProvider,
    limiter: RateLimiter,
    stable_prefix_text: str,
    batch: list[dict[str, Any]],
    *,
    glossary: list[dict],
    section_heading_by_id: dict[int, str],
    claims_by_unit: dict[int, list[VerifiableClaim]],
    prev_tgt: dict[int, str],
    next_source: str | None = None,
    config_use_context: bool,
    config_use_glossary: bool,
    config_use_fact_anchor: bool,
) -> list[dict[str, Any]]:
    units = [it["_unit"] for it in batch]
    # 1. section heading: use the first unit's section (batches are
    # paragraph-contiguous in practice).
    section_heading = None
    if config_use_context:
        first_section_id = units[0].section_id if units else None
        if first_section_id is not None:
            section_heading = section_heading_by_id.get(first_section_id)

    # 2. previous translation: use the immediately-prior unit by ord
    prev = None
    if config_use_context and units:
        first_ord = units[0].ord
        for prev_id, prev_text in prev_tgt.items():
            # quick heuristic: take the most recently completed (highest ord
            # less than first_ord). prev_tgt is unit-id keyed so we scan; this
            # is O(N²) overall but N=number of batches.
            pass
        # Cheaper: pass the most-recent translation we have at all (it'll be
        # the prior batch's last item by construction).
        if prev_tgt:
            last_id = next(reversed(prev_tgt))
            prev = prev_tgt.get(last_id)

    # 3. fact anchors: union of all claims in this batch (grouped by type)
    fact_anchors = None
    if config_use_fact_anchor:
        grouped: dict[str, list[str]] = {ct.value: [] for ct in ClaimType}
        seen: set[tuple[str, str]] = set()
        for u in units:
            for c in claims_by_unit.get(u.id, []):
                key = (c.claim_type.value, c.surface_form)
                if key in seen:
                    continue
                seen.add(key)
                grouped[c.claim_type.value].append(c.surface_form)
        fact_anchors = grouped

    # 4. glossary hits in this batch's text
    hits: list[dict] = []
    hits_meta: dict[str, Any] | None = None
    if config_use_glossary and glossary:
        text_blob = " ".join(u.src_plain for u in units)
        hits = _find_glossary_hits(text_blob, glossary)
        if hits:
            hits_meta = {"hits": [h["src"] for h in hits]}

    suffix = variable_suffix(
        batch,
        section_heading=section_heading,
        prev_translation=prev,
        next_source=next_source if config_use_context else None,
        glossary_hits=hits if hits else None,
        fact_anchors=fact_anchors,
    )

    est = provider.estimate_tokens(stable_prefix_text) + provider.estimate_tokens(suffix)
    await limiter.reserve(est)

    try:
        tr: TranslationResult = await provider.generate(
            prompt=suffix, system=stable_prefix_text,
            json_schema=TRANSLATION_BATCH_SCHEMA,
        )
    except Exception as e:
        logger.warning("batch failed (%d items): %s — fallback to source", len(batch), e)
        return [_fallback(it) for it in batch]

    parsed = _parse_response(tr.text)
    if parsed is None:
        logger.warning("batch JSON parse failed; fallback to source for %d items", len(batch))
        return [_fallback(it) for it in batch]

    by_id = {it["id"]: it for it in batch}
    out: list[dict[str, Any]] = []
    for arr in parsed:
        if not isinstance(arr, (list, tuple)) or len(arr) < 2:
            continue
        _id, tgt = arr[0], arr[1]
        if isinstance(_id, str) and _id.isdigit():
            _id = int(_id)
        if _id not in by_id:
            continue
        out.append({
            "id": _id, "tgt_text": tgt,
            "cached": tr.cached_prompt_tokens // max(1, len(batch)),
            "fresh": tr.fresh_prompt_tokens // max(1, len(batch)),
            "completion": tr.completion_tokens // max(1, len(batch)),
            "elapsed_ms": tr.elapsed_ms,
            "fallback": False,
            "glossary_hits": hits_meta,
        })
    return out


def _find_glossary_hits(text: str, glossary: list[dict]) -> list[dict]:
    hits = []
    for term in glossary:
        flags = 0 if term.get("kind") == "acronym" else re.IGNORECASE
        try:
            if re.search(r"\b" + re.escape(term["src"]) + r"\b", text, flags):
                hits.append(term)
        except re.error:
            continue
    return hits


def _parse_response(text: str) -> list | None:
    return parse_translation_array(text)


def _fallback(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"], "tgt_text": item["src_text"],
        "cached": 0, "fresh": 0, "completion": 0,
        "elapsed_ms": 0, "fallback": True,
    }


def _strip_placeholders(
    text: str, placeholders: dict[str, str], special_chars: dict[str, str]
) -> str:
    out = text
    for ph in placeholders:
        out = out.replace(ph, "")
    for ph in special_chars:
        out = out.replace(ph, " ")
    return " ".join(out.split())
