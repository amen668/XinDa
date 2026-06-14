"""GlossaryBuild stage: per-(paper, language) terminology table from full paper.

Runs after FactExtract, before FirstPassTranslate. Uses 1M-context model
(`settings.model_glossary`) to feed the entire paper text in one shot for
more thorough term extraction than v1's (title+abstract+first-para) input.

Idempotency: skip if glossary_terms rows already exist for (paper, lang).
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    GlossaryTerm,
    Paper,
    PipelineStage,
    TranslationUnit,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER, language_name

logger = setup_logger(__name__)


def _system_prompt(target_lang: str, paper_title: str, field: str) -> str:
    name = language_name(target_lang)
    return (
        "You are a terminology-extraction expert for scientific papers.\n"
        f'Current paper: "{paper_title}", field: {field}.\n'
        f"Target language: {name}.\n\n"
        "From the full paper, identify 20-60 term pairs in these three kinds:\n"
        "- acronym: an abbreviation; tgt = the English acronym itself (locked=true), "
        "definition = expansion (optional)\n"
        "- proper_noun: a proper name (model/dataset/method name, author/institution); "
        "tgt = the English original (locked=true)\n"
        f"- technical_term: a technical term that has a standard {name} rendering; "
        "tgt = that standard rendering (locked=false)\n\n"
        "Rules:\n"
        "- No common English words\n"
        "- Each term only once\n"
        "- Extract 20-60; fewer than 15 counts as under-extraction\n\n"
        "Output a JSON object with field `terms`, an array of objects each with "
        "src/tgt/kind/definition/locked fields.\n"
        + STRICT_JSON_FOOTER
    )


_GLOSSARY_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "tgt": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["acronym", "proper_noun", "technical_term"],
                    },
                    "definition": {"type": "string"},
                    "locked": {"type": "boolean"},
                },
                "required": ["src", "tgt", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["terms"],
    "additionalProperties": False,
}


class GlossaryBuild:
    """Stage: extract per-language glossary from full paper text."""

    name = PipelineStage.glossary
    recoverable = True

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.paper_id is None:
            return False
        if not ctx.config.use_glossary:
            return True  # ablation disabled — skip
        n = await session.scalar(
            select(func.count(GlossaryTerm.id))
            .where(
                GlossaryTerm.paper_id == ctx.paper_id,
                GlossaryTerm.language == ctx.config.language,
            )
        ) or 0
        return n > 0

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.paper_id is None:
            raise StageError("GlossaryBuild requires paper_id")

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
            raise StageError("no units; Extract didn't run?")

        # Chunk the paper for term extraction. A single whole-paper prompt blows the
        # provider's hard per-call timeout on large papers (a 989-unit paper trickled
        # past the 300s wait_for and was cancelled → zero terms). Chunks are sized to
        # finish well within that budget; terms are extracted per chunk and merged.
        chunks = _chunk_units(units, _GLOSSARY_CHUNK_CHARS, _GLOSSARY_MAX_CHUNKS)
        if not chunks:
            logger.warning("glossary build: no source text for paper %d", paper.id)
            return ctx

        provider = create_provider(settings.model_glossary)
        system = _system_prompt(
            ctx.config.language, paper.title, paper.field or "Other",
        )

        async def _extract_chunk(text: str) -> list[dict]:
            user_prompt = (
                f"<PAPER_EXCERPT>\n{text}\n</PAPER_EXCERPT>\n\n"
                "Extract the glossary from this excerpt."
            )
            try:
                tr = await provider.generate(
                    prompt=user_prompt, system=system, json_schema=_GLOSSARY_SCHEMA,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("glossary chunk failed for paper %d: %s", paper.id, e)
                return []
            return _parse_terms(tr.text)

        chunk_terms = await asyncio.gather(*[_extract_chunk(c) for c in chunks])
        terms = [t for sub in chunk_terms for t in sub]
        if not terms:
            logger.warning("glossary build: zero terms parsed for paper %d", paper.id)
            return ctx
        if len(chunks) > 1:
            logger.info(
                "glossary: %d raw terms from %d chunks for paper %d",
                len(terms), len(chunks), paper.id,
            )

        # Optional grounding: override the LLM's target rendering with an
        # authoritative term-bank entry where one exists.
        resolver = None
        if ctx.config.use_external_glossary:
            from xinda.translation.glossary_grounding import (
                GroundingResolver,
            )
            try:
                resolver = GroundingResolver(ctx.config.language)
                if not resolver.available:
                    resolver = None
            except Exception as e:  # noqa: BLE001
                logger.warning("glossary grounding unavailable: %s", e)

        # Persist (dedup on src_term)
        seen: set[str] = set()
        grounded = 0
        for t in terms:
            src = (t.get("src") or "").strip()
            tgt = (t.get("tgt") or "").strip()
            if not src or not tgt or src.lower() in seen:
                continue
            seen.add(src.lower())
            kind = t.get("kind") or "technical_term"
            locked = bool(t.get("locked", kind in ("acronym", "proper_noun")))
            grounding_source = None
            if resolver is not None:
                hit = resolver.ground(src, kind)
                if hit is not None:
                    tgt = hit.tgt
                    grounding_source = hit.source
                    locked = True  # authoritative rendering → lock it
                    grounded += 1
            session.add(GlossaryTerm(
                paper_id=paper.id,
                language=ctx.config.language,
                src_term=src,
                tgt_term=tgt,
                kind=kind,
                definition=t.get("definition"),
                locked=locked,
                confidence=1.0 if grounding_source else 0.8,
                grounding_source=grounding_source,
            ))
        await session.commit()
        logger.info(
            "glossary: %d terms for paper %d / lang %s (%d grounded externally)",
            len(seen), paper.id, ctx.config.language, grounded,
        )
        return ctx


# Per-chunk char budget for glossary extraction. Kept conservatively small so each
# call finishes well inside the provider's 300s hard timeout even on dense
# math/text (100k still tripped the timeout on ~1/4 of a 989-unit q-bio paper);
# a cap on chunk count bounds cost on pathological papers. Per-chunk failures are
# isolated — one slow chunk no longer zeroes the whole glossary.
_GLOSSARY_CHUNK_CHARS = 60_000
_GLOSSARY_MAX_CHUNKS = 20


def _chunk_units(units, budget: int, max_chunks: int) -> list[str]:
    """Group units (document order) into text chunks of at most `budget` chars.

    A single oversize unit becomes its own chunk. Stops after `max_chunks` chunks
    (the tail is dropped — glossary recall is already high from the body)."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for u in units:
        t = (u.src_plain or "").strip()
        if not t:
            continue
        if cur and size + len(t) > budget:
            chunks.append("\n\n".join(cur))
            if len(chunks) >= max_chunks:
                return chunks
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur and len(chunks) < max_chunks:
        chunks.append("\n\n".join(cur))
    return chunks


def _parse_terms(text: str) -> list[dict]:
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
    if isinstance(obj, dict) and "terms" in obj:
        terms = obj["terms"]
        return terms if isinstance(terms, list) else []
    if isinstance(obj, list):
        return obj
    return []
