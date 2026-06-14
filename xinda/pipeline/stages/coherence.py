"""Coherence stage: whole-paper discourse harmonization via 1M context.

Per-paragraph translation (even with PREV/NEXT windows) can still leave seam
artifacts across batch boundaries: a recurring non-glossary phrase rendered two
ways, a discourse connective ("however/therefore") that contradicts the prior
paragraph's logic, or a pronoun/reference whose antecedent sits paragraphs away.

This stage feeds the ENTIRE translated paper (in order, placeholders intact) to
the model in one call and asks it to FIX only cross-paragraph inconsistencies —
terminology, connectives, references — WITHOUT changing facts (numbers,
citations, comparison direction) and WITHOUT re-translating. It returns only the
units it edited; each edit lands as a new (higher) translation pass, which
ApplyXML then picks up automatically.

This is the discourse-coherence sibling of CrossDocFactVerify (which only
*detects* fact drift): here we *repair* style/term/reference drift.
"""

from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    GlossaryTerm,
    PipelineStage,
    Translation,
    TranslationJob,
    TranslationUnit,
    TuStatus,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.pipeline.stages.refine import _strip_plain
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER, language_name

logger = setup_logger(__name__)

# Stages at/after which this one counts as already done (idempotency via the
# job's resumability cursor, since a clean run may legitimately edit 0 units).
_DONE_AT = {
    PipelineStage.coherence, PipelineStage.apply,
    PipelineStage.render, PipelineStage.evaluate,
}


def _coherence_system(target_name: str) -> str:
    return (
        "You are a discourse-consistency editor for scientific-paper translations.\n"
        f"You will receive the full {target_name} translation of an English paper (in "
        "paragraph order) plus the paper's glossary.\n\n"
        "Your task: **fix cross-paragraph inconsistencies ONLY**, nothing else:\n"
        "- a term/proper noun translated differently across paragraphs → unify to the "
        "glossary's target form (or the single most appropriate one paper-wide)\n"
        "- a connective/logical link contradicting context (e.g. the previous paragraph says "
        "'increasing', this one wrongly says 'however it drops') → correct it\n"
        "- a reference/anaphora broken or unclear due to paragraph-wise translation → smooth it\n\n"
        "Strictly forbidden:\n"
        "- Do not change ANY fact: numbers, precision, cited object, comparison direction stay "
        "untouched\n"
        "- Do not re-translate or rewrite sentences that have no problem\n"
        "- Keep every {{PT_XXX_N}}, {{NL}}, {{TAB}}, {{RE}} placeholder verbatim\n"
        f"- The text is already in {target_name}; do not translate it\n\n"
        "Output ONLY the units you actually changed. JSON object, field `edits`, an array of "
        "[unit_id, full revised translation of that unit]. If nothing needs changing, edits is "
        "an empty array.\n"
        + STRICT_JSON_FOOTER
    )

_COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": ["integer", "string"]},
                "minItems": 2,
                "maxItems": 2,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


class Coherence:
    name = PipelineStage.coherence
    recoverable = True

    async def is_done(self, ctx: PipelineContext, session: AsyncSession) -> bool:
        if not ctx.config.use_coherence:
            return True  # ablation
        if ctx.job_id is None:
            return False
        job = await session.get(TranslationJob, ctx.job_id)
        return job is not None and job.last_stage in _DONE_AT

    async def run(self, ctx: PipelineContext, session: AsyncSession) -> PipelineContext:
        if not ctx.config.use_coherence:
            logger.info("coherence skipped by config")
            return ctx
        if ctx.job_id is None or ctx.paper_id is None:
            raise StageError("Coherence requires job_id/paper_id")

        # latest pass per unit, in document order
        rows = (
            await session.execute(
                select(TranslationUnit, Translation)
                .join(Translation, Translation.unit_id == TranslationUnit.id)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.status != TuStatus.pending,
                    TranslationUnit.paper_id == ctx.paper_id,
                )
                .order_by(TranslationUnit.ord, Translation.pass_no.desc())
            )
        ).all()
        latest: dict[int, tuple[TranslationUnit, Translation]] = {}
        maxpass: dict[int, int] = {}
        for u, t in rows:
            if u.id not in latest:
                latest[u.id] = (u, t)
            maxpass[u.id] = max(maxpass.get(u.id, 0), t.pass_no)
        if not latest:
            raise StageError("no translations to harmonize")
        ordered = sorted(latest.values(), key=lambda x: x[0].ord)

        # whole-doc body (placeholders intact), capped to fit context
        body_lines: list[str] = []
        total = 0
        for u, t in ordered:
            line = f"[unit {u.id}] {t.tgt_text or ''}"
            if total + len(line) > 600_000:
                break
            body_lines.append(line)
            total += len(line)
        body = "\n\n".join(body_lines)

        gterms = (
            await session.execute(
                select(GlossaryTerm).where(
                    GlossaryTerm.paper_id == ctx.paper_id,
                    GlossaryTerm.language == ctx.config.language,
                )
            )
        ).scalars().all()
        glossary = "\n".join(
            f"- {g.src_term} → {g.tgt_term}" + (" [locked]" if g.locked else "")
            for g in gterms
        ) or "(none)"

        user_prompt = (
            f"<GLOSSARY>\n{glossary}\n</GLOSSARY>\n\n"
            f"<TRANSLATED_PAPER>\n{body}\n</TRANSLATED_PAPER>\n\n"
            "Output only the units that must be revised to remove cross-paragraph "
            "inconsistencies."
        )

        provider = create_provider(settings.model_refine)
        try:
            tr = await provider.generate(
                prompt=user_prompt,
                system=_coherence_system(language_name(ctx.config.language)),
                json_schema=_COHERENCE_SCHEMA,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("coherence pass failed for job %d: %s", ctx.job_id, e)
            return ctx

        edits = _parse_edits(tr.text)
        unit_by_id = {u.id: u for u, _ in ordered}
        applied = 0
        for unit_id, new_text in edits:
            u = unit_by_id.get(unit_id)
            if u is None or not new_text:
                continue
            session.add(Translation(
                job_id=ctx.job_id, unit_id=u.id,
                status=TuStatus.refined,
                pass_no=maxpass.get(u.id, 1) + 1,
                model_used=f"coherence:{provider.model_name}",
                tgt_text=new_text,
                tgt_plain=_strip_plain(new_text, u.placeholders, u.special_chars),
            ))
            applied += 1
        await session.commit()
        logger.info("coherence: harmonized %d/%d units (job %d)",
                    applied, len(ordered), ctx.job_id)
        return ctx


def _parse_edits(text: str) -> list[tuple[int, str]]:
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
    raw = obj.get("edits") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[tuple[int, str]] = []
    for e in raw:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        uid, txt = e[0], e[1]
        if isinstance(uid, str) and uid.isdigit():
            uid = int(uid)
        if isinstance(uid, int) and isinstance(txt, str):
            out.append((uid, txt))
    return out
