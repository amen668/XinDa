"""ApplyXML stage: write translations back into the source XML by xpath.

Logic mirrors v1's apply_translation.apply_translations_to_xml:
- Load latest pass Translation row per unit (highest pass_no, status≠pending).
- Sort by xpath depth (deepest first) so child substitution doesn't disrupt
  parent xpaths.
- For each unit, look up the source XML element, clear it, and rewrite
  with translation text — restoring placeholders into wrapper subtree.

M2 keeps the v1 case-insensitive placeholder repair as a safety net (the
prompt asks the model to preserve placeholders verbatim, but Qwen still
occasionally mangles casing on first pass). The flag will be removed in
M6 once fact-anchor verification is enforcing strict placeholder fidelity.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from lxml import etree
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    PipelineStage,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.translation.placeholders import LTX_NS, NS

logger = setup_logger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+?\}\}")


class ApplyXML:
    """Stage: write translations back into the original XML."""

    name = PipelineStage.apply
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.xml_tgt_path is not None and ctx.xml_tgt_path.exists():
            return ctx.xml_tgt_path.stat().st_size > 0
        return False

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.xml_src_path is None or not ctx.xml_src_path.exists():
            raise StageError("xml_src_path missing (Convert didn't run?)")
        if ctx.job_id is None:
            raise StageError("ApplyXML needs job_id")

        # Build (unit, latest_translation) pairs ordered by xpath depth desc.
        units = (
            await session.execute(
                select(TranslationUnit)
                .where(TranslationUnit.paper_id == ctx.paper_id)
                .order_by(TranslationUnit.ord)
            )
        ).scalars().all()

        translations = (
            await session.execute(
                select(Translation)
                .where(
                    Translation.job_id == ctx.job_id,
                    Translation.status != TuStatus.pending,
                )
                .order_by(Translation.unit_id, Translation.pass_no.desc())
            )
        ).scalars().all()

        latest_by_unit: dict[int, Translation] = {}
        for t in translations:
            latest_by_unit.setdefault(t.unit_id, t)  # first (highest pass) wins

        # Prepare apply-list, deepest xpath first
        plan: list[dict[str, Any]] = []
        for u in units:
            t = latest_by_unit.get(u.id)
            if t is None or not t.tgt_text:
                continue
            plan.append({
                "xpath": u.xpath,
                "tgt_text": t.tgt_text,
                "placeholders": u.placeholders or {},
                "special_chars": u.special_chars or {},
            })
        plan.sort(key=lambda p: p["xpath"].count("/"), reverse=True)

        stem = Path(ctx.main_tex or "main.tex").stem
        out_path = ctx.workspace / f"{stem}_{ctx.config.language}.xml"

        await run_in_threadpool(_apply_sync, ctx.xml_src_path, out_path, plan)

        ctx.xml_tgt_path = out_path
        logger.info("apply: wrote %s", out_path)
        return ctx


# ────────────────────────── pure-fn XML rewriter ──────────────────────────


def _apply_sync(xml_in: Path, xml_out: Path, plan: list[dict[str, Any]]) -> None:
    tree = etree.parse(str(xml_in))
    root = tree.getroot()

    for entry in plan:
        xpath = entry["xpath"]
        tgt = entry["tgt_text"]
        placeholders = entry["placeholders"]
        special_chars = entry["special_chars"]

        # restore special chars first (\n etc.)
        if special_chars:
            for ph, ch in special_chars.items():
                tgt = tgt.replace(ph, ch)

        try:
            elements = root.xpath(xpath, namespaces=NS)
        except etree.XPathEvalError as e:
            logger.warning("bad xpath %s: %s", xpath, e)
            continue

        for elem in elements:
            elem.clear()
            _rewrite_element(elem, tgt, placeholders)

    xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(xml_out), encoding="utf-8", pretty_print=True, xml_declaration=True)


def _rewrite_element(
    elem: etree._Element, tgt: str, placeholders: dict[str, str]
) -> None:
    if not placeholders:
        _set_text_or_fragment(elem, tgt)
        return

    # case-insensitive placeholder repair (v1 bandaid carried forward)
    lower_map: dict[str, dict[str, str]] = {}
    for ph, original in placeholders.items():
        lower_map[ph.lower()] = {"original_ph": ph, "content": original}

    def replace_one(match: re.Match[str]) -> str:
        found = match.group(0)
        flat = re.sub(r"\s", "", found.lower())
        for key, entry in lower_map.items():
            if found.lower() == key or flat == re.sub(r"\s", "", key):
                return entry["content"]
        logger.debug("unmatched placeholder in translation: %s", found)
        return found

    expanded = _PLACEHOLDER_RE.sub(replace_one, tgt)
    _set_text_or_fragment(elem, expanded)


def _set_text_or_fragment(elem: etree._Element, content: str) -> None:
    """Try to parse `content` as an XML fragment (it contains inline tags from
    the placeholder restoration); fall back to plain text if it doesn't parse.
    """
    wrapper = f'<wrapper xmlns="{LTX_NS}">{content}</wrapper>'
    try:
        fragment = etree.fromstring(wrapper)
    except etree.XMLSyntaxError as e:
        logger.warning("XML fragment parse failed: %s; using plain text", e)
        elem.text = content
        return
    elem.text = fragment.text
    for child in fragment:
        elem.append(child)


def _iter_unique_xpaths(units: Iterable[TranslationUnit]) -> set[str]:
    return {u.xpath for u in units}
