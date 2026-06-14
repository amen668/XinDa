"""Extract stage: LaTeXML XML → sections tree + translation_units rows.

This is the v3 evolution of the old `extract_html.extract_translation_items`:
- Same per-paragraph extraction logic (placeholder protocol unchanged).
- NEW: also builds a `sections` table tree so that later stages can inject
  section context into translation prompts.
- Persists to DB (papers / sections / translation_units), not just JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from lxml import etree
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    PipelineStage,
    Section,
    TranslationUnit,
    UnitKind,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.translation.placeholders import (
    NS,
    PLACEHOLDER_FORMAT,
    PRESERVE_TAGS,
    SKIP_TAGS,
    SPECIAL_CHARS,
)

logger = setup_logger(__name__)


# Sectional headings — LaTeXML emits <ltx:section title="...">, etc.
_SECTION_TAGS = [
    "{{{}}}part".format(NS["ltx"]),
    "{{{}}}chapter".format(NS["ltx"]),
    "{{{}}}section".format(NS["ltx"]),
    "{{{}}}subsection".format(NS["ltx"]),
    "{{{}}}subsubsection".format(NS["ltx"]),
    "{{{}}}paragraph".format(NS["ltx"]),
    "{{{}}}appendix".format(NS["ltx"]),
]
_SECTION_DEPTH = {tag: i for i, tag in enumerate(_SECTION_TAGS)}


class Extract:
    """Stage: parse XML, build sections + translation_units in DB."""

    name = PipelineStage.extract
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.paper_id is None:
            return False
        stmt = select(TranslationUnit.id).where(
            TranslationUnit.paper_id == ctx.paper_id
        ).limit(1)
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.paper_id is None or ctx.xml_src_path is None:
            raise StageError("Extract requires paper_id and xml_src_path")
        if not ctx.xml_src_path.exists():
            raise StageError(f"XML not found: {ctx.xml_src_path}")

        # Idempotency: clear any partial prior rows for this paper
        # (only safe in M1 — once translations exist, FK cascade complicates this)
        await session.execute(
            delete(TranslationUnit).where(TranslationUnit.paper_id == ctx.paper_id)
        )
        await session.execute(
            delete(Section).where(Section.paper_id == ctx.paper_id)
        )
        await session.commit()

        sections, units = await run_in_threadpool(
            _parse_xml, ctx.xml_src_path
        )

        section_id_by_xpath: dict[str, int] = {}
        for s in sections:
            row = Section(
                paper_id=ctx.paper_id,
                parent_id=section_id_by_xpath.get(s["parent_xpath"]) if s["parent_xpath"] else None,
                ord=s["ord"],
                depth=s["depth"],
                xpath=s["xpath"],
                heading_src=s["heading"],
            )
            session.add(row)
            await session.flush()
            section_id_by_xpath[s["xpath"]] = row.id

        await session.commit()

        for u in units:
            session.add(TranslationUnit(
                paper_id=ctx.paper_id,
                section_id=section_id_by_xpath.get(u["section_xpath"]),
                ord=u["ord"],
                kind=UnitKind(u["kind"]),
                xpath=u["xpath"],
                src_text=u["src_text"],
                src_plain=u["src_plain"],
                placeholders=u["placeholders"],
                special_chars=u["special_chars"],
                char_count=u["char_count"],
            ))
        await session.commit()

        logger.info(
            "extracted %d sections, %d units for paper %d",
            len(sections), len(units), ctx.paper_id,
        )
        return ctx


# ──────────────────────────── XML parser ────────────────────────────


def _parse_xml(xml_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pure-function XML parser. Returns (sections, units) lists.

    Sections list is in document order with each entry carrying its parent's
    xpath (so the DB writer can resolve parent_id via a dict lookup).
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    sections = _collect_sections(tree, root)
    units = _collect_units(tree, root, sections)
    return sections, units


def _collect_sections(tree: etree._ElementTree, root: etree._Element) -> list[dict[str, Any]]:
    """Walk the document, recording every sectional element with depth+parent."""
    rows: list[dict[str, Any]] = []
    section_stack: list[dict[str, Any]] = []  # current ancestor chain
    ord_counters: dict[str, int] = {}         # per-parent ord index

    def walk(elem: etree._Element) -> None:
        tag = elem.tag
        is_section = isinstance(tag, str) and tag in _SECTION_TAGS
        if is_section:
            xpath = tree.getpath(elem)
            parent_xpath = section_stack[-1]["xpath"] if section_stack else None
            ord_key = parent_xpath or "<root>"
            ord_counters[ord_key] = ord_counters.get(ord_key, 0) + 1
            heading = _extract_section_heading(elem)
            row = {
                "xpath": xpath,
                "parent_xpath": parent_xpath,
                "depth": len(section_stack),
                "ord": ord_counters[ord_key],
                "heading": heading,
            }
            rows.append(row)
            section_stack.append(row)
            for child in elem:
                walk(child)
            section_stack.pop()
        else:
            for child in elem:
                walk(child)

    walk(root)
    return rows


def _extract_section_heading(section_elem: etree._Element) -> str | None:
    titles = section_elem.xpath("./ltx:title|./ltx:toctitle", namespaces=NS)
    if not titles:
        return None
    parts: list[str] = []
    for node in titles[0].iter():
        if isinstance(node.tag, str) and node.tag in PRESERVE_TAGS:
            parts.append("[…]")
            if node.tail:
                parts.append(node.tail)
            continue
        if node.text:
            parts.append(node.text)
    return "".join(parts).strip() or None


# Elements yielding translation_units. The mapping → UnitKind decides downstream
# behavior (abstract gets QA-bootstrapped, captions get less context, etc.).
_UNIT_ELEMENT_KIND: dict[str, str] = {
    "{{{}}}p".format(NS["ltx"]): "paragraph",
    "{{{}}}title".format(NS["ltx"]): "title",
    "{{{}}}caption".format(NS["ltx"]): "caption",
}


def _collect_units(
    tree: etree._ElementTree,
    root: etree._Element,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk in document order, emit one record per translatable element."""
    rows: list[dict[str, Any]] = []
    section_xpaths_sorted = sorted(
        (s["xpath"] for s in sections),
        key=lambda x: -len(x),     # longest (deepest) first
    )

    ord_idx = 0
    for elem in root.iter():
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        kind = _UNIT_ELEMENT_KIND.get(tag)
        if kind is None:
            continue

        # title elements inside section headings: classify as section_heading
        if kind == "title":
            parent = elem.getparent()
            if parent is not None and isinstance(parent.tag, str) and parent.tag in _SECTION_TAGS:
                kind = "section_heading"

        # paragraphs inside an <ltx:abstract>: classify as abstract
        if kind == "paragraph":
            anc = elem.getparent()
            while anc is not None:
                if isinstance(anc.tag, str) and anc.tag == "{{{}}}abstract".format(NS["ltx"]):
                    kind = "abstract"
                    break
                anc = anc.getparent()

        placeholders: dict[str, str] = {}
        special_chars: dict[str, str] = {}
        parts: list[str] = []

        def recurse(node: etree._Element) -> None:
            ntag = node.tag
            if isinstance(ntag, str) and ntag in PRESERVE_TAGS:
                tag_name = ntag.split("}")[-1].upper()
                ph = PLACEHOLDER_FORMAT.format(
                    tag=tag_name, index=len(placeholders) + 1
                )
                placeholders[ph] = etree.tostring(
                    node, encoding="unicode", with_tail=False
                )
                parts.append(ph)
                if node.tail:
                    parts.append(node.tail)
                return
            if isinstance(ntag, str) and ntag in SKIP_TAGS:
                if node.tail:
                    parts.append(node.tail)
                return
            if node.text:
                parts.append(node.text)
            for child in node:
                recurse(child)
            if node.tail:
                parts.append(node.tail)

        for child in elem:
            recurse(child)
        text_with_inline = (elem.text or "") + "".join(parts)
        text = text_with_inline.strip()
        if not text:
            continue

        # encode whitespace specials
        for char, ph in SPECIAL_CHARS.items():
            if char in text:
                text = text.replace(char, ph)
                special_chars[ph] = char

        # src_plain = text with placeholders + special-chars stripped (for COMET/tokens)
        src_plain = text
        for ph in placeholders:
            src_plain = src_plain.replace(ph, "")
        for ph in special_chars:
            src_plain = src_plain.replace(ph, " ")
        src_plain = " ".join(src_plain.split())  # collapse whitespace

        xpath = tree.getpath(elem)
        # nearest enclosing section by xpath prefix match
        section_xpath = next(
            (sx for sx in section_xpaths_sorted if xpath.startswith(sx + "/") or xpath == sx),
            None,
        )

        ord_idx += 1
        rows.append({
            "ord": ord_idx,
            "kind": kind,
            "xpath": xpath,
            "src_text": text,
            "src_plain": src_plain,
            "placeholders": placeholders,
            "special_chars": special_chars,
            "char_count": len(src_plain),
            "section_xpath": section_xpath,
        })

    return rows
