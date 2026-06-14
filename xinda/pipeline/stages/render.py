"""Render stage: XML(src) + XML(tgt) → HTML5 (latexmlpost) + bilingual.

M8 enhancements over M2:
- inject `data-unit-id` / `data-section-id` attributes onto rendered HTML
  elements so the frontend can do DOM-id-anchored sync scrolling
- embed `<script id="quality-metadata">` with fallback / refined unit IDs
  for client-side overlay highlighting
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from jinja2 import Template
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.db.models import (
    PipelineStage,
    RenderedFile,
    Translation,
    TranslationUnit,
    TuStatus,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.pipeline.stages.convert import xml_to_html

logger = setup_logger(__name__)


# location of the existing v1 templates / assets (relocated to frontend/ in M8)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATE_PATH = _REPO_ROOT / "static" / "templates" / "comparison.html"
_CSS_PATH = _REPO_ROOT / "static" / "css" / "comparison.css"
_JS_PATH = _REPO_ROOT / "static" / "js" / "comparison.js"


class Render:
    """Stage: produce html_src, html_tgt, html_bilingual files."""

    name = PipelineStage.render
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if (
            ctx.html_src_path is not None and ctx.html_src_path.exists()
            and ctx.html_tgt_path is not None and ctx.html_tgt_path.exists()
            and ctx.html_bilingual_path is not None and ctx.html_bilingual_path.exists()
        ):
            return True
        if ctx.job_id is None:
            return False
        rows = (
            await session.execute(
                select(RenderedFile).where(RenderedFile.job_id == ctx.job_id)
            )
        ).scalars().all()
        seen = {r.kind: Path(r.storage_path) for r in rows}
        ok = all(
            k in seen and seen[k].exists()
            for k in ("html_src", "html_tgt", "html_bilingual")
        )
        if ok:
            ctx.html_src_path = seen["html_src"]
            ctx.html_tgt_path = seen["html_tgt"]
            ctx.html_bilingual_path = seen["html_bilingual"]
        return ok

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.xml_src_path is None or not ctx.xml_src_path.exists():
            raise StageError("xml_src missing")
        if ctx.xml_tgt_path is None or not ctx.xml_tgt_path.exists():
            raise StageError("xml_tgt missing (ApplyXML didn't run?)")
        if ctx.job_id is None:
            raise StageError("Render needs job_id")

        stem = ctx.xml_src_path.stem
        html_src = ctx.workspace / f"{stem}_en.html"
        html_tgt = ctx.workspace / f"{stem}_{ctx.config.language}.html"
        html_bilingual = ctx.workspace / f"{stem}_bilingual.html"

        # parallel latexmlpost
        opts = {"--graphicimages": None}
        ok_src, ok_tgt = await asyncio.gather(
            xml_to_html(ctx.xml_src_path, html_src, opts),
            xml_to_html(ctx.xml_tgt_path, html_tgt, opts),
        )
        if not (ok_src and html_src.exists()):
            raise StageError(f"latexmlpost(src) failed for {ctx.xml_src_path}")
        if not (ok_tgt and html_tgt.exists()):
            raise StageError(f"latexmlpost(tgt) failed for {ctx.xml_tgt_path}")

        # Inject data-unit-id attributes for DOM-anchored sync scrolling
        units = (
            await session.execute(
                select(TranslationUnit)
                .where(TranslationUnit.paper_id == ctx.paper_id)
                .order_by(TranslationUnit.ord)
            )
        ).scalars().all()

        translations = (
            await session.execute(
                select(Translation).where(Translation.job_id == ctx.job_id)
            )
        ).scalars().all()

        fallback_unit_ids = [
            t.unit_id for t in translations if t.status == TuStatus.fallback
        ]
        refined_unit_ids = [
            t.unit_id for t in translations
            if t.status == TuStatus.refined and t.pass_no > 1
        ]
        quality_metadata = {
            "fallback_unit_ids": fallback_unit_ids,
            "refined_unit_ids": refined_unit_ids,
        }

        await asyncio.to_thread(
            _inject_unit_ids, html_src, html_tgt, units, quality_metadata,
        )

        await asyncio.to_thread(
            _render_bilingual, html_src, html_tgt, html_bilingual
        )

        # persist rendered_files rows
        for kind, p in (
            ("xml_src", ctx.xml_src_path),
            ("xml_tgt", ctx.xml_tgt_path),
            ("html_src", html_src),
            ("html_tgt", html_tgt),
            ("html_bilingual", html_bilingual),
        ):
            session.add(RenderedFile(
                job_id=ctx.job_id, kind=kind,
                storage_path=str(p),
                size_bytes=p.stat().st_size if p.exists() else None,
            ))
        await session.commit()

        ctx.html_src_path = html_src
        ctx.html_tgt_path = html_tgt
        ctx.html_bilingual_path = html_bilingual
        logger.info("rendered bilingual %s", html_bilingual)
        return ctx


# ────────────────────────── data-unit-id injection ──────────────────────────


_LTX_PARA_CLASS_RE = re.compile(
    r'<(?P<tag>div|p)\s+([^>]*?)class="(?P<cls>[^"]*?ltx_(?:para|p|title|caption)[^"]*?)"'
    r'([^>]*?)>',
    re.IGNORECASE,
)


def _inject_unit_ids(
    html_src: Path,
    html_tgt: Path,
    units,
    quality_metadata: dict,
) -> None:
    """Post-process LaTeXML HTML to add data-unit-id attributes.

    Heuristic: walk paragraph/title/caption elements in document order and
    assign translation_units IDs by their order index. Both src and tgt
    HTML use the SAME numbering (since they're rendered from XMLs with the
    same xpath structure), enabling frontend cross-pairing.
    """
    for path in (html_src, html_tgt):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        unit_ord_iter = iter([u.id for u in units])

        def _replace(match: re.Match) -> str:
            try:
                uid = next(unit_ord_iter)
            except StopIteration:
                return match.group(0)
            return (
                f'<{match.group("tag")} '
                f'{match.group(2)}'
                f'class="{match.group("cls")}" '
                f'data-unit-id="{uid}"'
                f'{match.group(4)}>'
            )

        content = _LTX_PARA_CLASS_RE.sub(_replace, content)

        # Inject quality metadata as a script block in <head>
        script = (
            f'<script id="quality-metadata" type="application/json">'
            f'{json.dumps(quality_metadata, ensure_ascii=False)}'
            f"</script>"
        )
        if "<head>" in content.lower():
            content = re.sub(
                r"(<head[^>]*>)", r"\1" + script, content, count=1, flags=re.IGNORECASE
            )
        else:
            content = script + content

        path.write_text(content, encoding="utf-8")


# ────────────────────────── bilingual renderer ──────────────────────────


def _render_bilingual(html_src: Path, html_tgt: Path, out: Path) -> None:
    if not _TEMPLATE_PATH.exists():
        raise StageError(f"template missing: {_TEMPLATE_PATH}")

    original = html_src.read_text(encoding="utf-8")
    translated = html_tgt.read_text(encoding="utf-8")
    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    rendered = template.render(
        original_content=original, translated_content=translated
    )

    # copy assets next to the bilingual file
    if _CSS_PATH.exists():
        shutil.copy2(_CSS_PATH, out.parent / "comparison.css")
    if _JS_PATH.exists():
        shutil.copy2(_JS_PATH, out.parent / "comparison.js")

    out.write_text(rendered, encoding="utf-8")
