"""Convert stage: LaTeX → XML via latexmlc; XML → HTML5 via latexmlpost.

Refactored from `latexml_convert.{convert_tex_to_xml,convert_xml_to_html}`
to fit the Stage protocol. The XML conversion is part of the main pipeline
(M1); HTML conversion is exposed as a free function used later by the
Render stage (M2+).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import PipelineStage
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.pipeline.stages._subprocess import run_command_live
from xinda.pipeline.stages.acquire import _copy_to_workspace

logger = setup_logger(__name__)


class Convert:
    """Stage: copy source to workspace, find main .tex, run latexmlc → XML."""

    name = PipelineStage.convert
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        if ctx.source_dir is None:
            return False
        # workspace/{arxiv}/{ts}/{stem}.xml exists?
        candidate_tex = self._find_main_tex(ctx.workspace) if ctx.workspace.exists() else None
        if candidate_tex is None:
            return False
        stem = Path(candidate_tex).stem
        xml_path = ctx.workspace / f"{stem}.xml"
        if not xml_path.exists() or xml_path.stat().st_size == 0:
            return False
        ctx.main_tex = candidate_tex
        ctx.xml_src_path = xml_path
        return True

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        if ctx.source_dir is None:
            raise StageError("source_dir is None (Acquire didn't run?)")

        # 1. materialize source into workspace
        ctx.workspace.mkdir(parents=True, exist_ok=True)
        _copy_to_workspace(ctx.source_dir, ctx.workspace)

        # 2. find main .tex (contains \documentclass)
        main_tex = self._find_main_tex(ctx.workspace)
        if main_tex is None:
            raise StageError("no main .tex with \\documentclass found")
        ctx.main_tex = main_tex
        stem = Path(main_tex).stem

        # 3. run latexmlc
        tex_path = ctx.workspace / main_tex
        xml_path = ctx.workspace / f"{stem}.xml"
        log_file = ctx.workspace / f"{stem}.latexml.log"
        internal_log = ctx.workspace / f"{stem}.latexml.internal.log"

        def _cmd(include_styles: bool) -> list[str]:
            # --includestyles makes LaTeXML parse the raw .sty/.cls TeX for styles
            # it has no binding for — which crashes on exotic classes that pull heavy
            # expl3/LaTeX3 (e.g. ATLAS/CMS atlasdoc.cls). Dropping it makes LaTeXML
            # use built-in bindings and gracefully SKIP unknown styles: body+math
            # still convert (some custom-macro fidelity lost) instead of total failure.
            return [
                settings.latexmlc_path,
                *(["--includestyles"] if include_styles else []),
                "--preload=arxiv.sty,amsmath.sty",
                "--nocomments",
                f"--path={ctx.workspace}",
                "--format=xml",
                f"--log={internal_log}",
                f"--destination={xml_path}",
                str(tex_path),
            ]

        def _produced() -> bool:
            return xml_path.exists() and xml_path.stat().st_size > 0

        logger.info("latexmlc %s -> %s", tex_path.name, xml_path.name)
        ok = await run_in_threadpool(
            run_command_live, _cmd(True), log_file, settings.latexml_timeout_sec
        )
        if not ok or not _produced():
            # fallback: retry WITHOUT --includestyles (handles exotic .cls + expl3)
            logger.warning(
                "latexmlc failed with --includestyles; retrying without it "
                "(unknown styles will be skipped, not parsed)"
            )
            ok = await run_in_threadpool(
                run_command_live, _cmd(False), log_file, settings.latexml_timeout_sec
            )
        if not ok or not _produced():
            raise StageError(f"latexmlc failed; see {log_file}")

        ctx.xml_src_path = xml_path
        return ctx

    @staticmethod
    def _find_main_tex(workspace: Path) -> str | None:
        for entry in workspace.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".tex":
                continue
            try:
                text = entry.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "\\documentclass" in text:
                return entry.name
        return None


# ────────────────────── XML → HTML free function ─────────────────────────────


async def xml_to_html(
    xml_path: Path,
    html_path: Path,
    extra_opts: dict[str, str | None] | None = None,
    timeout: int | None = None,
) -> bool:
    """Run latexmlpost. Used by Render stage (not part of M1)."""
    log_file = html_path.parent / f"{html_path.stem}.latexmlpost.log"
    internal_log = html_path.parent / f"{html_path.stem}.latexmlpost.internal.log"

    cmd = [
        settings.latexmlpost_path,
        "--format=html5",
        f"--log={internal_log}",
        f"--destination={html_path}",
        "--graphicimages",
    ]
    if extra_opts:
        for opt, val in extra_opts.items():
            cmd.append(opt if val is None else f"{opt}={val}")
    cmd.append(str(xml_path))

    return await asyncio.to_thread(
        run_command_live, cmd, log_file, timeout or settings.latexml_timeout_sec
    )
