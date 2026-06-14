"""Acquire stage: download + extract arxiv source, persist Paper row.

Refactored from `daily_arxiv_translator.{download_source_file,
extract_source_file}` to fit the Stage protocol with idempotency.
"""

from __future__ import annotations

import asyncio
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import Paper, PipelineStage
from xinda.logger_config import setup_logger
from xinda.pipeline.context import PipelineContext
from xinda.pipeline.orchestrator import StageError
from xinda.providers.arxiv_meta import get_arxiv_metadata

logger = setup_logger(__name__)

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"


class Acquire:
    """Stage: ensure arxiv source files exist on disk + Paper row exists."""

    name = PipelineStage.acquire
    recoverable = False

    async def is_done(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> bool:
        src_dir = settings.input_dir / ctx.arxiv_id
        if not src_dir.exists() or not any(src_dir.iterdir()):
            return False
        if ctx.paper_id is None:
            stmt = select(Paper).where(Paper.arxiv_id == ctx.arxiv_id)
            paper = (await session.execute(stmt)).scalar_one_or_none()
            if paper is None:
                return False
            ctx.paper_id = paper.id
        ctx.source_dir = src_dir
        return True

    async def run(
        self, ctx: PipelineContext, session: AsyncSession
    ) -> PipelineContext:
        # 1. metadata + Paper row
        meta = await get_arxiv_metadata(ctx.arxiv_id)
        if meta.get("error"):
            raise StageError(f"arxiv metadata: {meta['error']}")
        ctx.paper_meta = meta

        paper = await self._upsert_paper(ctx, meta, session)
        ctx.paper_id = paper.id

        # 2. source files
        src_dir = settings.input_dir / ctx.arxiv_id
        if src_dir.exists() and any(src_dir.iterdir()):
            logger.info("source for %s already present, skipping download", ctx.arxiv_id)
        else:
            tar_path = await self._download(ctx.arxiv_id)
            await self._extract(tar_path, src_dir)
            tar_path.unlink(missing_ok=True)

            # update Paper status
            paper.download_status = "success"
            paper.extract_status = "success"
            paper.source_path = str(src_dir)
            await session.commit()

        ctx.source_dir = src_dir
        return ctx

    # ──────────────────────────── helpers ────────────────────────────

    async def _upsert_paper(
        self, ctx: PipelineContext, meta: dict, session: AsyncSession
    ) -> Paper:
        stmt = select(Paper).where(Paper.arxiv_id == ctx.arxiv_id)
        paper = (await session.execute(stmt)).scalar_one_or_none()
        if paper is not None:
            return paper

        paper = Paper(
            arxiv_id=ctx.arxiv_id,
            title=meta["title"],
            authors=", ".join(meta.get("authors", [])),
            source_abstract=meta.get("source_abstract"),
            main_category=meta.get("main_category"),
            categories=",".join(meta.get("categories", [])),
            field=meta.get("field"),
            published=(
                datetime.strptime(meta["published"], "%Y-%m-%d").date()
                if meta.get("published") else None
            ),
            updated=(
                datetime.strptime(meta["updated"], "%Y-%m-%d").date()
                if meta.get("updated") else None
            ),
            pdf_url=meta.get("pdf_url"),
            license=meta.get("license"),
            license_label=meta.get("license_label"),
            license_permissive=meta.get("license_permissive"),
        )
        session.add(paper)
        await session.commit()
        await session.refresh(paper)
        return paper

    async def _download(self, arxiv_id: str) -> Path:
        settings.downloads_dir.mkdir(parents=True, exist_ok=True)
        out = settings.downloads_dir / f"{arxiv_id}.tar.gz"
        url = ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
        logger.info("downloading %s", url)
        try:
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as cli:
                resp = await cli.get(url)
                resp.raise_for_status()
                out.write_bytes(resp.content)
        except Exception as e:
            raise StageError(f"download failed: {e}") from e
        return out

    async def _extract(self, tar_path: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(self._extract_sync, tar_path, target_dir)
        except Exception as e:
            raise StageError(f"extract failed: {e}") from e

    @staticmethod
    def _extract_sync(tar_path: Path, target_dir: Path) -> None:
        with tarfile.open(tar_path) as tar:
            # safety: avoid extracting files outside target_dir
            for m in tar.getmembers():
                p = (target_dir / m.name).resolve()
                if not str(p).startswith(str(target_dir.resolve())):
                    raise ValueError(f"unsafe tar member: {m.name}")
            tar.extractall(path=target_dir)


def _copy_to_workspace(src_dir: Path, workspace: Path) -> None:
    """Utility used by Convert stage to materialize source into workspace."""
    workspace.mkdir(parents=True, exist_ok=True)
    for entry in src_dir.iterdir():
        dst = workspace / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst)
