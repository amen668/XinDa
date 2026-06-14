"""PipelineContext — the state object threaded through every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xinda.pipeline.config import PipelineConfig


@dataclass
class PipelineContext:
    """State carried between stages.

    Stages read/write fields on this context; they also persist to DB so the
    pipeline is resumable. `is_done` checks consult DB, not the context.
    """

    arxiv_id: str
    paper_id: int | None              # populated by Acquire
    job_id: int | None                # populated by job creation in route handler
    workspace: Path
    config: PipelineConfig

    # ─── populated by stages ───
    source_dir: Path | None = None        # static/input/{arxiv_id}
    main_tex: str | None = None           # e.g. "main.tex" (with \documentclass)
    xml_src_path: Path | None = None      # post-LaTeXML XML
    xml_tgt_path: Path | None = None
    html_src_path: Path | None = None
    html_tgt_path: Path | None = None
    html_bilingual_path: Path | None = None

    # ─── metadata cache ───
    paper_meta: dict[str, Any] = field(default_factory=dict)
