"""Benchmark CLI: run the full evaluation matrix and export tables.

Usage:
    # run 100 papers × 3 langs × all variants
    python -m xinda.cli.benchmark \\
        --papers paper_ids.txt --langs zh,fr,es --variants all

    # export the standing matrix to CSV + LaTeX (no re-runs)
    python -m xinda.cli.benchmark --export-only --out-dir results/
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from xinda.evaluation.benchmark import (
    export_csv,
    export_latex_appendix_table,
    export_latex_main_table,
    run_matrix,
)
from xinda.logger_config import setup_logger
from xinda.pipeline.config import variants_for

logger = setup_logger(__name__)

ALL_VARIANTS = list(variants_for("zh").keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--papers", type=Path,
        help="path to file with one arxiv_id per line",
    )
    p.add_argument(
        "--langs", default="zh,fr,es",
        help="comma-separated target languages",
    )
    p.add_argument(
        "--variants", default="all",
        help="comma-separated variants or 'all'",
    )
    p.add_argument(
        "--concurrency", type=int, default=2,
        help="paper-level concurrency",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("results"),
        help="where to write results.csv + results.tex",
    )
    p.add_argument(
        "--export-only", action="store_true",
        help="skip runs; export current DB state to CSV + LaTeX",
    )
    return p.parse_args()


async def amain(args: argparse.Namespace) -> None:
    languages = [s.strip() for s in args.langs.split(",") if s.strip()]
    if args.variants == "all":
        variants = ALL_VARIANTS
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if not args.export_only:
        if not args.papers or not args.papers.exists():
            raise SystemExit("must pass --papers <file> (one arxiv_id per line)")
        arxiv_ids = [
            line.strip() for line in args.papers.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        logger.info(
            "running matrix: %d papers × %d langs × %d variants = %d jobs",
            len(arxiv_ids), len(languages), len(variants),
            len(arxiv_ids) * len(languages) * len(variants),
        )
        await run_matrix(
            arxiv_ids, languages, variants, concurrency=args.concurrency,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    await export_csv(args.out_dir / "results.csv")
    await export_latex_main_table(args.out_dir / "results.tex", languages)
    await export_latex_appendix_table(args.out_dir / "results_appendix.tex", languages)
    print(f"wrote {args.out_dir / 'results.csv'}")
    print(f"wrote {args.out_dir / 'results.tex'}")
    print(f"wrote {args.out_dir / 'results_appendix.tex'}")


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
