"""Meta-evaluation CLI: produce judge-trust report.

Usage:
    # checks 1-3 from local DB
    python -m xinda.cli.meta_eval

    # check 4: provide WMT24 evaluation samples (CSV with src,mt,human_score)
    python -m xinda.cli.meta_eval --wmt24 wmt24_samples.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from xinda.db.engine import async_session
from xinda.evaluation import judge_rubric, meta_eval


async def amain(wmt24: Path | None) -> None:
    async with async_session() as session:
        report = await meta_eval.full_report(session)

    if wmt24 and wmt24.exists():
        # Read CSV with cols: src, mt, human_score (float)
        with wmt24.open() as f:
            rows = list(csv.DictReader(f))
        judge_scores: list[float] = []
        human_scores: list[float] = []
        for row in rows:
            try:
                hs = float(row["human_score"])
            except (KeyError, ValueError):
                continue
            # judge each pair via rubric_mqm
            res = await judge_rubric.judge_one(row["src"], row["mt"], runs=1)
            js = res.get("rubric_score")
            if js is not None:
                judge_scores.append(float(js))
                human_scores.append(hs)
        report["check_4_wmt24_calibration"] = meta_eval.wmt24_calibration(
            judge_scores, human_scores,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wmt24", type=Path,
        help="WMT24 sample CSV (src,mt,human_score columns) for check 4",
    )
    args = ap.parse_args()
    asyncio.run(amain(args.wmt24))


if __name__ == "__main__":
    main()
