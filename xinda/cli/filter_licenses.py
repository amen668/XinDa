"""License filter for the benchmark corpus — keeps only papers whose license
permits creating AND redistributing a translation (a derivative work).

Translation is a derivative work, so the safe-to-publish set is **CC0 / CC-BY /
CC-BY-SA** only. ND forbids translation; NC is research-only; the arXiv default
``nonexclusive-distrib`` (and a missing license) grants no third-party derivative
right. Run this on a candidate id list BEFORE feeding `--papers` to the benchmark,
so场景 A (translating others' papers) is license-clean at the source.

    python -m xinda.cli.filter_licenses candidates.txt \
        --out paper_ids.txt [--report licenses.csv] [--include-nc]

`--include-nc` also keeps CC-BY-NC* (derivatives allowed, non-commercial) for
research-only use — keep it OFF for the publishable corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from xinda.logger_config import setup_logger
from xinda.providers.arxiv_meta import (
    classify_license,
    _fetch_license_sync,
)

logger = setup_logger(__name__)


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


async def amain(args: argparse.Namespace) -> None:
    ids = _read_ids(args.candidates)
    sem = asyncio.Semaphore(args.concurrency)

    async def classify(aid: str) -> dict:
        async with sem:
            lic = await asyncio.to_thread(_fetch_license_sync, aid)
            c = classify_license(lic)
            c["arxiv_id"] = aid
            return c

    results = await asyncio.gather(*(classify(a) for a in ids))

    kept: list[str] = []
    for r in results:
        ok = r["permissive"] or (args.include_nc and r["derivatives_ok"])
        if ok:
            kept.append(r["arxiv_id"])

    args.out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    if args.report:
        with args.report.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arxiv_id", "license_label", "license_url",
                        "derivatives_ok", "permissive", "kept"])
            for r in sorted(results, key=lambda x: x["label"]):
                ok = r["permissive"] or (args.include_nc and r["derivatives_ok"])
                w.writerow([r["arxiv_id"], r["label"], r["license"] or "",
                            r["derivatives_ok"], r["permissive"], ok])

    # console summary
    from collections import Counter
    tally = Counter(r["label"] for r in results)
    print("=" * 60)
    print(f"license filter: {len(ids)} candidates → {len(kept)} kept "
          f"({'CC0/BY/BY-SA' + ('+NC' if args.include_nc else '')})")
    print("-" * 60)
    for label, n in tally.most_common():
        print(f"  {label:24}{n:>5}")
    print("=" * 60)
    print(f"wrote allowlist → {args.out}")
    if args.report:
        print(f"wrote per-paper report → {args.report}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path, help="file with one arxiv_id per line")
    ap.add_argument("--out", type=Path, default=Path("paper_ids.txt"),
                    help="where to write the license-clean allowlist")
    ap.add_argument("--report", type=Path, default=None,
                    help="optional per-paper license CSV")
    ap.add_argument("--include-nc", action="store_true",
                    help="also keep CC-BY-NC* (research-only, NOT for publication)")
    ap.add_argument("--concurrency", type=int, default=4)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
