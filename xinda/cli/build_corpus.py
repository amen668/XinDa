"""Build a discipline-stratified, license-clean benchmark corpus from arXiv.

Harvests via **OAI-PMH ListRecords** (the only arXiv endpoint that carries the
license, inline, ~1000 records/page). For each discipline (an arXiv archive
`set`: cs, math, physics, q-bio, stat, eess, econ) it pages a recent date
window, keeps only papers whose license permits redistributing a translation
(**CC0 / CC-BY / CC-BY-SA**; see `arxiv_meta.classify_license`), and takes
`--per-field` survivors. ~40% of recent papers in active archives are CC-BY, so
a few pages per field suffice.

Output: a flat `paper_ids.txt` (feed to `cli/benchmark --papers`) + a manifest
CSV (field / primary_category / license / title) for the corpus-composition
table. Stratifying by discipline matters so the structure-fidelity claims
(PPA/MFR, structure-break) generalize across math-heavy and prose-heavy fields.

    python -m xinda.cli.build_corpus \
        --per-field 5 --from 2025-01-01 \
        --out paper_ids.txt --manifest corpus.csv

    # custom disciplines (arXiv archive set codes) + research-only NC papers:
    python -m xinda.cli.build_corpus \
        --fields cs,math,stat,eess,q-bio,econ --include-nc

License is authoritative (OAI). `--include-nc` also keeps CC-BY-NC* (derivatives
allowed, non-commercial → research-only, NOT for publication).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path

from xinda.logger_config import setup_logger
from xinda.providers.arxiv_meta import (
    citations_batch,
    citations_openalex,
    classify_license,
    list_set_page,
)

logger = setup_logger(__name__)

# arXiv archive `set` codes = disciplines. Mixes math-heavy (math/stat/eess) with
# prose-heavy (cs/econ) so the structure-fidelity metrics have contrast.
DEFAULT_FIELDS = ["cs", "math", "stat", "eess", "q-bio", "econ", "physics"]

_FIELD_LABEL = {
    "cs": "Computer Science", "math": "Mathematics", "physics": "Physics",
    "q-bio": "Quantitative Biology", "stat": "Statistics",
    "eess": "Electrical Engineering", "econ": "Economics",
    "q-fin": "Quantitative Finance",
}


def _harvest_field(field: str, gather_n: int, from_date: str, until_date: str | None,
                   include_nc: bool, published_only: bool, max_pages: int, delay: float,
                   seen: set[str]) -> list[dict]:
    """Page OAI ListRecords for `field`, collect up to `gather_n` license-clean
    candidates (a pool that's later ranked by citations, then trimmed).

    `seen` is shared across fields so a cross-listed paper (e.g. primary econ.EM
    appearing in both the stat and econ sets) is counted once, under whichever
    discipline harvests it first.
    """
    kept: list[dict] = []
    token: str | None = None
    scanned = 0
    for page in range(max_pages):
        try:
            records, token = list_set_page(
                set_spec=field if page == 0 else None,
                from_date=from_date if page == 0 else None,
                until_date=until_date if page == 0 else None,
                token=token,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: OAI page %d failed: %s", field, page, e)
            break
        for r in records:
            scanned += 1
            aid = r["arxiv_id"]
            if not aid or aid in seen:
                continue
            c = classify_license(r["license"])
            ok = c["permissive"] or (include_nc and c["derivatives_ok"])
            # quality gate: a DOI or journal-ref ⇒ formally published / peer-reviewed
            published = bool(r.get("doi") or r.get("journal_ref"))
            if published_only and not published:
                continue
            if ok:
                seen.add(aid)
                kept.append({
                    "arxiv_id": r["arxiv_id"], "field": field,
                    "category": r["primary_category"] or field,
                    "license_label": c["label"], "license": r["license"] or "",
                    "title": r["title"], "authors": r.get("authors", ""),
                    "doi": r.get("doi") or "", "journal_ref": r.get("journal_ref") or "",
                    "published": published,
                })
                if len(kept) >= gather_n:
                    break
        if len(kept) >= gather_n or not token:
            break
        time.sleep(delay)  # politeness between OAI pages
    logger.info("%s: scanned %d → pooled %d candidates", field, scanned, len(kept))
    return kept


async def amain(args: argparse.Namespace) -> None:
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    seen: set[str] = set()          # global dedup across cross-listed sets
    # gather a larger candidate pool per field so citation-ranking has choices
    gather_n = max(args.per_field, args.pool) if args.rank_by_citations else args.per_field
    pools: dict[str, list[dict]] = {}
    for field in fields:
        pools[field] = await asyncio.to_thread(
            _harvest_field, field, gather_n, getattr(args, "from"),
            args.until, args.include_nc, args.published_only,
            args.max_pages, args.delay, seen,
        )

    # rank each field's pool by citation count (high = vetted/impactful), keep top
    if args.rank_by_citations:
        all_ids = [r["arxiv_id"] for rows in pools.values() for r in rows]
        logger.info("fetching citations for %d candidates …", len(all_ids))
        cites = await asyncio.to_thread(citations_batch, all_ids)
        for rows in pools.values():
            for r in rows:
                meta = cites.get(r["arxiv_id"], {})
                r["citations"] = meta.get("citations")
                r["year"] = meta.get("year")
                r["venue"] = meta.get("venue", "")
        # OpenAlex fallback (by DOI) for candidates S2 didn't cover (e.g. SIGMA math)
        missing = {r["arxiv_id"]: r["doi"]
                   for rows in pools.values() for r in rows
                   if r.get("citations") is None and r.get("doi")}
        if missing:
            logger.info("OpenAlex fallback for %d uncited candidates …", len(missing))
            oa = await asyncio.to_thread(citations_openalex, missing)
            for rows in pools.values():
                for r in rows:
                    if r.get("citations") is None and r["arxiv_id"] in oa:
                        r["citations"] = oa[r["arxiv_id"]]
                        r["venue"] = r.get("venue") or r.get("journal_ref", "")

    all_rows: list[dict] = []
    for field in fields:
        rows = pools[field]
        if args.rank_by_citations:
            rows = sorted(rows, key=lambda r: (r.get("citations") or -1), reverse=True)
        all_rows.extend(rows[:args.per_field])

    ids = [r["arxiv_id"] for r in all_rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    if args.manifest:
        with args.manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arxiv_id", "field", "field_label", "category",
                        "license_label", "license", "citations", "year", "venue",
                        "published", "doi", "journal_ref", "authors", "title"])
            for r in all_rows:
                w.writerow([r["arxiv_id"], r["field"],
                            _FIELD_LABEL.get(r["field"], r["field"]), r["category"],
                            r["license_label"], r["license"],
                            r.get("citations", ""), r.get("year", ""), r.get("venue", ""),
                            r.get("published", ""), r.get("doi", ""),
                            r.get("journal_ref", ""), r.get("authors", ""), r["title"]])

    print("=" * 66)
    print(f"corpus: {len(ids)} papers across {len(fields)} disciplines "
          f"(CC0/BY/BY-SA{'+NC' if args.include_nc else ''}, from {getattr(args, 'from')})")
    print("-" * 66)
    by_field: dict[str, int] = {}
    for r in all_rows:
        by_field[r["field"]] = by_field.get(r["field"], 0) + 1
    cited = {r["arxiv_id"]: r.get("citations") for r in all_rows}
    for field in fields:
        fc = [c for r in all_rows if r["field"] == field
              and (c := cited.get(r["arxiv_id"])) is not None]
        cit_note = f"  (citations {min(fc)}–{max(fc)})" if fc else ""
        print(f"  {_FIELD_LABEL.get(field, field):24}{by_field.get(field, 0):>3} kept{cit_note}")
    print("=" * 66)
    print(f"wrote ids → {args.out}")
    if args.manifest:
        print(f"wrote manifest → {args.manifest}")
    short = [f for f in fields if by_field.get(f, 0) < args.per_field]
    if short:
        print(f"NOTE: under target ({args.per_field}) for: {', '.join(short)} "
              f"— raise --max-pages or widen --from.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="comma-separated arXiv archive set codes (disciplines)")
    ap.add_argument("--per-field", type=int, default=5,
                    help="target license-clean papers per discipline")
    ap.add_argument("--from", default="2025-01-01",
                    help="OAI 'from' datestamp (YYYY-MM-DD); recent = more CC")
    ap.add_argument("--until", default=None, help="OAI 'until' datestamp")
    ap.add_argument("--max-pages", type=int, default=4,
                    help="max OAI pages per discipline (~1000 records each)")
    ap.add_argument("--include-nc", action="store_true",
                    help="also keep CC-BY-NC* (research-only, NOT for publication)")
    ap.add_argument("--published-only", action="store_true",
                    help="keep only papers with a DOI/journal-ref (peer-reviewed = higher quality)")
    ap.add_argument("--rank-by-citations", action="store_true",
                    help="pool candidates per field, keep the most-cited (Semantic Scholar)")
    ap.add_argument("--pool", type=int, default=40,
                    help="candidates to pool per field before citation-ranking")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between OAI pages (politeness)")
    ap.add_argument("--out", type=Path, default=Path("paper_ids.txt"))
    ap.add_argument("--manifest", type=Path, default=Path("corpus.csv"))
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
