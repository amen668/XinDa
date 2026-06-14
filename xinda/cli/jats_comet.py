"""Reference-free QE (COMET-Kiwi) for the DB-free JATS leg — a credible translation-
*quality* number to sit beside the structure (PPA/MFR) and cost tables.

`neural_qe` scores DB `translations` rows by job_id; the JATS head-to-head is file-based,
so this gives the JATS contract translations their own CometKiwi score without the DB.

Two phases, because translation needs the DashScope API (host) while scoring needs the
GPU `qe` container (torch + unbabel-comet). They hand off via a JSON under `workspace/`
(the only dir mounted into the qe container):

    # 1. translate + align src/mt unit pairs on the HOST (has DASHSCOPE_API_KEY):
    python -m xinda.cli.jats_comet --corpus-dir corpus/jats_formula \\
        --lang en --pairs workspace/jats_comet_pairs.json

    # 2. score the pairs in the GPU container (no db needed → --no-deps):
    docker compose run --rm --no-deps qe \\
        python -m xinda.cli.jats_comet --score \\
        --pairs workspace/jats_comet_pairs.json --out results/jats_comet.csv

CometKiwi is reference-free (src, mt) — `src` is the source-language prose, `mt` the
translation, both placeholder-stripped (we score prose quality, not structure).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from lxml import etree

from xinda.formats import JATS_PROFILE, extract_units
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)


def _align_pairs(src_tree, out_tree) -> list[tuple[str, str]]:
    """(src_plain, mt_plain) prose pairs aligned by xpath between source and translation."""
    _ss, su = extract_units(src_tree, JATS_PROFILE)
    _os, ou = extract_units(out_tree, JATS_PROFILE)
    mt_by_xpath = {u["xpath"]: u["src_plain"] for u in ou}
    pairs: list[tuple[str, str]] = []
    for u in su:
        mt = mt_by_xpath.get(u["xpath"], "")
        if u["src_plain"] and mt:
            pairs.append((u["src_plain"], mt))
    return pairs


async def _build_pairs(corpus: Path, lang: str, model: str, limit: int) -> list[dict]:
    # imported lazily: pulls in the openai-backed provider, absent from the lean qe image
    from xinda.evaluation.jats_baselines import translate_jats  # noqa: PLC0415

    xmls = sorted(corpus.glob("*.xml"))
    if limit:
        xmls = xmls[:limit]
    if not xmls:
        raise SystemExit(f"no .xml under {corpus}")
    records: list[dict] = []
    for xml in xmls:
        out_tree, _toks, _n = await translate_jats(str(xml), lang, model, "contract")
        src_tree = etree.parse(str(xml))
        pairs = _align_pairs(src_tree, out_tree)
        for src, mt in pairs:
            records.append({"pmcid": xml.stem, "src": src, "mt": mt})
        logger.info("%s: %d prose pairs", xml.stem, len(pairs))
    return records


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _score(pairs_path: Path, out_csv: Path) -> None:
    import csv

    from xinda.evaluation.comet import score_pairs  # noqa: PLC0415

    records = json.loads(pairs_path.read_text(encoding="utf-8"))
    scores = score_pairs([(r["src"], r["mt"]) for r in records], gpus=1)
    for r, s in zip(records, scores, strict=True):
        r["cometkiwi"] = s

    by_paper: dict[str, list[float]] = {}
    for r in records:
        by_paper.setdefault(r["pmcid"], []).append(r["cometkiwi"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pmcid", "n_units", "median", "p10", "mean"])
        for pmcid, vals in by_paper.items():
            w.writerow([
                pmcid, len(vals), round(statistics.median(vals), 4),
                round(_percentile(vals, 10), 4), round(statistics.mean(vals), 4),
            ])
        allv = [r["cometkiwi"] for r in records]
        w.writerow([
            "ALL", len(allv), round(statistics.median(allv), 4),
            round(_percentile(allv, 10), 4), round(statistics.mean(allv), 4),
        ])
    print(f"\nCometKiwi (reference-free) over {len(records)} prose units:")
    print(f"  median={statistics.median(allv):.4f}  p10={_percentile(allv, 10):.4f}  "
          f"mean={statistics.mean(allv):.4f}")
    print(f"wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CometKiwi QE for JATS contract translations")
    p.add_argument("--corpus-dir", default="corpus/jats_formula")
    p.add_argument("--lang", default="en")
    p.add_argument("--model", default="qwen-plus", help="translation model (host phase)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pairs", default="workspace/jats_comet_pairs.json",
                   help="JSON handoff file (written by host phase, read by --score)")
    p.add_argument("--score", action="store_true",
                   help="GPU phase: score the pairs file with CometKiwi")
    p.add_argument("--out", default="results/jats_comet.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.score:
        _score(Path(args.pairs), Path(args.out))
        return
    records = asyncio.run(
        _build_pairs(Path(args.corpus_dir), args.lang, args.model, args.limit)
    )
    pairs_path = Path(args.pairs)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} pairs → {pairs_path}  (now run --score in the qe container)")


if __name__ == "__main__":
    main()
