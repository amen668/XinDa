"""Batch JATS evaluation — structure head-to-head (P2) + cost (P1) + quality (level).

Runs each corpus article through the systems (contract / raw_xml / naive,
`evaluation/jats_baselines.py`) across one or more DOMESTIC models, measuring:
- **structure**: JATS-aware PPA/MFR (`metrics.compute(..., profile=JATS_PROFILE)`)
- **cost**:      ¥/paper (`cost.cost_cny`)
- **quality** (`--judge`): RUBRIC-MQM 1-5 (`judge_rubric.judge_one`) on a unit sample —
  the "what level is the translation" number. The judge model is fixed
  (`settings.model_judge_rubric`), independent of the translation model.

The good-model-vs-cheap-model sweep shows the thesis: the contract keeps structure ~100
for BOTH models (structure decoupled from model strength) while raw_xml degrades on the
cheap model; quality, however, still tracks model strength — which is exactly what the
quality-gated review (P3) is for. DB-free. Domestic models only.

Usage:
    python -m xinda.cli.jats_benchmark --models deepseek-v3.2,qwen-turbo --judge --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import statistics
from pathlib import Path

from lxml import etree

from xinda.config import settings
from xinda.evaluation import metrics
from xinda.evaluation.cost import cost_cny
from xinda.evaluation.jats_baselines import SYSTEMS, translate_jats
from xinda.formats import JATS_PROFILE, LTX_PROFILE, extract_units
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

_FIELDS = [
    "pmcid", "model", "system", "ppa", "mfr", "coverage",
    # absolute denominators — a "100%" on 0 formulas is vacuous; report the counts so
    # the reader sees how much structure was actually under test (preserved/original).
    "n_ann", "ann_kept", "n_math", "math_kept",
    "ppa_ordered", "mfr_ordered",  # appendix only (ordered ≈ 100, low signal)
    "judge", "n_units", "fresh_tok", "completion_tok", "cost_cny", "wellformed",
]


# Fields that are numeric (or empty=None) on disk; the rest (pmcid/model/system) stay str.
_STR_FIELDS = {"pmcid", "model", "system"}


def _coerce_row(r: dict) -> dict:
    """Restore a CSV-loaded row (all strings) to the in-memory types `_summary` expects:
    "" → None, numbers → float, wellformed → bool."""
    out: dict = {}
    for k, v in r.items():
        if k in _STR_FIELDS:
            out[k] = v
        elif k == "wellformed":
            out[k] = (v == "True")
        elif v == "" or v is None:
            out[k] = None
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


async def _judge_quality(src_path: str, out_path: str, k: int, profile=JATS_PROFILE) -> float | None:
    """Mean RUBRIC-MQM (1-5) over up to `k` sampled units: source vs translation.
    Aligns src/out units by xpath; judges prose units (skips tiny labels)."""
    from xinda.evaluation.judge_rubric import judge_one

    src_tree, out_tree = etree.parse(src_path), etree.parse(out_path)
    _s, su = extract_units(src_tree, profile)
    _o, ou = extract_units(out_tree, profile)
    tgt_by_xpath = {u["xpath"]: u["src_plain"] for u in ou}
    pairs = [
        (u["src_plain"], tgt_by_xpath[u["xpath"]])
        for u in su
        if u["xpath"] in tgt_by_xpath
        and len(u["src_plain"]) > 30
        and tgt_by_xpath[u["xpath"]]
    ]
    if not pairs:
        return None
    random.Random(0).shuffle(pairs)
    pairs = pairs[:k]
    results = await asyncio.gather(*(
        judge_one(s, t, field="Other", runs=1) for s, t in pairs
    ))
    scores = [r["rubric_score"] for r in results if r.get("valid") and r.get("rubric_score")]
    return round(statistics.mean(scores), 2) if scores else None


async def _eval_one(
    xml_path: Path, lang: str, model: str, systems: list[str],
    tmp: Path, judge: bool, judge_sample: int, profile=JATS_PROFILE,
) -> list[dict]:
    rows: list[dict] = []
    # total translatable units → coverage denominator (full-text vs abstract-level).
    _s, _u = extract_units(etree.parse(str(xml_path)), profile)
    total_units = max(1, sum(1 for u in _u if u["src_plain"]))
    for system in systems:
        try:
            out_tree, toks, n_units = await translate_jats(
                str(xml_path), lang, model, system, profile,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("%s/%s/%s failed: %s", xml_path.name, model, system, e)
            continue
        out_path = tmp / f"{xml_path.stem}__{model}__{system}.xml"
        out_tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        wellformed = True
        try:
            etree.parse(str(out_path), etree.XMLParser(collect_ids=False))
        except etree.XMLSyntaxError:
            wellformed = False
        m = metrics.compute(str(xml_path), str(out_path), profile=profile)
        st = m["stats"]
        n_ann, n_math = st["annotation"]["original"], st["math"]["original"]
        # A score over a 0-element denominator is vacuous (4/5 PMC papers carry no
        # formulas). Report None → excluded from the mean, shown as N/A, never a fake 100.
        ppa_val = round(m["ppa"], 1) if n_ann else None
        mfr_val = round(m["mfr"], 1) if n_math else None
        cost = cost_cny(
            model_name=model, fresh_prompt_tok=toks["fresh"],
            cached_prompt_tok=toks["cached"], completion_tok=toks["completion"],
        )
        judge_score = None
        if judge:
            judge_score = await _judge_quality(str(xml_path), str(out_path), judge_sample, profile)
        row = {
            "pmcid": xml_path.stem, "model": model, "system": system,
            "ppa": ppa_val, "mfr": mfr_val,
            "coverage": round(100.0 * n_units / total_units, 1),
            "n_ann": n_ann, "ann_kept": st["annotation"]["preserved"],
            "n_math": n_math, "math_kept": st["math"]["preserved"],
            "ppa_ordered": round(m["ppa_ordered"], 1) if n_ann else None,
            "mfr_ordered": round(m["mfr_ordered"], 1) if n_math else None,
            "judge": judge_score, "n_units": n_units, "fresh_tok": toks["fresh"],
            "completion_tok": toks["completion"], "cost_cny": round(cost, 4),
            "wellformed": wellformed,
        }
        rows.append(row)
        logger.info(
            "%s/%s/%-8s ppa=%s(%d/%d) mfr=%s(%d/%d) cov=%.0f%% judge=%s cost=¥%.4f",
            xml_path.stem, model, system,
            f"{ppa_val:.1f}" if ppa_val is not None else "N/A", row["ann_kept"], n_ann,
            f"{mfr_val:.1f}" if mfr_val is not None else "N/A", row["math_kept"], n_math,
            row["coverage"], judge_score, cost,
        )
    return rows


def _summary(rows: list[dict], models: list[str], systems: list[str]) -> str:
    # Headline columns only: structure (PPA/MFR), coverage (full-text vs abstract),
    # quality (judge), cost. Ordered PPA/MFR live in the CSV appendix (≈100, low signal).
    lines = ["", f"{'model':16}| {'system':9}| PPA  | MFR  | cov% | judge | ¥/paper | wf%",
             "-" * 72]
    for model in models:
        for s in systems:
            srows = [r for r in rows if r["model"] == model and r["system"] == s]
            if not srows:
                continue
            def mean(k: str) -> float:
                vals = [r[k] for r in srows if r[k] is not None]
                return statistics.mean(vals) if vals else float("nan")
            def cell(k: str) -> str:  # mean or "N/A" when every row had a 0 denominator
                v = mean(k)
                return f"{v:4.1f}" if v == v else " N/A"  # nan check
            wf = 100.0 * sum(1 for r in srows if r["wellformed"]) / len(srows)
            j = mean("judge")
            jstr = f"{j:5.2f}" if j == j else "  -  "  # nan check
            lines.append(
                f"{model:16}| {s:9}| {cell('ppa')} | {cell('mfr')} | "
                f"{mean('coverage'):4.0f} | {jstr} | {mean('cost_cny'):7.4f} | {wf:3.0f}"
            )
    # Footnote: how much structure was actually under test (drives whether MFR is meaningful).
    n_combos = max(1, len({(r["pmcid"], r["model"]) for r in rows}))
    tot_ann = sum(r["n_ann"] for r in rows) // n_combos
    n_with_math = len({r["pmcid"] for r in rows if r["n_math"]})
    n_papers = len({r["pmcid"] for r in rows})
    lines.append(
        f"\n[structure under test] ~{tot_ann} annotations/paper, "
        f"{n_with_math}/{n_papers} papers carry formulas "
        f"(MFR is N/A on the formula-free ones — pick formula-dense papers to stress it)."
    )
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> None:
    profile = LTX_PROFILE if args.format == "ltx" else JATS_PROFILE
    corpus = Path(args.corpus_dir)
    xmls = sorted(corpus.glob("*.xml"))
    if args.limit:
        xmls = xmls[: args.limit]
    if not xmls:
        raise SystemExit(f"no .xml under {corpus}")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    bad = [s for s in systems if s not in SYSTEMS]
    if bad:
        raise SystemExit(f"unknown systems {bad}; choose from {SYSTEMS}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(xmls)} papers × {len(models)} models × {len(systems)} systems "
        f"→ {args.lang}  (format={args.format} judge={'on' if args.judge else 'off'})"
    )

    # Translated XML is PERSISTED (not a temp dir): rendering/spot-check need the actual
    # documents, and re-translating to recover them is the expensive thing to avoid.
    xml_dir = out_dir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "jats_structure.csv"

    # Resume + incremental flush: a long sweep that dies (Arrearage, 429-storm, OOM) must
    # NOT lose hours of finished papers. We load any prior rows, skip (paper,model) pairs
    # already fully done, and append each paper's rows to disk THE MOMENT it finishes —
    # so the CSV is always current, never an all-or-nothing write at the very end.
    all_rows: list[dict] = []
    done_systems: dict[tuple[str, str], set[str]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                all_rows.append(_coerce_row(r))  # CSV gives strings → restore floats/None/bool
                done_systems.setdefault((r["pmcid"], r["model"]), set()).add(r["system"])
        print(f"resume: {len(all_rows)} rows already on disk")

    fh_csv = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh_csv, fieldnames=_FIELDS)
    if fh_csv.tell() == 0:
        writer.writeheader()
        fh_csv.flush()
    write_lock = asyncio.Lock()

    # Papers are independent → process up to `--concurrency` at once. Each paper still
    # owns its RateLimiter, so keep this LOW for the rate-limited coding endpoint (429s);
    # the regular qwen-plus endpoint (RPM 600) has headroom for 3-4.
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def one(xml: Path, model: str) -> None:
        if done_systems.get((xml.stem, model), set()) >= set(systems):
            return  # every requested system already on disk → skip (resume)
        async with sem:
            try:
                rows = await _eval_one(
                    xml, args.lang, model, systems, xml_dir, args.judge, args.judge_sample,
                    profile,
                )
            except Exception as e:  # noqa: BLE001 — one bad paper must not abort the run
                logger.warning("paper %s / %s failed, skipped: %s", xml.name, model, e)
                return
        async with write_lock:  # flush this paper's rows immediately
            all_rows.extend(rows)
            writer.writerows(rows)
            fh_csv.flush()

    try:
        await asyncio.gather(*(one(xml, model) for xml in xmls for model in models))
    finally:
        fh_csv.close()

    print(_summary(all_rows, models, systems))
    print(f"\nwrote {csv_path}  ({len(all_rows)} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch JATS structure/cost/quality head-to-head")
    p.add_argument("--corpus-dir", default="corpus/jats", help="dir of JATS XML files")
    p.add_argument("--lang", default="en", help="target language (Chinese journals → en)")
    p.add_argument("--models", default=settings.model_first_pass,
                   help="comma-separated DOMESTIC model aliases (good vs cheap sweep)")
    p.add_argument("--systems", default="contract,raw_xml,naive,abstract",
                   help="comma-separated subset of contract,raw_xml,naive,abstract")
    p.add_argument("--judge", action="store_true",
                   help="score translation quality (RUBRIC-MQM 1-5) on a unit sample")
    p.add_argument("--judge-sample", type=int, default=8,
                   help="units sampled per (paper,model,system) for the judge")
    p.add_argument("--out-dir", default="results/jats", help="output dir")
    p.add_argument("--format", choices=["jats", "ltx"], default="jats",
                   help="XML dialect: jats (journal, zh→en) or ltx (LaTeXML/arXiv, en→zh)")
    p.add_argument("--limit", type=int, default=0, help="cap number of papers (0 = all)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="papers processed in parallel; keep 1 for the coding endpoint (429s), "
                        "3-4 OK for regular qwen-plus (RPM 600)")
    return p.parse_args()


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
