"""Build a license-clean Chinese-JATS evaluation corpus from PMC Open Access.

PMC is the reliable open source of real Chinese-journal JATS XML (`chinese[lang] AND
open access[filter]`). BUT translation is a *derivative work*, and the most common
Chinese-journal OA license is **CC BY-NC-ND** — the **ND clause forbids derivatives**,
so those articles are unusable for a translation corpus. This builder keeps ONLY
derivative-permitting licenses (CC-BY / CC-BY-SA / CC0 / public-domain), and further
requires real Chinese body text plus structure (cross-refs, and ideally formulas/tables)
so the corpus actually exercises structure preservation.

Output: `<out-dir>/<pmcid>.xml` per kept article + a `manifest.csv` (pmcid, journal,
title, license, cjk_chars, n_xref, n_formula, n_table, n_p).

Usage:
    python -m xinda.cli.jats_corpus --n 5 --out-dir corpus/jats
    python -m xinda.cli.jats_corpus --n 50 --pool 400 --min-cjk 1500
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from lxml import etree

from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DERIVATIVE_OK = {"by", "by-sa"}  # CC codes that permit derivatives; +zero/publicdomain


def _local(elem: etree._Element) -> str | None:
    return elem.tag.split("}")[-1] if isinstance(elem.tag, str) else None


def _get(url: str, *, timeout: int = 60, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ft-jats-corpus/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries}: {url}: {last}")


def _search_ids(lang: str, pool: int) -> list[str]:
    term = urllib.parse.quote(f"{lang}[lang] AND open access[filter]")
    url = f"{_EUTILS}/esearch.fcgi?db=pmc&term={term}&retmax={pool}&retmode=json"
    import json
    data = json.loads(_get(url).decode())
    return data.get("esearchresult", {}).get("idlist", [])


def _license_code(article: etree._Element) -> tuple[str, str]:
    """Return (cc_code, license_url). cc_code ∈ {by, by-nc-nd, zero, …, none}."""
    blob_parts: list[str] = []
    for lic in article.iter():
        if _local(lic) in ("license", "license-p", "ali:license_ref"):
            href = lic.get("{http://www.w3.org/1999/xlink}href") or ""
            blob_parts.append(href + " " + "".join(lic.itertext()))
    blob = " ".join(blob_parts).lower()
    if "publicdomain" in blob or "/zero/" in blob:
        return "zero", "creativecommons.org/publicdomain/zero"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/", blob)
    if m:
        return m.group(1), m.group(0)
    return "none", ""


def _cjk_chars(article: etree._Element) -> int:
    n = 0
    in_body = False
    for e in article.iter():
        lc = _local(e)
        if lc == "body":
            in_body = True
        if not in_body or lc != "p":
            continue
        for ch in (e.text or ""):
            if "㐀" <= ch <= "鿿":
                n += 1
    return n


def _count(article: etree._Element, locals_: set[str]) -> dict[str, int]:
    c = dict.fromkeys(locals_, 0)
    for e in article.iter():
        lc = _local(e)
        if lc in c:
            c[lc] += 1
    return c


def _text(article: etree._Element, tag: str) -> str:
    for e in article.iter():
        if _local(e) == tag:
            return " ".join("".join(e.itertext()).split())
    return ""


def _select(
    candidates: list[dict], *, n: int, formula_quota: int, per_journal: int
) -> list[dict]:
    """Pick `n` candidates with stratification: first fill up to `formula_quota` with
    formula-bearing articles (PMC Chinese OA is clinical-heavy → formulas are scarce, so
    they must be reserved), then fill the rest; cap `per_journal` throughout."""
    out: list[dict] = []
    seen_pmcid: set[str] = set()
    per_j: dict[str, int] = {}

    def take(c: dict) -> None:
        out.append(c)
        seen_pmcid.add(c["pmcid"])
        per_j[c["journal"]] = per_j.get(c["journal"], 0) + 1

    def eligible(c: dict) -> bool:
        return (
            c["pmcid"] not in seen_pmcid
            and per_j.get(c["journal"], 0) < per_journal
        )

    # pass 1: formula-bearing, richest first, up to the quota
    for c in sorted(candidates, key=lambda c: -c["n_formula"]):
        if sum(1 for x in out if x["n_formula"] >= 1) >= formula_quota:
            break
        if c["n_formula"] >= 1 and eligible(c):
            take(c)
    # pass 2: fill remaining slots with anything left
    for c in candidates:
        if len(out) >= n:
            break
        if eligible(c):
            take(c)
    return out[:n]


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _search_ids(args.lang, args.pool)
    logger.info("PMC %s[lang] OA pool: %d candidate ids", args.lang, len(ids))

    # Collect ALL qualifying candidates first (don't stop early), then select with
    # formula stratification — a single greedy pass can't reserve slots for the rare
    # formula-bearing articles.
    candidates: list[dict] = []
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        raw = _get(f"{_EUTILS}/efetch.fcgi?db=pmc&id={','.join(chunk)}&rettype=xml")
        try:
            root = etree.fromstring(raw)
        except etree.XMLSyntaxError:
            continue
        for art in (e for e in root.iter() if _local(e) == "article"):
            code, url = _license_code(art)
            if code not in _DERIVATIVE_OK and code != "zero":
                continue
            cjk = _cjk_chars(art)
            if cjk < args.min_cjk:
                continue
            cnt = _count(art, {"xref", "inline-formula", "disp-formula", "table", "p"})
            n_formula = cnt["inline-formula"] + cnt["disp-formula"]
            if cnt["xref"] < args.min_xref or n_formula < args.min_formula:
                continue
            pmcid = ""
            for aid in art.iter():
                # PMC efetch tags the id as pub-id-type="pmcid" (value already "PMC…");
                # some records use bare "pmc" with a numeric value.
                if _local(aid) == "article-id" and aid.get("pub-id-type") in ("pmcid", "pmc"):
                    v = (aid.text or "").strip()
                    pmcid = v if v.upper().startswith("PMC") else f"PMC{v}"
                    break
            if not pmcid:
                continue
            candidates.append({
                "pmcid": pmcid, "journal": _text(art, "journal-title") or "?",
                "title": _text(art, "article-title")[:120],
                "license": f"CC-{code.upper()}" if code != "zero" else "CC0",
                "license_url": url,
                "cjk_chars": cjk, "n_xref": cnt["xref"],
                "n_formula": n_formula, "n_table": cnt["table"], "n_p": cnt["p"],
                "_xml": etree.tostring(art, encoding="utf-8", xml_declaration=True),
            })

    logger.info(
        "%d qualifying candidates (%d with formulas); selecting %d (formula quota %d)",
        len(candidates), sum(1 for c in candidates if c["n_formula"] >= 1),
        args.n, args.formula_quota,
    )
    kept = _select(
        candidates, n=args.n, formula_quota=args.formula_quota,
        per_journal=args.per_journal,
    )
    for c in kept:
        (out_dir / f"{c['pmcid']}.xml").write_bytes(c.pop("_xml"))

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "pmcid", "journal", "title", "license", "license_url",
            "cjk_chars", "n_xref", "n_formula", "n_table", "n_p",
        ])
        w.writeheader()
        w.writerows(kept)
    print(f"\nkept {len(kept)} articles → {out_dir}")
    print(f"manifest: {manifest}")
    for k in kept:
        print(
            f"  {k['pmcid']} [{k['license']}] cjk={k['cjk_chars']} "
            f"xref={k['n_xref']} formula={k['n_formula']} table={k['n_table']} "
            f"— {k['journal']}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a license-clean Chinese-JATS corpus from PMC OA")
    p.add_argument("--lang", default="chinese", help="PMC [lang] filter (default: chinese)")
    p.add_argument("--n", type=int, default=5, help="number of articles to keep")
    p.add_argument("--pool", type=int, default=120, help="candidate ids to scan")
    p.add_argument("--min-cjk", type=int, default=800, help="min CJK chars in body")
    p.add_argument("--min-xref", type=int, default=10, help="min cross-references")
    p.add_argument("--min-formula", type=int, default=0,
                   help="hard per-article minimum inline+disp formulas (0 = no filter)")
    p.add_argument("--formula-quota", type=int, default=0,
                   help="reserve this many of --n slots for formula-bearing articles "
                        "(stratification — PMC Chinese OA is formula-scarce)")
    p.add_argument("--per-journal", type=int, default=2, help="cap articles per journal")
    p.add_argument("--out-dir", default="corpus/jats", help="output directory")
    return p.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
