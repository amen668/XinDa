"""JATS translate CLI — structure-preserving JATS→JATS full-text translation.

This is the framework's *second* XML frontend. The arXiv pipeline ingests LaTeXML
XML; journals (PMC / CrossRef / CNKI / most production systems) hold their full text
as **JATS** (Journal Article Tag Suite). The same placeholder contract
(`translation/placeholders.py`) and translation engine (`providers` + `prompts` +
`batching`) drive both — only the `FormatProfile` differs. So a journal's existing
JATS article can be translated into any target language with its cross-references,
equations, tables, and figures preserved byte-for-byte.

It is intentionally DB-free: it operates directly on a JATS file so the publishing
demonstration needs no schema. (The full 13-stage pipeline — glossary, fact
verification, refine, eval — remains the LaTeXML/arXiv path; this CLI proves the
frontend generalizes and produces valid, structure-preserving JATS output.)

Usage:
    python -m xinda.cli.jats_translate article.xml zh
    python -m xinda.cli.jats_translate article.xml fr --out fr.xml --max-units 40
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path
from typing import Any

from lxml import etree

from xinda.config import settings
from xinda.evaluation.cost import cost_cny
from xinda.formats import JATS_PROFILE, apply_units, extract_units
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.batching import batch_all
from xinda.translation.prompts import (
    TRANSLATION_BATCH_SCHEMA,
    language_name,
    stable_prefix,
    variable_suffix,
)
from xinda.translation.rate_limit import RateLimiter
from xinda.util import parse_translation_array

logger = setup_logger(__name__)

_PRESERVE_LOCALS = (
    "xref", "inline-formula", "disp-formula", "ext-link", "inline-graphic",
    "graphic", "table", "table-wrap", "fig", "tr", "td", "sec", "list",
)


def _local(elem: etree._Element) -> str | None:
    return elem.tag.split("}")[-1] if isinstance(elem.tag, str) else None


def _article_title(tree: etree._ElementTree) -> str:
    root = tree.getroot()
    for el in root.iter():  # JATS
        if _local(el) == "article-title":
            return " ".join("".join(el.itertext()).split())
    for el in root.iter():  # LaTeXML / fallback: the document's first <title>
        if _local(el) == "title":
            t = " ".join("".join(el.itertext()).split())
            if t:
                return t
    return "Untitled"


def _count_tags(tree: etree._ElementTree, locals_: tuple[str, ...]) -> dict[str, int]:
    counts = dict.fromkeys(locals_, 0)
    for el in tree.getroot().iter():
        lc = _local(el)
        if lc in counts:
            counts[lc] += 1
    return counts


def _parse_response(text: str) -> list | None:
    return parse_translation_array(text)


async def _translate_units(
    units: list[dict[str, Any]], lang: str, title: str, model: str,
    extra_system: str = "",
) -> tuple[dict[int, str], dict[str, int]]:
    """Translate units' src_text via the real provider. Returns (id→tgt, token totals).

    `extra_system` appends a line to the cached system prompt — used by the baseline
    systems (`evaluation/jats_baselines.py`) to swap the structure-handling instruction
    (e.g. "keep XML tags verbatim") while sharing the exact same translation engine, so
    contract vs raw-XML vs naive differ ONLY in structure strategy, not in model/prompt scaffold.
    """
    provider = create_provider(model)
    limiter = RateLimiter(provider.rpm, provider.tpm)

    section_outline = [
        u["src_text"] for u in units if u["kind"] == "section_heading"
    ][:40]
    prefix = stable_prefix(
        paper_title=title, arxiv_id="jats", field="Other",
        target_language=lang, section_outline=section_outline or None,
    )
    if extra_system:
        prefix = prefix + "\n" + extra_system

    items = [
        {"id": u["_id"], "src_text": u["src_text"], "char_count": u["char_count"]}
        for u in units
    ]
    batches = batch_all(items, token_fn=provider.estimate_tokens)
    logger.info(
        "JATS translate: %d units → %d batches (model=%s lang=%s)",
        len(items), len(batches), provider.model_name, lang,
    )

    out: dict[int, str] = {}
    tokens = {"cached": 0, "fresh": 0, "completion": 0}
    prev_tgt: str | None = None

    for bi, batch in enumerate(batches):
        next_source = None
        if bi + 1 < len(batches):
            next_source = " ".join(it["src_text"] for it in batches[bi + 1])[:800]
        suffix = variable_suffix(
            batch, prev_translation=prev_tgt, next_source=next_source,
        )
        est = provider.estimate_tokens(prefix) + provider.estimate_tokens(suffix)
        await limiter.reserve(est)
        t0 = time.monotonic()
        try:
            tr = await provider.generate(
                prompt=suffix, system=prefix, json_schema=TRANSLATION_BATCH_SCHEMA,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("batch %d failed (%s) — source fallback", bi, e)
            continue
        tokens["cached"] += tr.cached_prompt_tokens
        tokens["fresh"] += tr.fresh_prompt_tokens
        tokens["completion"] += tr.completion_tokens

        parsed = _parse_response(tr.text)
        if parsed is None:
            logger.warning("batch %d JSON parse failed — source fallback", bi)
            continue
        by_id = {it["id"] for it in batch}
        last = None
        for arr in parsed:
            if not isinstance(arr, (list, tuple)) or len(arr) < 2:
                continue
            _id, tgt = arr[0], arr[1]
            if isinstance(_id, str) and _id.isdigit():
                _id = int(_id)
            if _id in by_id:
                out[_id] = tgt
                last = tgt
        prev_tgt = last or prev_tgt
        logger.info(
            "batch %d/%d done (%d items, %.1fs)",
            bi + 1, len(batches), len(batch), time.monotonic() - t0,
        )
    return out, tokens


def _report(src_tree: etree._ElementTree, out_tree: etree._ElementTree) -> bool:
    src = _count_tags(src_tree, _PRESERVE_LOCALS)
    tgt = _count_tags(out_tree, _PRESERVE_LOCALS)
    print("\nstructure preservation (source → translated):")
    all_ok = True
    for lc in _PRESERVE_LOCALS:
        s, g = src[lc], tgt[lc]
        if s == 0 and g == 0:
            continue
        ok = s == g
        all_ok &= ok
        print(f"  {lc:16} {s:5} → {g:5}   {'OK' if ok else 'LOSS'}")
    print(f"\n  ALL STRUCTURE PRESERVED: {all_ok}")
    return all_ok


async def amain(args: argparse.Namespace) -> None:
    src_path = Path(args.input)
    if not src_path.exists():
        raise SystemExit(f"input not found: {src_path}")
    out_path = Path(args.out) if args.out else src_path.with_name(
        f"{src_path.stem}_{args.lang}.xml"
    )

    tree = etree.parse(str(src_path))
    _sections, units = extract_units(tree, JATS_PROFILE)
    for i, u in enumerate(units):
        u["_id"] = i
    title = _article_title(tree)

    translatable = [u for u in units if u["src_plain"]]
    if args.max_units:
        translatable = translatable[: args.max_units]
    print(
        f"article: {title[:80]!r}\n"
        f"units: {len(units)} total, {len(translatable)} sent for "
        f"{language_name(args.lang)} translation"
    )

    tgt_by_id, tokens = await _translate_units(
        translatable, args.lang, title, args.model
    )

    plan = []
    for u in translatable:
        tgt = tgt_by_id.get(u["_id"])
        if not tgt:
            continue
        plan.append({
            "xpath": u["xpath"], "tgt_text": tgt,
            "placeholders": u["placeholders"], "special_chars": u["special_chars"],
        })
    out_tree = apply_units(tree, plan, JATS_PROFILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

    print(f"\ntranslated {len(plan)}/{len(translatable)} units → {out_path}")
    all_ok = _report(tree, out_tree)
    cost = cost_cny(
        model_name=args.model,
        fresh_prompt_tok=tokens["fresh"],
        cached_prompt_tok=tokens["cached"],
        completion_tok=tokens["completion"],
    )
    print(
        f"  tokens: fresh={tokens['fresh']} cached={tokens['cached']} "
        f"completion={tokens['completion']}  →  cost ¥{cost:.4f}"
    )
    # validate output re-parses
    etree.parse(str(out_path))
    print(f"  output is well-formed XML: True")
    if not all_ok:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structure-preserving JATS translation")
    p.add_argument("input", help="path to a JATS XML file")
    p.add_argument("lang", help="target language ISO code (zh, fr, es, …)")
    p.add_argument("--out", help="output path (default: <input>_<lang>.xml)")
    p.add_argument("--model", default=settings.model_first_pass, help="model alias")
    p.add_argument(
        "--max-units", type=int, default=0,
        help="cap units sent to the model (0 = all) — for quick demos",
    )
    return p.parse_args()


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
