"""Head-to-head JATS translation systems for the structure-preservation experiment.

Three systems translate the SAME units with the SAME model/prompt scaffold, differing
ONLY in how document structure is handled — so the comparison isolates the structure
strategy, not the model:

- **contract**  — our placeholder contract: inline elements (xref/formula/…) are
  tokenized out of the model's view and restored verbatim. (= `cli.jats_translate`.)
- **raw_xml**   — the prior-art approach (Science Across Languages): the raw inline XML
  is sent to the model with an instruction to reproduce every tag exactly, then parsed
  back. When the model drops/mangles a tag, structure is lost — which the PPA/MFR metric
  then measures. This is a *fair* replication (the model IS told to keep tags), framed in
  the paper as a design justification, not a takedown.
- **naive**     — plain text only, no structure concept; inline elements inside a unit
  are gone by construction (coverage/structure floor).

`translate_jats(path, lang, model, system)` returns `(out_tree, tokens, n_units)`.
Reuses `cli.jats_translate._translate_units` (the shared engine) + `formats` extract/apply.
"""

from __future__ import annotations

from typing import Any

from lxml import etree

from xinda.cli.jats_translate import _article_title, _translate_units
from xinda.formats import JATS_PROFILE, apply_units, extract_units
from xinda.formats.profiles import FormatProfile
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

SYSTEMS = ("contract", "raw_xml", "naive", "abstract")

# Fair instruction for the raw-XML system: tell the model to preserve every tag (this is
# what the prior art does — they translate JATS in place). Structure still breaks when the
# model slips, which is exactly the phenomenon under test.
_RAW_RULE = (
    "STRUCTURE: The source text contains inline XML tags (e.g. <xref ...>, "
    "<inline-formula>...</inline-formula>, <ext-link ...>, <mml:math>...). Reproduce "
    "EVERY tag and ALL its attributes exactly and unchanged, in the correct position; "
    "translate only the human-readable text between tags. Do not drop, merge, reorder, "
    "or rewrite any tag or attribute."
)


def _raw_units(
    tree: etree._ElementTree, profile: FormatProfile
) -> list[dict[str, Any]]:
    """Same unit selection as the contract path, but each unit's source is its INNER raw
    XML (tags intact) rather than placeholder-tokenized text."""
    _secs, units = extract_units(tree, profile)
    out: list[dict[str, Any]] = []
    for i, u in enumerate(units):
        els = tree.xpath(u["xpath"])
        if not els:
            continue
        el = els[0]
        inner = (el.text or "") + "".join(
            etree.tostring(c, encoding="unicode") for c in el
        )
        if not inner.strip():
            continue
        out.append({
            "_id": i, "xpath": u["xpath"], "kind": u["kind"],
            "src_text": inner, "char_count": u["char_count"],
        })
    return out


def _apply_raw(
    tree: etree._ElementTree, plan: list[dict[str, Any]], profile: FormatProfile
) -> etree._ElementTree:
    """Write each system output back by parsing it as an XML fragment. A model that broke
    the XML (truncation/nesting/dropped tag) fails to parse → falls back to escaped text,
    i.e. the inline elements vanish — the structure loss the metric is meant to catch."""
    import copy
    out = copy.deepcopy(tree)
    for entry in sorted(plan, key=lambda p: p["xpath"].count("/"), reverse=True):
        els = out.xpath(entry["xpath"])
        if not els:
            continue
        el = els[0]
        el.clear()
        tgt = entry["tgt_text"]
        wrapper = (
            f'<wrapper xmlns="{profile.wrapper_ns}">{tgt}</wrapper>'
            if profile.wrapper_ns else f"<wrapper>{tgt}</wrapper>"
        )
        try:
            frag = etree.fromstring(wrapper)
        except etree.XMLSyntaxError:
            el.text = tgt  # model broke the markup → structure lost (the point)
            continue
        el.text = frag.text
        for c in frag:
            el.append(c)
    return out


async def translate_jats(
    jats_path: str, lang: str, model: str, system: str,
    profile: FormatProfile = JATS_PROFILE,
) -> tuple[etree._ElementTree, dict[str, int], int]:
    """Translate an XML file under one of SYSTEMS. Returns (out_tree, tokens, n_units).

    `profile` selects the XML dialect: JATS_PROFILE (journal, zh→en) or LTX_PROFILE
    (LaTeXML/arXiv, en→zh). The same placeholder contract drives both — this is the
    format-agnostic claim, and lets the head-to-head run on either leg.
    """
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
    tree = etree.parse(jats_path)
    title = _article_title(tree)

    if system == "raw_xml":
        units = _raw_units(tree, profile)
        tgt, toks = await _translate_units(
            units, lang, title, model, extra_system=_RAW_RULE,
        )
        plan = [
            {"xpath": u["xpath"], "tgt_text": tgt[u["_id"]]}
            for u in units if u["_id"] in tgt
        ]
        return _apply_raw(tree, plan, profile), toks, len(units)

    # contract / naive / abstract all use extract_units + apply_units (placeholder
    # contract), differing only in WHICH units are translated.
    _secs, units = extract_units(tree, profile)
    for i, u in enumerate(units):
        u["_id"] = i
    translatable = [u for u in units if u["src_plain"]]

    if system == "abstract":
        # status-quo baseline: translate ONLY the abstract (most journals' current
        # multilingual practice is abstract-level). Coverage will be a fraction of full
        # text — that gap is the differentiation vs the摘要级 status quo.
        translatable = [u for u in translatable if "abstract" in u["xpath"].lower()]

    if system == "naive":
        items = [
            {"_id": u["_id"], "src_text": u["src_plain"], "kind": u["kind"],
             "char_count": u["char_count"]}
            for u in translatable
        ]
        tgt, toks = await _translate_units(items, lang, title, model)
        plan = [
            {"xpath": u["xpath"], "tgt_text": tgt[u["_id"]],
             "placeholders": {}, "special_chars": {}}
            for u in translatable if u["_id"] in tgt
        ]
        return apply_units(tree, plan, profile), toks, len(translatable)

    # contract
    tgt, toks = await _translate_units(translatable, lang, title, model)
    plan = [
        {"xpath": u["xpath"], "tgt_text": tgt[u["_id"]],
         "placeholders": u["placeholders"], "special_chars": u["special_chars"]}
        for u in translatable if u["_id"] in tgt
    ]
    return apply_units(tree, plan, profile), toks, len(translatable)
