"""Annotation Preservation (PPA) and Math Formula Retention (MFR).

A preserved element is matched by its **content** (serialized subtree, with
volatile `xml:id` attributes stripped), NOT by its tree position. Position is
captured separately by the ORDERED variants (Kendall tau). This split matters:
translation legitimately re-indexes sibling elements (text runs split/merge
around inline math), which shifts an element's xpath while leaving the formula
or citation byte-identical. An xpath-keyed signature scores those benign shifts
as "lost" (observed: an eess paper with all 15 formulas intact scored MFR 56);
a content key scores them as preserved (MFR 100) and lets the ordered variant
report any real positional drift on its own axis.

v3's two numbers therefore read as: unordered = "is this content still present"
(multiset), ordered = "is it in the same relative order" (Kendall tau).
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

from lxml import etree

from xinda.translation.placeholders import LTX_NS, NS

if TYPE_CHECKING:
    from xinda.formats.profiles import FormatProfile

_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def _tag(local: str) -> str:
    return f"{{{LTX_NS}}}{local}"


# Tags counted as preserved annotations.
# NOTE: `caption`/`footnote` are intentionally EXCLUDED — they hold translatable
# prose whose text legitimately changes, so a verbatim-preservation check is the
# wrong tool (their quality is covered by COMET/judges/RCS). `XMath` is excluded as
# the inner-semantics duplicate of its `Math` parent (would double-count formulas).
ANNOTATION_TAGS: set[str] = {
    _tag("cite"), _tag("bibref"), _tag("ref"), _tag("label"),
    _tag("tag"), _tag("eqref"), _tag("equation"),
    _tag("url"), _tag("Math"), _tag("math"), _tag("bibrefphrase"),
    _tag("autoref"), _tag("pageref"), _tag("XMRef"),
    _tag("indexmark"),
}

# Tags counted for math fidelity (one entry per formula — not the inner XMath).
MATH_TAGS: set[str] = {
    _tag("Math"), _tag("equation"),
}

# Attributes that carry an element's stable cross-lingual identity (link/citation
# targets, source LaTeX), tried in order before falling back to a content hash.
_KEY_ATTRS: tuple[str, ...] = ("tex", "bibrefs", "labelref", "idref", "href", "key")


def _content_hash(elem: etree._Element, canonical: bool = False) -> str:
    """Position-independent fingerprint of an element: its serialized subtree with
    volatile `xml:id` attributes stripped (LaTeXML's ids are positional and survive
    apply verbatim, but stripping them keeps the key robust if anything renumbers).

    `canonical` serializes via C14N instead of plain `tostring`. JATS apply re-inserts
    an inline element through a wrapper re-parse, which renormalizes namespace-decl
    placement/whitespace — byte-different but semantically identical (a `<mml:math>` is
    unchanged). C14N collapses that benign noise so a preserved formula isn't scored as
    lost. LaTeXML keeps plain hashing (canonical=False) for byte-parity with stored values."""
    clone = copy.deepcopy(elem)
    for e in clone.iter():
        if not isinstance(e.tag, str):
            continue
        for k in list(e.attrib):
            if k == _XML_ID or k.endswith("}id"):
                del e.attrib[k]
    raw = etree.tostring(clone, method="c14n") if canonical else etree.tostring(clone)
    return hashlib.md5(raw).hexdigest()  # noqa: S324 — non-crypto fingerprint


def _element_key(
    elem: etree._Element,
    key_attrs: tuple[str, ...] = _KEY_ATTRS,
    canonical: bool = False,
) -> str:
    """Stable, position- and translation-independent identity of an element.

    The full serialized subtree is the WRONG key for container elements that wrap
    rendered or translated surface (a `cite` wraps "[1]" brackets, a `Math` wraps a
    `text="r"` rendering): those change benignly and would read as loss. Instead key
    by what actually identifies the element — its source LaTeX (`tex`) or its
    link/citation target (`bibrefs`/`labelref`/`href`/…). A `cite` with no own target
    inherits its descendant `bibref` keys. Pure structural elements with no such
    attribute fall back to a content hash.

    `key_attrs` is profile-specific (LaTeXML: tex/bibrefs/…; JATS: rid/ref-type/
    xlink:href/id) so the same identity logic serves both dialects.
    """
    local = elem.tag.split("}")[-1]
    for attr in key_attrs:
        v = elem.get(attr)
        if v:
            return f"{local}:{attr}={v}"
    if local == "cite":
        keys = [
            b.get("bibrefs", "")
            for b in elem.iter()
            if isinstance(b.tag, str) and b.tag.endswith("}bibref")
        ]
        if any(keys):
            return f"cite:{'|'.join(keys)}"
    return f"{local}:{_content_hash(elem, canonical)}"


def _collect_signatures(
    tree: etree._ElementTree,
    tags: Iterable[str],
    key_attrs: tuple[str, ...] = _KEY_ATTRS,
    canonical: bool = False,
) -> list[str]:
    """Document-order list of `key#occurrence` signatures for tags ∈ `tags`.

    The key is content/identity-based (not xpath) so benign sibling re-indexing during
    translation doesn't read as element loss. The `#occurrence` suffix disambiguates
    identical-identity elements (e.g. many inline `$r$`), giving multiset semantics to
    the unordered set-intersection AND keeping items unique for the ordered Kendall tau.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    root = tree.getroot()
    tag_set = set(tags)
    # iterate in document order
    for elem in root.iter():
        if not isinstance(elem.tag, str) or elem.tag not in tag_set:
            continue
        sig = _element_key(elem, key_attrs, canonical)
        idx = seen.get(sig, 0)
        seen[sig] = idx + 1
        out.append(f"{sig}#{idx}")
    return out


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Order-preservation Kendall tau between two sequences sharing the same items.

    Operates only on the items present in BOTH lists; non-shared items count
    as missing and are not part of the order comparison (they reduce the
    intersection size which downstream callers also report).
    """
    common = [x for x in order_a if x in set(order_b)]
    if len(common) < 2:
        return 1.0 if common else 0.0
    rank_a = {x: i for i, x in enumerate(order_a)}
    rank_b = {x: i for i, x in enumerate(order_b)}
    # Number of concordant pairs (i, j) where rank_a and rank_b agree on order.
    n = len(common)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = common[i], common[j]
            sa = rank_a[xi] - rank_a[xj]
            sb = rank_b[xi] - rank_b[xj]
            if sa * sb > 0:
                concordant += 1
            elif sa * sb < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def compute(
    xml_src: str,
    xml_tgt: str,
    *,
    profile: "FormatProfile | None" = None,
) -> dict[str, float | dict[str, int]]:
    """Return PPA, MFR + ordered variants on (src, tgt) XML pair.

    Default (`profile=None`) measures LaTeXML output with the module-level
    ANNOTATION_TAGS/MATH_TAGS/_KEY_ATTRS — byte-identical to the original behavior.
    Passing a `FormatProfile` that carries `metric_annotation_tags`/`metric_math_tags`
    (e.g. `formats.JATS_PROFILE`) measures that dialect instead, using its `key_attrs`
    for element identity. This is what lets the same PPA/MFR serve JATS translations.
    """
    if profile is not None and profile.metric_annotation_tags:
        ann_tags: Iterable[str] = profile.metric_annotation_tags
        math_tags: Iterable[str] = profile.metric_math_tags
        key_attrs = profile.key_attrs or _KEY_ATTRS
        canonical = True  # JATS apply renormalizes serialization → C14N content hash
    else:
        ann_tags, math_tags, key_attrs = ANNOTATION_TAGS, MATH_TAGS, _KEY_ATTRS
        canonical = False  # LaTeXML: plain hash, byte-parity with stored values

    # collect_ids=False: don't build the xml:id table, so a duplicate xml:id in the
    # translated output (LaTeXML apply can re-emit a placeholder element whose id repeats)
    # doesn't raise "ID … already defined". Ids are irrelevant to the metric anyway —
    # _content_hash strips them — we only need the element tree.
    _parser = etree.XMLParser(collect_ids=False)
    src_tree = etree.parse(xml_src, _parser)
    tgt_tree = etree.parse(xml_tgt, _parser)

    ann_src = _collect_signatures(src_tree, ann_tags, key_attrs, canonical)
    ann_tgt = _collect_signatures(tgt_tree, ann_tags, key_attrs, canonical)
    math_src = _collect_signatures(src_tree, math_tags, key_attrs, canonical)
    math_tgt = _collect_signatures(tgt_tree, math_tags, key_attrs, canonical)

    preserved_ann = set(ann_src) & set(ann_tgt)
    preserved_math = set(math_src) & set(math_tgt)

    ppa = (100.0 * len(preserved_ann) / len(ann_src)) if ann_src else 100.0
    mfr = (100.0 * len(preserved_math) / len(math_src)) if math_src else 100.0

    ppa_ordered = 100.0 * _kendall_tau(ann_src, ann_tgt)
    mfr_ordered = 100.0 * _kendall_tau(math_src, math_tgt)

    return {
        "ppa": ppa,
        "ppa_ordered": ppa_ordered,
        "mfr": mfr,
        "mfr_ordered": mfr_ordered,
        "stats": {
            "annotation": {
                "original": len(ann_src),
                "translated": len(ann_tgt),
                "preserved": len(preserved_ann),
            },
            "math": {
                "original": len(math_src),
                "translated": len(math_tgt),
                "preserved": len(preserved_math),
            },
        },
    }
