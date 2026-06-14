"""Dialect-agnostic XML ↔ translation-unit conversion.

These are the pure-function core of extract/apply, parametrized by a
`FormatProfile` instead of LaTeXML-hard-coded constants. The LaTeXML pipeline
stages keep their own copy for now (validated, load-bearing); this module powers
the JATS frontend and is held byte-compatible with them by
`tests/test_format_parity.py`.

`extract_units(tree, profile)` → (sections, units): walk the document, record
sectioning elements (with depth + parent) and emit one record per translatable
block element, serializing preserved inline elements to `{{PT_<TAG>_<n>}}`
placeholders and whitespace to special-char tokens — exactly the contract in
`translation/placeholders.py`.

`apply_units(src_tree, plan, profile)` → mutates a copy: for each (xpath, target
text, placeholders, special_chars), find the element, clear it, and rewrite with
the translated text, restoring placeholders into an inline fragment.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from lxml import etree

from xinda.formats.profiles import FormatProfile
from xinda.translation.placeholders import (
    PLACEHOLDER_FORMAT,
    SPECIAL_CHARS,
)

_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]+?\}\}")


# ─────────────────────────────── extract ───────────────────────────────


def extract_units(
    tree: etree._ElementTree, profile: FormatProfile
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (sections, units) for a parsed XML tree under `profile`."""
    root = tree.getroot()
    sections = _collect_sections(tree, root, profile)
    units = _collect_units(tree, root, sections, profile)
    return sections, units


def _collect_sections(
    tree: etree._ElementTree, root: etree._Element, profile: FormatProfile
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    section_stack: list[dict[str, Any]] = []
    ord_counters: dict[str, int] = {}
    section_set = set(profile.section_tags)

    def walk(elem: etree._Element) -> None:
        tag = elem.tag
        if isinstance(tag, str) and tag in section_set:
            xpath = tree.getpath(elem)
            parent_xpath = section_stack[-1]["xpath"] if section_stack else None
            ord_key = parent_xpath or "<root>"
            ord_counters[ord_key] = ord_counters.get(ord_key, 0) + 1
            row = {
                "xpath": xpath,
                "parent_xpath": parent_xpath,
                "depth": len(section_stack),
                "ord": ord_counters[ord_key],
                "heading": _section_heading(elem, profile),
            }
            rows.append(row)
            section_stack.append(row)
            for child in elem:
                walk(child)
            section_stack.pop()
        else:
            for child in elem:
                walk(child)

    walk(root)
    return rows


def _section_heading(
    section_elem: etree._Element, profile: FormatProfile
) -> str | None:
    """First child heading element's plain text (preserved inline → '[…]')."""
    title = None
    for child in section_elem:
        local = child.tag.split("}")[-1] if isinstance(child.tag, str) else None
        if local in profile.title_locals:
            title = child
            break
    if title is None:
        return None
    parts: list[str] = []
    for node in title.iter():
        if isinstance(node.tag, str) and node.tag in profile.preserve_tags:
            parts.append("[…]")
            if node.tail:
                parts.append(node.tail)
            continue
        if node.text:
            parts.append(node.text)
    return "".join(parts).strip() or None


def _collect_units(
    tree: etree._ElementTree,
    root: etree._Element,
    sections: list[dict[str, Any]],
    profile: FormatProfile,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    section_xpaths_sorted = sorted(
        (s["xpath"] for s in sections), key=lambda x: -len(x)
    )
    section_set = set(profile.section_tags)
    # An element nested inside another unit/skip/preserve block must NOT be emitted
    # as its own unit: its text is already carried by (or tokenized into) the enclosing
    # block, and emitting it would make ApplyXML clear a parent and a child, wiping the
    # child's rewrite. We therefore keep only OUTERMOST units. (LaTeXML leaves this off
    # for byte-parity — its blocks never nest p/title/caption.)
    unit_tag_set = set(profile.unit_tags)
    # Leaf-only units (table cells / list items) do NOT suppress prose units nested in
    # them — a <p> inside a <td> stays its own unit — so they're kept out of the
    # container set that triggers nested-unit suppression.
    container_tags: set[str] = set()
    if profile.suppress_nested_units:
        container_tags = (
            (unit_tag_set - profile.leaf_only_units)
            | profile.skip_tags
            | profile.preserve_tags
        )

    ord_idx = 0
    for elem in root.iter():
        tag = elem.tag
        if not isinstance(tag, str):
            continue
        kind = profile.unit_tags.get(tag)
        if kind is None:
            continue

        if container_tags and _has_container_ancestor(elem, container_tags):
            continue

        # A leaf-only unit defers to any unit-tag descendant it wraps (e.g. a <td>
        # containing <p> → the <p> is the unit, not the cell), avoiding double-translation.
        if tag in profile.leaf_only_units and _has_unit_descendant(elem, unit_tag_set):
            continue

        # a heading-element directly under a sectioning element → section_heading
        if kind == "title":
            parent = elem.getparent()
            if (
                parent is not None
                and isinstance(parent.tag, str)
                and parent.tag in section_set
            ):
                kind = "section_heading"

        placeholders, special_chars, text = _serialize_unit(elem, profile)
        if not text:
            continue

        src_plain = text
        for ph in placeholders:
            src_plain = src_plain.replace(ph, "")
        for ph in special_chars:
            src_plain = src_plain.replace(ph, " ")
        src_plain = " ".join(src_plain.split())

        # Pure-data cells (0.05, n=42, "C1=85 (5.70%)") carry no prose: skip them so the
        # model never "helpfully" reformats statistics. Applies to bare leaf cells AND to
        # any unit sitting inside a table cell (a <p> that wraps the cell's numbers).
        guarded = tag in profile.leaf_only_units or (
            bool(profile.table_cell_tags)
            and _has_container_ancestor(elem, set(profile.table_cell_tags))
        )
        if guarded and not _is_prose(src_plain):
            continue

        xpath = tree.getpath(elem)
        section_xpath = next(
            (
                sx for sx in section_xpaths_sorted
                if xpath.startswith(sx + "/") or xpath == sx
            ),
            None,
        )

        ord_idx += 1
        rows.append({
            "ord": ord_idx,
            "kind": kind,
            "xpath": xpath,
            "src_text": text,
            "src_plain": src_plain,
            "placeholders": placeholders,
            "special_chars": special_chars,
            "char_count": len(src_plain),
            "section_xpath": section_xpath,
        })

    return rows


def _has_container_ancestor(
    elem: etree._Element, container_tags: set[str]
) -> bool:
    parent = elem.getparent()
    while parent is not None:
        if isinstance(parent.tag, str) and parent.tag in container_tags:
            return True
        parent = parent.getparent()
    return False


def _has_unit_descendant(elem: etree._Element, unit_tags: set[str]) -> bool:
    for node in elem.iterdescendants():
        if isinstance(node.tag, str) and node.tag in unit_tags:
            return True
    return False


def _is_prose(s: str) -> bool:
    """True if `s` is a translatable textual label, not statistical data.

    A cell qualifies if it has any CJK char, or if it has a real word (≥3 alphabetic
    chars) AND more letters than digits. The second clause is what protects data cells
    like ``76.76 (75.23 to 78.28)`` or ``n=90`` (the stray ``to``/``n`` is not a ≥3-letter
    word, and digits dominate) from being sent to the model and silently reformatted,
    while still translating headers like ``Number of classes`` / ``组别``."""
    if any("㐀" <= ch <= "鿿" for ch in s):
        return True
    if not re.search(r"[^\W\d_]{3,}", s):
        return False
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    return letters > digits


def _serialize_unit(
    elem: etree._Element, profile: FormatProfile
) -> tuple[dict[str, str], dict[str, str], str]:
    """Flatten one block element to (placeholders, special_chars, text)."""
    placeholders: dict[str, str] = {}
    parts: list[str] = []

    def tokenize(node: etree._Element) -> None:
        tag_name = node.tag.split("}")[-1].upper().replace("-", "")
        ph = PLACEHOLDER_FORMAT.format(tag=tag_name, index=len(placeholders) + 1)
        placeholders[ph] = etree.tostring(
            node, encoding="unicode", with_tail=False
        )
        parts.append(ph)
        if node.tail:
            parts.append(node.tail)

    def recurse(node: etree._Element) -> None:
        ntag = node.tag
        if not isinstance(ntag, str):
            return
        if ntag in profile.preserve_tags:
            tokenize(node)
            return
        if ntag in profile.skip_tags:
            if node.tail:
                parts.append(node.tail)
            return
        # A non-inline block embedded in a unit (e.g. <fig>/<table-wrap>/<list> inside
        # a <p>) is tokenized verbatim so apply restores it rather than destroying it
        # on clear(). Inline formatting (italic/bold/sup/…) is descended into normally.
        if (
            profile.tokenize_unknown_blocks
            and ntag not in profile.inline_passthrough
        ):
            tokenize(node)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            recurse(child)
        if node.tail:
            parts.append(node.tail)

    for child in elem:
        recurse(child)
    text = ((elem.text or "") + "".join(parts)).strip()

    special_chars: dict[str, str] = {}
    for char, ph in SPECIAL_CHARS.items():
        if char in text:
            text = text.replace(char, ph)
            special_chars[ph] = char

    return placeholders, special_chars, text


# ──────────────────────────────── apply ────────────────────────────────


def apply_units(
    src_tree: etree._ElementTree,
    plan: list[dict[str, Any]],
    profile: FormatProfile,
) -> etree._ElementTree:
    """Return a *new* tree (deep-copied) with translations written back.

    `plan` entries: {xpath, tgt_text, placeholders, special_chars}. They are
    applied deepest-xpath-first so child rewrites don't disturb parent xpaths.
    """
    tree = copy.deepcopy(src_tree)
    root = tree.getroot()
    ordered = sorted(plan, key=lambda p: p["xpath"].count("/"), reverse=True)

    for entry in ordered:
        tgt = entry["tgt_text"]
        for ph, ch in (entry.get("special_chars") or {}).items():
            tgt = tgt.replace(ph, ch)
        try:
            elements = root.xpath(entry["xpath"], namespaces=profile.ns)
        except etree.XPathEvalError:
            continue
        for elem in elements:
            elem.clear()
            _rewrite_element(elem, tgt, entry.get("placeholders") or {}, profile)

    return tree


def _rewrite_element(
    elem: etree._Element,
    tgt: str,
    placeholders: dict[str, str],
    profile: FormatProfile,
) -> None:
    """Rewrite `elem`'s content from translated `tgt`, restoring placeholders.

    Built by *structured insertion*, not string concatenation + reparse: free text
    segments are assigned to `.text`/`.tail` (lxml escapes them), and only the clean
    serialized placeholder XML is parsed back into elements. This is what lets the
    translated prose contain literal ``<``, ``>`` or ``&`` (``p<0.05``, ``R&D``,
    ``<65 years``) without collapsing the restored inline tags into escaped text.
    """
    elem.text = ""
    if not placeholders:
        elem.text = tgt
        return

    # case-insensitive lookup, whitespace-tolerant (model occasionally mangles casing)
    lower_map: dict[str, str] = {ph.lower(): content for ph, content in placeholders.items()}
    flat_map: dict[str, str] = {
        re.sub(r"\s", "", k): v for k, v in lower_map.items()
    }

    last: etree._Element | None = None

    def add_text(s: str) -> None:
        nonlocal last
        if not s:
            return
        if last is None:
            elem.text = (elem.text or "") + s
        else:
            last.tail = (last.tail or "") + s

    idx = 0
    for m in _PLACEHOLDER_RE.finditer(tgt):
        add_text(tgt[idx:m.start()])
        idx = m.end()
        found = m.group(0)
        content = lower_map.get(found.lower()) or flat_map.get(
            re.sub(r"\s", "", found.lower())
        )
        if content is None:
            add_text(found)  # unknown token → leave literal
            continue
        wrapper = (
            f'<wrapper xmlns="{profile.wrapper_ns}">{content}</wrapper>'
            if profile.wrapper_ns
            else f"<wrapper>{content}</wrapper>"
        )
        try:
            frag = etree.fromstring(wrapper)
        except etree.XMLSyntaxError:
            add_text(found)
            continue
        if frag.text:
            add_text(frag.text)
        for sub in frag:
            elem.append(sub)
            last = sub
    add_text(tgt[idx:])
