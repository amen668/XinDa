"""Format profiles — one knob set per source-XML dialect.

The placeholder contract (`translation/placeholders.py`) and the extract/apply
logic are dialect-agnostic *in shape* but were hard-coded to LaTeXML's namespace
and tag names. A `FormatProfile` lifts those constants out so the same machinery
drives a second XML frontend.

Two profiles ship today:

- `LTX_PROFILE` — LaTeXML output (the arXiv backbone). Mirrors the constants in
  `pipeline/stages/extract.py` / `apply.py` exactly, so a paper extracted via the
  generalized `formats.xml_units` path yields byte-identical units to the legacy
  stage (guarded by `tests/test_format_parity.py`).
- `JATS_PROFILE` — Journal Article Tag Suite, the publishing-world standard
  (PMC / CrossRef / CNKI / most journal production systems emit JATS XML). This is
  the deployment-facing frontend: journals already hold their full text as JATS, so
  structure-preserving JATS→JATS translation is the realistic integration point.

A profile names: the namespace map used for xpath, the set of inline elements to
*preserve* verbatim (tokenized into `{{PT_…}}` placeholders), the set to *skip*
(drop, keep tail text), the block elements that become translation units (mapped to
a `UnitKind` string), the sectioning elements, and the local name of the heading
element inside a section. `wrapper_ns` is the default namespace used when re-parsing
a restored fragment on apply (empty string ⇒ null namespace, as JATS uses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

LTX_NS = "http://dlmf.nist.gov/LaTeXML"
MML_NS = "http://www.w3.org/1998/Math/MathML"
XLINK_NS = "http://www.w3.org/1999/xlink"


@dataclass(frozen=True)
class FormatProfile:
    """Per-dialect constants that parametrize extract/apply."""

    name: str
    ns: dict[str, str]                 # namespace map for xpath evaluation
    preserve_tags: frozenset[str]      # fully-qualified {ns}local — tokenized verbatim
    skip_tags: frozenset[str]          # fully-qualified {ns}local — dropped (tail kept)
    unit_tags: dict[str, str]          # {ns}local -> UnitKind string
    section_tags: tuple[str, ...]      # fully-qualified {ns}local, outermost→innermost
    title_locals: tuple[str, ...]      # local names whose child holds a section heading
    wrapper_ns: str = ""               # default ns for fragment re-parse on apply
    # element-identity attributes for structure-preservation scoring, tried in order
    key_attrs: tuple[str, ...] = ()
    # Inline formatting elements descended into (their text IS translated): italic,
    # bold, sup/sub, … Only consulted when `tokenize_unknown_blocks` is on.
    inline_passthrough: frozenset[str] = field(default_factory=frozenset)
    # If True, any element inside a unit that is NOT preserve/skip/inline-passthrough
    # is treated as a block and tokenized verbatim (so apply restores it) instead of
    # being descended into. Prevents apply's clear() from destroying block children
    # (figures/tables/lists) embedded in a paragraph. LaTeXML keeps this OFF (its
    # blocks are siblings of <p>, never inside it) for byte-parity with the legacy stage.
    tokenize_unknown_blocks: bool = False
    # If True, a candidate unit nested inside another *prose* unit/preserve/skip element
    # is not emitted (kept only as part of the enclosing block). OFF for LaTeXML parity.
    suppress_nested_units: bool = False
    # Unit tags emitted ONLY when "leaf" (no nested unit-tag descendant) and only when
    # they hold actual prose (≥2 letters or any CJK). For table cells / list items: a
    # cell wrapping a <p> defers to that <p>; a bare-text cell becomes its own unit; a
    # pure-numeric cell (0.05, n=42) is skipped so the model never reformats figures.
    # These tags also do NOT suppress prose units nested in them (a <p> in a <td> stays
    # its own unit), so they are excluded from the nested-unit container set.
    leaf_only_units: frozenset[str] = field(default_factory=frozenset)
    # Table-cell tags: ANY unit at/under one of these (a bare <td>, or a <p> wrapping the
    # cell's stats) is held to the prose test, so pure-data cells — whether bare or
    # p-wrapped — are never sent to the model and silently reformatted.
    table_cell_tags: frozenset[str] = field(default_factory=frozenset)
    # Structure-preservation metric (evaluation/metrics.compute) tag sets: which
    # elements count toward PPA (link/citation-like annotations) and MFR (formulas).
    # Left empty for LaTeXML (metrics.py uses its own hand-curated module constants to
    # preserve byte-parity); set for JATS so the same PPA/MFR works on journal XML.
    metric_annotation_tags: frozenset[str] = field(default_factory=frozenset)
    metric_math_tags: frozenset[str] = field(default_factory=frozenset)


def _q(ns: str, local: str) -> str:
    """Qualify a local name with a namespace (empty ns ⇒ null-namespace local)."""
    return f"{{{ns}}}{local}" if ns else local


# ───────────────────────────── LaTeXML ─────────────────────────────

_LTX_PRESERVE = (
    "Math", "math", "cite", "bibref", "bibrefphrase", "tag", "ref",
    "indexmark", "break", "label", "pageref", "eqref", "autoref", "url",
    "XMRef", "footnote",
)
_LTX_SKIP = (
    "XMath", "equation", "graphics", "figure", "bibitem", "biblist",
    "bibliography", "bibblock", "creator", "toccaption", "toctitle", "tags",
    "pagination", "picture", "resource", "table", "tabular", "thead", "tbody",
    "tr", "td", "itemize", "enumerate",
)
_LTX_SECTIONS = (
    "part", "chapter", "section", "subsection", "subsubsection",
    "paragraph", "appendix",
)

LTX_PROFILE = FormatProfile(
    name="latexml",
    ns={"ltx": LTX_NS},
    preserve_tags=frozenset(_q(LTX_NS, t) for t in _LTX_PRESERVE),
    skip_tags=frozenset(_q(LTX_NS, t) for t in _LTX_SKIP),
    unit_tags={
        _q(LTX_NS, "p"): "paragraph",
        _q(LTX_NS, "title"): "title",
        _q(LTX_NS, "caption"): "caption",
    },
    section_tags=tuple(_q(LTX_NS, t) for t in _LTX_SECTIONS),
    title_locals=("title", "toctitle"),
    wrapper_ns=LTX_NS,
    key_attrs=("tex", "bibrefs", "labelref", "idref", "href", "key"),
)


# ─────────────────────────────── JATS ───────────────────────────────
# JATS body elements live in the null namespace; only embedded MathML carries
# a namespace (mml:). We preserve the *whole* <inline-formula>/<disp-formula>
# wrapper verbatim, so the MathML/tex-math inside is never visited or translated.

_JATS_PRESERVE_NULL = (
    "xref",            # cross-reference (bibr / fig / table / sec / fn) — JATS analog of cite/ref
    "inline-formula",  # wraps mml:math or tex-math
    "disp-formula",    # display equation
    "ext-link",        # external hyperlink (xlink:href)
    "uri", "email",    # literal URIs / addresses
    "inline-graphic",  # inline image (e.g. a small formula rendered as image)
    "label",           # equation/figure label
)
_JATS_PRESERVE = (
    tuple(_q("", t) for t in _JATS_PRESERVE_NULL)
    + (_q(MML_NS, "math"),)        # defensive: bare inline mml:math not wrapped in a formula
)
# JATS sectioning: <sec> nests; abstract is sec-like for heading purposes.
_JATS_SECTIONS = ("sec",)

# Inline formatting whose text IS translated (descended into, not tokenized).
_JATS_INLINE = (
    "italic", "bold", "sup", "sub", "sc", "underline", "overline", "strike",
    "monospace", "roman", "sans-serif", "named-content", "styled-content",
    "abbrev", "italic", "bold", "break", "x",
)

JATS_PROFILE = FormatProfile(
    name="jats",
    ns={"mml": MML_NS, "xlink": XLINK_NS},
    preserve_tags=frozenset(_JATS_PRESERVE),
    # No drop-set: any non-inline block inside a unit (fig/table-wrap/list/…) is
    # tokenized verbatim instead, so apply never destroys it (tokenize_unknown_blocks).
    skip_tags=frozenset(),
    unit_tags={
        # NB: <caption> is intentionally NOT a unit — in JATS it merely wraps
        # <title>/<p>, which are units in their own right; listing it too would
        # double-translate the caption text.
        _q("", "p"): "paragraph",
        _q("", "title"): "title",
        _q("", "article-title"): "title",
        # leaf-only units (see leaf_only_units): table-cell text and bare list items
        _q("", "td"): "table_cell",
        _q("", "th"): "table_cell",
        _q("", "list-item"): "list_item",
    },
    section_tags=tuple(_q("", t) for t in _JATS_SECTIONS),
    title_locals=("title",),
    wrapper_ns="",
    key_attrs=("rid", "ref-type", "{%s}href" % XLINK_NS, "id"),
    inline_passthrough=frozenset(_q("", t) for t in _JATS_INLINE),
    tokenize_unknown_blocks=True,
    suppress_nested_units=True,
    leaf_only_units=frozenset(_q("", t) for t in ("td", "th", "list-item")),
    table_cell_tags=frozenset(_q("", t) for t in ("td", "th")),
    # PPA = citation/link-like inline elements; MFR = formula wrappers. Both
    # null-namespace; identity comes from key_attrs (rid/ref-type/xlink:href/id).
    metric_annotation_tags=frozenset(
        _q("", t) for t in ("xref", "ext-link", "uri", "email", "inline-graphic", "label")
    ),
    metric_math_tags=frozenset(_q("", t) for t in ("inline-formula", "disp-formula")),
)


PROFILES: dict[str, FormatProfile] = {
    LTX_PROFILE.name: LTX_PROFILE,
    JATS_PROFILE.name: JATS_PROFILE,
}


def get_profile(name: str) -> FormatProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown format profile {name!r}; known: {sorted(PROFILES)}"
        ) from None
