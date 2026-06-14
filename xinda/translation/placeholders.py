"""Placeholder protocol shared between extract / translate / apply stages.

The placeholder system is load-bearing. PRESERVE_TAGS get serialized into
`{{PT_<TAG>_<n>}}` tokens; SKIP_TAGS get dropped (only tail text retained);
SPECIAL_CHARS map whitespace to recoverable tokens.

Any change here must be reflected in:
  - pipeline/stages/extract.py (this consumes PRESERVE_TAGS/SKIP_TAGS)
  - pipeline/stages/apply.py (this consumes the placeholder format on writeback)
  - translation/prompts.py (the prompt mentions the format to the model)
"""

from __future__ import annotations

LTX_NS = "http://dlmf.nist.gov/LaTeXML"
NS = {"ltx": LTX_NS}


def _tag(local: str) -> str:
    return f"{{{LTX_NS}}}{local}"


# Inline elements to preserve verbatim (serialized + placeholder-tokenized)
PRESERVE_TAGS: set[str] = {
    _tag("Math"),
    _tag("math"),
    _tag("cite"),
    _tag("bibref"),
    _tag("bibrefphrase"),
    _tag("tag"),
    _tag("ref"),
    _tag("indexmark"),
    _tag("break"),
    _tag("label"),
    _tag("pageref"),
    _tag("eqref"),
    _tag("autoref"),
    _tag("url"),
    _tag("XMRef"),
    _tag("footnote"),
}

# Block elements to drop entirely (only tail text kept)
SKIP_TAGS: set[str] = {
    _tag("XMath"),
    _tag("equation"),
    _tag("graphics"),
    _tag("figure"),
    _tag("bibitem"),
    _tag("biblist"),
    _tag("bibliography"),
    _tag("bibblock"),
    _tag("creator"),
    _tag("toccaption"),
    _tag("toctitle"),
    _tag("tags"),
    _tag("pagination"),
    _tag("picture"),
    _tag("resource"),
    _tag("table"),
    _tag("tabular"),
    _tag("thead"),
    _tag("tbody"),
    _tag("tr"),
    _tag("td"),
    _tag("itemize"),
    _tag("enumerate"),
}

# {{NL}} etc. mapping (whitespace -> recoverable token)
SPECIAL_CHARS: dict[str, str] = {
    "\n": "{{NL}}",
    "\t": "{{TAB}}",
    "\r": "{{RE}}",
}

PLACEHOLDER_FORMAT = "{{{{PT_{tag}_{index}}}}}"
