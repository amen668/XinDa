"""5-category claim normalization + drift detection for Fact-Anchor protocol.

For each of the 5 claim types, define:
- a `normalize_<type>` function that maps raw surface to a canonical form
- a `check_<type>_drift` function that compares src vs tgt claims and
  returns a DriftType enum value + magnitude
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from xinda.db.models import ClaimType, DriftType


@dataclass
class ClaimRecord:
    """In-memory claim record (mirrors VerifiableClaim row)."""

    claim_type: ClaimType
    surface_form: str
    normalized: str
    metadata: dict[str, Any]
    span_start: int | None = None
    span_end: int | None = None

    def __post_init__(self) -> None:
        # The LLM occasionally emits `metadata` as a str/None instead of an
        # object; downstream check_* functions call `.metadata.get(...)`.
        if not isinstance(self.metadata, dict):
            self.metadata = {}


# ────────────────────────── normalization ──────────────────────────


_NUMERIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([%KMBkmb]|×10[⁰¹²³⁴⁵⁶⁷⁸⁹\^\d]+|\^\d+)?"
)


def normalize_numeric(surface: str) -> tuple[str, dict[str, Any]]:
    """Strip % / unit modifier and return canonical decimal string + meta."""
    s = surface.strip()
    m = _NUMERIC_RE.match(s)
    if not m:
        return s, {}
    value, suffix = m.group(1), (m.group(2) or "").strip()
    meta: dict[str, Any] = {}
    if suffix:
        meta["unit"] = suffix
    # detect precision
    if "." in value:
        meta["precision"] = len(value.split(".")[1])
    else:
        meta["precision"] = 0
    return value, meta


_CITATION_AUTHOR_YEAR_RE = re.compile(
    r"([A-Z][A-Za-z\-']+)(?:\s+et\s+al\.?|\s+&\s+[A-Z][A-Za-z\-']+)?\s*[\(\[]?(\d{4})[\)\]]?"
)
_CITATION_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def normalize_citation(surface: str) -> tuple[str, dict[str, Any]]:
    """`Vaswani et al. (2017)` -> `Vaswani2017`; `[12]` -> `ref:12`."""
    s = surface.strip()
    m = _CITATION_AUTHOR_YEAR_RE.search(s)
    if m:
        author, year = m.group(1), m.group(2)
        return f"{author}{year}", {"first_author": author, "year": int(year)}
    m2 = _CITATION_NUMERIC_RE.search(s)
    if m2:
        nums = [int(n.strip()) for n in m2.group(1).split(",")]
        return f"ref:{','.join(str(n) for n in nums)}", {"refs": nums}
    return s, {}


def normalize_method_name(surface: str) -> tuple[str, dict[str, Any]]:
    """Preserve as-is, just trim whitespace."""
    return surface.strip(), {}


def normalize_symbol(surface: str) -> tuple[str, dict[str, Any]]:
    """`α = 0.1` -> `α=0.1` (whitespace-collapsed)."""
    return re.sub(r"\s+", "", surface.strip()), {}


_COMPARISON_RE = re.compile(
    r"(outperforms?|exceeds?|surpasses?|beats?|defeats?|"
    r"(?:is\s+)?(?:smaller|larger|better|worse|higher|lower|greater|less)\s+than|"
    r"=|equals?)"
    r"\s+([A-Za-z0-9\-_]+)"
    r"(?:\s+by\s+([+-]?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)


def normalize_comparison(surface: str) -> tuple[str, dict[str, Any]]:
    """Parse `outperforms BERT by 2.3` -> normalized `gt(BERT,+2.3)`."""
    s = surface.strip()
    m = _COMPARISON_RE.search(s)
    if not m:
        return s, {}
    verb, baseline, delta = m.group(1).lower(), m.group(2), m.group(3)
    direction = _classify_direction(verb)
    meta: dict[str, Any] = {"baseline": baseline, "direction": direction}
    if delta is not None:
        meta["delta"] = float(delta)
    op = {">": "gt", "<": "lt", "=": "eq", "?": "rel"}[direction]
    norm = f"{op}({baseline}{',' + ('+' + delta if not delta.startswith('-') else delta) if delta else ''})"
    return norm, meta


def _classify_direction(verb: str) -> str:
    verb = verb.lower()
    if any(w in verb for w in [
        "outperforms", "outperform", "exceeds", "exceed",
        "surpasses", "surpass", "beats", "beat", "defeats", "defeat",
        "larger", "better", "higher", "greater",
    ]):
        return ">"
    if any(w in verb for w in ["smaller", "worse", "lower", "less"]):
        return "<"
    if "equal" in verb or verb == "=":
        return "="
    return "?"


# ────────────────────────── drift detection ──────────────────────────


def check_numeric_drift(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    """0 = identical; positive magnitude = relative numeric error."""
    try:
        v_src = float(src.normalized)
        v_tgt = float(tgt.normalized)
    except ValueError:
        return DriftType.numeric_drift, 1.0
    if v_src == v_tgt:
        # also require the % suffix / unit to match
        if src.metadata.get("unit") != tgt.metadata.get("unit"):
            return DriftType.numeric_drift, 0.01
        return DriftType.verified, 0.0
    if v_src == 0:
        return DriftType.numeric_drift, abs(v_tgt)
    return DriftType.numeric_drift, abs(v_tgt - v_src) / abs(v_src)


def check_citation_swap(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    """Same author+year (or same numeric refs) -> verified; else citation_swap."""
    if src.normalized == tgt.normalized:
        return DriftType.verified, 0.0
    # tolerate Chinese transliteration: if same year + author surname appears,
    # accept. (Heuristic; refined by qwen judge in FactVerify stage.)
    src_year = src.metadata.get("year")
    tgt_year = tgt.metadata.get("year")
    if src_year is not None and src_year == tgt_year:
        return DriftType.verified, 0.0
    return DriftType.citation_swap, 1.0


def check_comparison_flip(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    """Reversed direction triggers comparison_flip drift."""
    if src.normalized == tgt.normalized:
        return DriftType.verified, 0.0
    sd = src.metadata.get("direction")
    td = tgt.metadata.get("direction")
    if sd and td and sd != td and (
        (sd, td) in [(">", "<"), ("<", ">")]
    ):
        return DriftType.comparison_flip, 1.0
    return DriftType.partial, 0.5


def check_method_drift(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    """Same method name (case-insensitive) verified; else method_drift."""
    if src.normalized.lower() == tgt.normalized.lower():
        return DriftType.verified, 0.0
    return DriftType.method_drift, 1.0


def check_symbol_drift(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    if src.normalized == tgt.normalized:
        return DriftType.verified, 0.0
    return DriftType.symbol_drift, 1.0


_CHECK_FN = {
    ClaimType.numeric: check_numeric_drift,
    ClaimType.citation: check_citation_swap,
    ClaimType.comparison: check_comparison_flip,
    ClaimType.method_name: check_method_drift,
    ClaimType.symbol: check_symbol_drift,
}


def check_drift(src: ClaimRecord, tgt: ClaimRecord) -> tuple[DriftType, float]:
    """Dispatcher: run the type-specific drift check."""
    fn = _CHECK_FN.get(src.claim_type)
    if fn is None:
        return DriftType.partial, 0.5
    return fn(src, tgt)


# ────────────────────────── verbatim anchor preservation ──────────────────────────
#
# The re-extract-then-match verifier is brittle CROSS-LINGUALLY: a claim that was
# faithfully *translated* (e.g. "dialogue agents" → "对话智能体") re-extracts to a
# target-language surface that the English-normalized matcher cannot align, and is
# wrongly scored as `missing`/drift. But the whole point of a Fact-Anchor is that
# its *value* is language-invariant and should survive verbatim in the target.
# `anchor_preserved` checks exactly that, scoped to one unit's translation text,
# and is used as a lenient override: if the anchor is present, the claim is
# preserved regardless of what the re-extraction produced.

_DIGIT_CORE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_REF_NUMS_RE = re.compile(r"\d+")

# Spelled-out small numbers → their digit form (the common scientific cases).
_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _digit_anchors(surface: str) -> list[str]:
    """All digit cores in a surface, plus comma-stripped variants."""
    out: list[str] = []
    for m in _DIGIT_CORE_RE.findall(surface):
        out.append(m)
        if "," in m:
            out.append(m.replace(",", ""))
    return [a for a in out if a]


def anchor_preserved(
    claim: ClaimRecord,
    tgt_plain: str,
    glossary: dict[str, str] | None = None,
) -> bool:
    """True if the claim's language-invariant value survives in `tgt_plain`.

    For `method_name`, a claim also counts as preserved when its glossary target
    rendering appears (the named method was translated *consistently* per the
    locked glossary), not only when the English surface is kept verbatim.

    Returns False (no positive evidence) for genuinely language-dependent claims
    (most `comparison`), so the caller keeps the re-extraction verdict for those.
    """
    if not tgt_plain:
        return False
    text = tgt_plain
    ct = claim.claim_type

    if ct == ClaimType.numeric:
        anchors = _digit_anchors(claim.surface_form)
        if anchors:
            return any(a in text for a in anchors)
        # spelled-out number (e.g. "two"): accept if its digit form appears
        w = claim.surface_form.strip().lower()
        return _WORD_NUM.get(w) is not None and _WORD_NUM[w] in text

    if ct == ClaimType.citation:
        years = _YEAR_RE.findall(claim.surface_form)
        if years:
            return all(y in text for y in years)
        nums = _REF_NUMS_RE.findall(claim.surface_form)
        return bool(nums) and all(n in text for n in nums)

    if ct == ClaimType.symbol:
        norm = re.sub(r"\s+", "", claim.normalized or claim.surface_form)
        return bool(norm) and norm in re.sub(r"\s+", "", text)

    if ct == ClaimType.method_name:
        # method names are frequently kept in English; accept verbatim survival,
        # or the glossary's consistent target rendering of the same term.
        surf = claim.surface_form.strip()
        if len(surf) >= 3 and surf in text:
            return True
        if glossary:
            tgt_term = glossary.get(surf.lower())
            if tgt_term and tgt_term in text:
                return True
        return False

    return False


_NORMALIZE_FN = {
    ClaimType.numeric: normalize_numeric,
    ClaimType.citation: normalize_citation,
    ClaimType.comparison: normalize_comparison,
    ClaimType.method_name: normalize_method_name,
    ClaimType.symbol: normalize_symbol,
}


def normalize(claim_type: ClaimType, surface: str) -> tuple[str, dict[str, Any]]:
    fn = _NORMALIZE_FN.get(claim_type, normalize_method_name)
    return fn(surface)


# ────────────────────────── matching ──────────────────────────


def match_source_to_target(
    src_claims: list[ClaimRecord], tgt_claims: list[ClaimRecord]
) -> dict[int, ClaimRecord | None]:
    """Greedy matcher: for each src claim, find best target by type + normalized."""
    out: dict[int, ClaimRecord | None] = {}
    used: set[int] = set()
    for i, sc in enumerate(src_claims):
        candidates = [
            (j, tc) for j, tc in enumerate(tgt_claims)
            if j not in used and tc.claim_type == sc.claim_type
        ]
        # 1. exact normalized match
        match = next((j_t for j_t in candidates if j_t[1].normalized == sc.normalized), None)
        if match is None:
            # 2. for numeric, prefer same precision
            match = next(
                (j_t for j_t in candidates
                 if sc.claim_type == ClaimType.numeric
                 and j_t[1].metadata.get("precision") == sc.metadata.get("precision")),
                None,
            )
        if match is None and candidates:
            # 3. fall back to first candidate of same type (drift will flag)
            match = candidates[0]
        if match is not None:
            used.add(match[0])
            out[i] = match[1]
        else:
            out[i] = None
    return out
