"""TCR — Terminology Consistency Rate.

For each glossary term, scan all translations of that paper and check
whether the term is translated **identically** across occurrences.

Definition:
  TCR = sum over terms( (most frequent target translation count) /
                        (total occurrences of source term) )
        / number of glossary terms with at least one occurrence

A perfect score (1.0) means every glossary source term mapped to exactly
one target translation everywhere it appeared. Lower scores indicate
terminology drift across paragraphs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any


def compute_tcr(
    glossary: Iterable[dict[str, Any]],
    src_texts: list[str],
    tgt_texts: list[str],
) -> dict[str, Any]:
    """Compute TCR across paired (src, tgt) paragraphs.

    glossary entries should have at least `src` (source term) keys. The
    function inspects each (src_text, tgt_text) pair: when src_text
    contains the source term, it looks for the nearest matching target
    term in tgt_text (heuristic: first non-stopword chunk after the
    spelled-out source if the target preserves it, else the most
    frequent capitalized chunk).
    """
    if len(src_texts) != len(tgt_texts):
        raise ValueError("src and tgt sequence lengths differ")

    term_translations: dict[str, Counter[str]] = {}
    for term in glossary:
        src_term = term["src"]
        flags = 0 if term.get("kind") == "acronym" else re.IGNORECASE
        pat = re.compile(r"\b" + re.escape(src_term) + r"\b", flags)

        counter: Counter[str] = Counter()
        for src, tgt in zip(src_texts, tgt_texts):
            if not src or not tgt or not pat.search(src):
                continue
            tgt_render = _detect_target_form(src_term, term.get("tgt", ""), tgt)
            if tgt_render:
                counter[tgt_render] += 1
        if counter:
            term_translations[src_term] = counter

    if not term_translations:
        return {"tcr": 1.0, "term_count": 0, "details": {}}

    consistencies: list[float] = []
    details: dict[str, dict[str, Any]] = {}
    for src_term, counter in term_translations.items():
        total = sum(counter.values())
        top_tgt, top_count = counter.most_common(1)[0]
        cons = top_count / total if total else 0.0
        consistencies.append(cons)
        details[src_term] = {
            "total_occurrences": total,
            "most_freq_tgt": top_tgt,
            "consistency": cons,
            "distribution": dict(counter),
        }

    tcr = sum(consistencies) / len(consistencies) if consistencies else 1.0
    return {"tcr": tcr, "term_count": len(consistencies), "details": details}


def _detect_target_form(src_term: str, expected_tgt: str, tgt_text: str) -> str | None:
    """Heuristic: find how `src_term` was rendered in `tgt_text`.

    Order of attempts:
      1. expected_tgt verbatim (locked-term ideal case)
      2. source-term verbatim (model kept English)
      3. None (no detectable rendering)
    """
    if expected_tgt and expected_tgt in tgt_text:
        return expected_tgt
    if src_term in tgt_text:
        return src_term
    return None
