"""System + user prompt construction.

Prompt structure designed for DashScope Context Cache:
- STABLE PREFIX (cached, ~3-5k tokens): system + paper meta + glossary +
  section outline. Same across all paragraph translations of one paper.
- VARIABLE SUFFIX (per call): section heading, prev paragraph, fact anchors,
  input.
"""

from __future__ import annotations

import json
from typing import Any

# Qwen3 outputs tend to verbose preamble. Footer forces strict JSON.
# English meta-instruction so it never leaks target-language text into es/fr output.
STRICT_JSON_FOOTER = (
    "IMPORTANT: Output JSON directly — no preamble, no markdown fences, no trailing "
    "comments. Any non-JSON content will break downstream parsing."
)

# Faithful single-paragraph translation (direction-preserving) — used by the eval
# harnesses (`cli.eval_comparison_verifier`, `cli.fidelity_vs_qa`) to build ground-truth
# translations whose comparison direction must reflect the input verbatim. `{lang}`.
FAITHFUL_TRANSLATE_SYSTEM = (
    "Translate the English scientific-paper paragraph into {lang}. Output ONLY the "
    "translation. Preserve every fact exactly — especially any comparison's direction "
    "(A vs B). Even if a comparison looks wrong, keep its direction; do NOT 'correct' it."
)

# JSON schema for batched translation response: list of [id, translation] arrays.
TRANSLATION_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer"},
                    {"type": "string"},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


# Canonical ISO code → English language name. The single source of truth for
# language naming across the whole framework (import this; do not re-declare).
# Generic by design: any unlisted code falls back to itself via language_name(),
# so a new target language needs no code change here.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
    "ar": "Arabic",
}


def language_name(code: str) -> str:
    """English name for an ISO language code; falls back to the code itself."""
    return LANGUAGE_NAMES.get(code, code)


def stable_prefix(
    *,
    paper_title: str,
    arxiv_id: str,
    field: str,
    target_language: str,
    glossary_terms: list[dict] | None = None,
    section_outline: list[str] | None = None,
    abstract: str | None = None,
) -> str:
    """Cacheable per-paper prefix.

    All inputs here are stable across paragraphs, so DashScope's prompt
    cache can serve this portion at 10% price after the first call.
    Minimum 1024 tokens; with glossary + outline + abstract we typically
    hit 3000-5000 tokens.
    """
    target_name = language_name(target_language)
    lines: list[str] = [
        "<SYSTEM>",
        "You are a professional, cross-disciplinary academic-paper translator.",
        f'Current paper: "{paper_title}" (arxivId: {arxiv_id}), field: {field}.',
        f"Translate the source paper fragments into {target_name}.",
        "",
        "Translation requirements:",
        "1. Every placeholder of the form {{PT_XXX_N}}, {{NL}}, {{TAB}}, {{RE}} MUST be kept "
        "verbatim, unchanged.",
        "2. Use each discipline's standard academic terminology; stay precise and professional.",
        "3. Preserve the source's logical structure and academic rigor.",
        "4. Numbers, citations, method names and symbols must be preserved completely and "
        "equivalently:",
        "   - Do not change numeric precision (93.4% must not become ≈93%).",
        "   - Do not change the cited object (Vaswani 2017 must not become Vaswani 2018).",
        "   - Do not reverse comparison direction (A outperforms B must not become B outperforms A).",
        "5. Locked terms must appear verbatim in the target form specified in GLOSSARY.",
        f"6. Match the academic-journal writing conventions of {target_name}.",
        "",
        "Output format: a JSON object with field `translations`, a 2-D array "
        "[[id, translation], ...].",
        STRICT_JSON_FOOTER,
        "</SYSTEM>",
    ]

    if abstract:
        lines += ["", "<PAPER_ABSTRACT>", abstract.strip()[:2000], "</PAPER_ABSTRACT>"]

    if section_outline:
        lines += ["", "<SECTION_OUTLINE>"]
        for s in section_outline:
            lines.append(f"- {s}")
        lines += ["</SECTION_OUTLINE>"]

    if glossary_terms:
        lines += ["", "<GLOSSARY_FULL>"]
        for t in glossary_terms:
            lock = "[locked]" if t.get("locked") else ""
            kind = t.get("kind", "")
            defn = t.get("definition")
            line = f"- {t['src']} → {t['tgt']} {lock} ({kind})"
            if defn:
                line += f" // {defn}"
            lines.append(line)
        lines += ["</GLOSSARY_FULL>"]

    return "\n".join(lines)


def variable_suffix(
    items: list[dict[str, Any]],
    *,
    section_heading: str | None = None,
    prev_translation: str | None = None,
    next_source: str | None = None,
    glossary_hits: list[dict] | None = None,
    fact_anchors: dict[str, list[str]] | None = None,
) -> str:
    """Per-batch variable suffix (uncached portion).

    `fact_anchors` is a dict like {"numeric": ["93.4%", "100k"],
    "citation": ["Vaswani et al. (2017)"], ...}.

    `prev_translation` is the already-produced translation of the preceding
    span (backward coherence); `next_source` is the *source* of the upcoming
    span (forward coherence — lets the model resolve cataphora / terms that are
    only fully introduced later). Both are read-only context: ONLY <INPUT> is
    to be translated.
    """
    parts: list[str] = []

    if section_heading:
        parts += [f"<SECTION>{section_heading}</SECTION>"]

    if prev_translation:
        # backward context: prior translation (cap to stay token-light)
        snippet = prev_translation.strip()[:1200]
        parts += [
            f"<PREV_TRANSLATION>{snippet}</PREV_TRANSLATION>",
            "(Already-finalized translation above — for style/terminology/reference "
            "consistency only; do NOT re-translate it.)",
        ]

    if next_source:
        # forward context: upcoming SOURCE (cap; this is English, not yet translated)
        snippet = next_source.strip()[:800]
        parts += [
            f"<NEXT_SOURCE>{snippet}</NEXT_SOURCE>",
            "(Upcoming SOURCE below — only to understand forward dependencies/terms; "
            "do NOT translate it.)",
        ]

    if glossary_hits:
        lines = ["<GLOSSARY_HITS>"]
        for t in glossary_hits:
            lock = "[locked]" if t.get("locked") else ""
            lines.append(f"- {t['src']} → {t['tgt']} {lock}")
        lines.append("</GLOSSARY_HITS>")
        parts.append("\n".join(lines))

    if fact_anchors:
        non_empty = {k: v for k, v in fact_anchors.items() if v}
        if non_empty:
            lines = ["<FACT_ANCHORS>"]
            for kind, vals in non_empty.items():
                lines.append(f"  {kind}: {json.dumps(vals, ensure_ascii=False)}")
            lines.append("</FACT_ANCHORS>")
            lines.append(
                "HARD CONSTRAINT: every fact in FACT_ANCHORS must be preserved equivalently "
                "(or localized) in the translation — do not alter numeric precision, do not "
                "change the cited object, do not reverse comparison direction."
            )
            parts.append("\n".join(lines))

    slim = [[it["id"], it["src_text"]] for it in items]
    parts.append(
        "Translate every source item in the following [id, source] array:\n"
        f"<INPUT>{json.dumps(slim, ensure_ascii=False)}</INPUT>"
    )

    return "\n\n".join(parts)


# ────────────────────── legacy helpers (used by M2 first-pass code) ──────────


def system_prompt_translate(
    *,
    paper_title: str,
    arxiv_id: str,
    field: str,
    target_language: str,
) -> str:
    """Backwards-compatible wrapper used by M2 FirstPassTranslate."""
    return stable_prefix(
        paper_title=paper_title, arxiv_id=arxiv_id, field=field,
        target_language=target_language,
    )


def user_prompt_translate_batch(items: list[dict[str, Any]]) -> str:
    """Backwards-compatible wrapper used by M2 FirstPassTranslate."""
    return variable_suffix(items)


# ─────────── Refine prompt (M7) ───────────


def refine_user_prompt(
    *,
    src_text: str,
    draft_text: str,
    failed_claims: list[dict] | None = None,
    cross_doc_issues: list[str] | None = None,
    section_heading: str | None = None,
    prev_translation: str | None = None,
    glossary_hits: list[dict] | None = None,
) -> str:
    """Prompt body for refinement pass (after first translation failed gates)."""
    parts: list[str] = [
        "<TASK>The previous translation failed the quality/fidelity gate. "
        "Re-translate this span.</TASK>"
    ]
    if section_heading:
        parts.append(f"<SECTION>{section_heading}</SECTION>")
    if prev_translation:
        parts.append(f"<PREV>{prev_translation.strip()[:400]}</PREV>")
    if glossary_hits:
        lines = ["<GLOSSARY_HITS>"]
        for t in glossary_hits:
            lock = "[locked]" if t.get("locked") else ""
            lines.append(f"- {t['src']} → {t['tgt']} {lock}")
        lines.append("</GLOSSARY_HITS>")
        parts.append("\n".join(lines))
    parts.append(f"<SOURCE>{src_text}</SOURCE>")
    parts.append(f"<DRAFT>{draft_text}</DRAFT>")
    if failed_claims:
        lines = ["<FAILED_CLAIMS>"]
        for fc in failed_claims:
            lines.append(
                f"- {fc.get('drift', 'unknown')}: "
                f"src '{fc.get('src_surface', '')}' → "
                f"tgt '{fc.get('tgt_surface', '')}'"
            )
        lines.append("</FAILED_CLAIMS>")
        lines.append("You MUST fix every factual claim listed above.")
        parts.append("\n".join(lines))
    if cross_doc_issues:
        lines = ["<CROSS_DOC_ISSUES>"]
        for issue in cross_doc_issues:
            lines.append(f"- {issue}")
        lines.append("</CROSS_DOC_ISSUES>")
        parts.append("\n".join(lines))
    parts.append(
        "Output a JSON object with field `translation` whose value is the improved "
        f"translation string.\n{STRICT_JSON_FOOTER}"
    )
    return "\n\n".join(parts)


REFINE_SCHEMA = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}
