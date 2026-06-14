"""Cross-lingual comparison-claim fidelity verifier (method contribution).

A `comparison` claim states a relation in prose — "A outperforms B by 2.3",
"X is lower than Y". When the sentence is translated, its surface changes
language, so the English-normalised matcher in `fact_anchors` cannot align it and
the re-extract verifier in FactVerify wrongly scores a *faithful* comparison as
`missing`. Substring/regex checks are therefore invalid here (the load-bearing
gap noted in CLAUDE.md).

This module verifies comparison fidelity **across languages** with a two-part
protocol that keeps the verdict auditable instead of trusting a yes/no judge:

  1. **Structured extraction** — the source comparison is parsed deterministically
     (`fact_anchors.normalize_comparison`); the target comparison is extracted by an
     LLM into a canonical tuple `(entity_a, entity_b, direction, dimension)` whose
     entities are *normalised back to English* so alignment happens in one space.
  2. **Cross-lingual entity alignment** — target entities are matched to the source
     baseline/subject via language-invariant anchors + the paper glossary; if the
     two entities are swapped, the extracted direction is flipped accordingly.
  3. **Direction-preservation verdict** — computed deterministically from the
     aligned directions: ``preserved`` / ``reversed`` / ``dropped`` / ``weakened``.

The LLM only does what it is good at (cross-lingual language understanding); the
*verdict logic* is plain code, so the method can be validated for precision/recall
against the synthetic `comparison_reversal` fact-traps (ground truth is known by
construction).
"""

from __future__ import annotations

from dataclasses import dataclass

from xinda.db.models import DriftType
from xinda.util import loads_dict
from xinda.logger_config import setup_logger
from xinda.translation import fact_anchors
from xinda.translation.prompts import STRICT_JSON_FOOTER, language_name

logger = setup_logger(__name__)

# Verdict taxonomy (the method's output classes).
PRESERVED = "preserved"
REVERSED = "reversed"
DROPPED = "dropped"
WEAKENED = "weakened"

# Verdict → DriftType, so FactVerify can persist it through the existing schema.
VERDICT_DRIFT: dict[str, DriftType] = {
    PRESERVED: DriftType.verified,
    REVERSED: DriftType.comparison_flip,
    DROPPED: DriftType.missing,
    WEAKENED: DriftType.partial,
}


@dataclass
class ComparisonTuple:
    entity_a: str  # the subject ("A"); may be empty if not stated in the surface
    entity_b: str  # the baseline ("B")
    direction: str  # ">" (A greater), "<" (A less), "=" (equal), "?" (unclear)
    dimension: str | None = None


# normalize_comparison may emit non-strict operators (">=", "<="); the verdict logic
# (`_FLIP`, the LLM enum) only handles the strict 4-symbol set, so collapse them.
_DIR_CANON = {">": ">", "<": "<", "=": "=", "?": "?", ">=": ">", "<=": "<", "==": "="}


def _canon_dir(direction: str | None) -> str:
    return _DIR_CANON.get((direction or "?").strip(), "?")


def parse_source_comparison(surface: str) -> ComparisonTuple:
    """Deterministic source-side tuple via the existing English normaliser.

    `normalize_comparison` yields the baseline (B) and the direction of A vs B;
    the subject (A) is frequently absent from the claim surface, so it may be "".
    """
    _norm, meta = fact_anchors.normalize_comparison(surface)
    return ComparisonTuple(
        entity_a="",
        entity_b=(meta.get("baseline") or "").strip(),
        direction=_canon_dir(meta.get("direction")),
    )


_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        # SOURCE comparison tuple (parsed from the English source phrasing)
        "source_a_en": {"type": "string"},
        "source_b_en": {"type": "string"},
        "source_direction": {"type": "string", "enum": [">", "<", "=", "?"]},
        # TARGET comparison tuple (entities normalised BACK TO ENGLISH so alignment
        # happens in one space)
        "target_has_comparison": {"type": "boolean"},
        "target_a_en": {"type": "string"},
        "target_b_en": {"type": "string"},
        "target_direction": {"type": "string", "enum": [">", "<", "=", "?"]},
        # true when the target states the relation only vaguely / drops direction
        "weakened": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "source_direction", "target_has_comparison",
        "target_a_en", "target_b_en", "target_direction", "weakened",
    ],
    "additionalProperties": False,
}


def _extract_system(target_name: str) -> str:
    return (
        f"You audit a comparison statement inside a {target_name} translation of an "
        "English scientific paper.\n"
        "You are given the SOURCE comparison (English) and the full TARGET translation. "
        "Extract BOTH as canonical tuples and report whether the target preserves the "
        "source comparison's direction.\n\n"
        "Direction is always about entity_a relative to entity_b:\n"
        "- '>' : A is greater / better / outperforms / higher than B\n"
        "- '<' : A is smaller / worse / lower than B\n"
        "- '=' : A equals B\n"
        "- '?' : direction not clearly stated\n\n"
        "Rules:\n"
        "- If an <AUDIT_THIS_COMPARISON> tag is given, audit ONLY that specific comparison "
        "(the same entities/quantity) on BOTH sides — ignore other comparisons in the text. "
        "This is critical when the passage contains several comparisons.\n"
        "- source_* describes the SOURCE comparison; find the CORRESPONDING comparison in "
        "the target for target_*.\n"
        "- Normalise ALL entities to ENGLISH (use the original English names; for translated "
        "common nouns give their English equivalent). Keep proper names / method names / "
        "symbols verbatim.\n"
        "- Report whatever direction each side ACTUALLY states, even if it looks wrong — you "
        "are auditing fidelity, NOT correcting the science.\n"
        "- target_has_comparison=false if the target has no corresponding comparison.\n"
        "- weakened=true if the target states the relation only vaguely (e.g. 'differs from', "
        "'is comparable to') or drops the magnitude/direction.\n\n"
        "Output a JSON object with fields source_a_en, source_b_en, source_direction, "
        "target_has_comparison, target_a_en, target_b_en, target_direction, weakened, "
        "reasoning.\n"
        + STRICT_JSON_FOOTER
    )


_STOPWORDS = frozenset(
    "the a an of on in to for by and or with at as is are be its their our this that "
    "obtained on combined experiment dataset data".split()
)


def _content_tokens(s: str) -> set[str]:
    return {t for t in s.lower().replace("-", " ").split() if t and t not in _STOPWORDS}


def _same_entity(a: str, b: str, glossary: dict[str, str] | None) -> bool:
    """Loose cross-lingual entity equality in English space (+ glossary bridge)."""
    a0, b0 = a.strip().lower(), b.strip().lower()
    if not a0 or not b0:
        return False
    if a0 == b0 or a0 in b0 or b0 in a0:
        return True
    if glossary:
        # glossary maps src(lower) → tgt; bridge either direction
        if glossary.get(a0, "").strip().lower() == b0:
            return True
        if glossary.get(b0, "").strip().lower() == a0:
            return True
    # paraphrase fallback: entity surfaces are often reworded across extraction/translation
    # ("LHCb experiment yields on Run1" vs "yields obtained by the LHCb experiment on the
    # Run1 dataset") — match on content-token overlap (Jaccard) so the swap is still detected.
    ta, tb = _content_tokens(a), _content_tokens(b)
    if ta and tb:
        inter = len(ta & tb)
        jac = inter / len(ta | tb)
        if jac >= 0.6 or inter >= 3:
            return True
    return False


_FLIP = {">": "<", "<": ">", "=": "=", "?": "?"}


def decide_verdict(
    src: ComparisonTuple,
    tgt: ComparisonTuple,
    *,
    has_comparison: bool,
    weakened: bool,
    glossary: dict[str, str] | None = None,
) -> str:
    """Deterministic verdict from aligned source/target tuples (the method core).

    Orientation is resolved by aligning the target's two entities to the source
    baseline (B): if the target's B matches the source B, orientation is kept; if
    the target's B matches the source A (entities swapped), the target direction is
    flipped before comparison.
    """
    if not has_comparison:
        return DROPPED
    if weakened or tgt.direction == "?":
        return WEAKENED

    eff_dir = tgt.direction
    if src.entity_b:
        tgt_b_is_src_b = _same_entity(tgt.entity_b, src.entity_b, glossary)
        tgt_a_is_src_b = _same_entity(tgt.entity_a, src.entity_b, glossary)
        if tgt_a_is_src_b and not tgt_b_is_src_b:
            # entities swapped relative to the source framing → flip
            eff_dir = _FLIP.get(tgt.direction, "?")
        elif not tgt_b_is_src_b and not tgt_a_is_src_b:
            # could not anchor either entity to the source baseline; fall back to
            # the raw (unswapped) direction comparison rather than guessing.
            pass

    if src.direction == "?":
        return WEAKENED
    if eff_dir == src.direction:
        return PRESERVED
    if eff_dir == _FLIP.get(src.direction, "?") and eff_dir != "?":
        return REVERSED
    return WEAKENED


def _parse_json(text: str) -> dict | None:
    return loads_dict(text)


async def verify(
    provider,
    *,
    source_comparison: str,
    target_text: str,
    language: str,
    glossary: dict[str, str] | None = None,
    focus: str | None = None,
) -> tuple[str, dict]:
    """Verify whether `target_text` preserves the comparison in `source_comparison`.

    Returns ``(verdict, details)`` where verdict ∈ {preserved, reversed, dropped,
    weakened}. `source_comparison` is the (possibly trapped) source surface whose
    direction is the reference to preserve.

    `focus` names the SPECIFIC comparison to audit. When the source/target are full
    paragraphs containing several comparisons, this stops the model from picking a
    different (more salient) comparison on each side — the wrong-comparison error that
    otherwise makes a preserved-direction read as a match. In the pipeline this is the
    `verifiable_claim` surface; in production the claim being checked is always known.
    """
    focus_line = (
        f"<AUDIT_THIS_COMPARISON>{focus}</AUDIT_THIS_COMPARISON>\n" if focus else ""
    )
    prompt = (
        f"{focus_line}"
        f"<SOURCE_COMPARISON>{source_comparison}</SOURCE_COMPARISON>\n"
        f"<TARGET_TRANSLATION>{target_text}</TARGET_TRANSLATION>"
    )
    if glossary:
        hits = "; ".join(f"{k} → {v}" for k, v in list(glossary.items())[:40])
        prompt += f"\n<GLOSSARY>{hits}</GLOSSARY>"

    try:
        tr = await provider.generate(
            prompt=prompt,
            system=_extract_system(language_name(language)),
            json_schema=_EXTRACT_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("comparison verify LLM call failed: %s", e)
        # No evidence either way → treat as weakened (neither pass nor flip).
        return WEAKENED, {"error": str(e)}

    obj = _parse_json(tr.text)
    if obj is None:
        return WEAKENED, {"error": "unparseable"}

    # Source tuple is LLM-extracted (robust); fall back to the regex parse only if
    # the model couldn't determine the source direction.
    src = ComparisonTuple(
        entity_a=(obj.get("source_a_en") or "").strip(),
        entity_b=(obj.get("source_b_en") or "").strip(),
        # DashScope doesn't strictly enforce the schema enum, so the model can still
        # emit ">=", "<=" etc. — canonicalize to the strict set the verdict logic needs.
        direction=_canon_dir(obj.get("source_direction")),
    )
    if src.direction == "?":
        rp = parse_source_comparison(source_comparison)
        if rp.direction != "?":
            src = rp
    tgt = ComparisonTuple(
        entity_a=(obj.get("target_a_en") or "").strip(),
        entity_b=(obj.get("target_b_en") or "").strip(),
        direction=_canon_dir(obj.get("target_direction")),
    )
    verdict = decide_verdict(
        src, tgt,
        has_comparison=bool(obj.get("target_has_comparison")),
        weakened=bool(obj.get("weakened")),
        glossary=glossary,
    )
    return verdict, {
        "source": src.__dict__,
        "target": tgt.__dict__,
        "has_comparison": bool(obj.get("target_has_comparison")),
        "weakened": bool(obj.get("weakened")),
        "reasoning": obj.get("reasoning"),
    }


async def verify_drift(
    provider,
    *,
    source_comparison: str,
    target_text: str,
    language: str,
    glossary: dict[str, str] | None = None,
) -> tuple[DriftType, float, str]:
    """FactVerify adapter: returns (DriftType, magnitude, verdict)."""
    verdict, _details = await verify(
        provider, source_comparison=source_comparison,
        target_text=target_text, language=language, glossary=glossary,
    )
    drift = VERDICT_DRIFT[verdict]
    magnitude = 0.0 if verdict == PRESERVED else (1.0 if verdict == REVERSED else 0.5)
    return drift, magnitude, verdict


async def is_faithful(
    provider,
    *,
    source_comparison: str,
    target_text: str,
    language: str,
    glossary: dict[str, str] | None = None,
) -> bool:
    """Fact-Trap adapter: True iff the (trapped) comparison was preserved."""
    verdict, _ = await verify(
        provider, source_comparison=source_comparison,
        target_text=target_text, language=language, glossary=glossary,
    )
    return verdict == PRESERVED
