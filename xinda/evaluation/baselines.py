"""External / naive baselines for the differentiation comparison.

The headline contrast for the paper is **structure preservation on full text**:
our pipeline keeps every inline math/citation/reference (serialised as
``{{PT_<TAG>_<n>}}`` placeholders) intact, whereas a translator without the
placeholder contract corrupts them — quantified here as a placeholder/math
**preservation rate** over the same source units.

Baselines (kept ISOLATED from the core pipeline — used only by the comparison CLI):

- ``naive_llm``  — the SAME model with a plain translate prompt that does NOT
  mention the placeholders. Primary baseline: holds the model constant, so the
  gap is attributable to the pipeline/contract, and it is fully reachable.
- ``google``     — real-world competitor via the free web endpoint. Best-effort:
  the free endpoint frequently returns HTTP 429, so this may be unavailable
  (returns None) without a paid key; the study should not depend on it.
- ``abstract_only`` — coverage baseline: only the abstract is translated, so the
  contrast is the **coverage** dimension (full-text vs abstract-only), the status
  quo of journal multilingual publishing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from xinda.logger_config import setup_logger
from xinda.translation.prompts import language_name

logger = setup_logger(__name__)

# Only the PT_ placeholders carry inline structure (math/cite/ref/…); {{NL}} etc.
# are whitespace and excluded from the structure metric.
PLACEHOLDER_RE = re.compile(r"\{\{PT_[A-Za-z]+_\d+\}\}")
MATH_TOKEN_RE = re.compile(r"\{\{PT_(?:Math|XMath|equation)_\d+\}\}")

# Google free web endpoint locale codes for our targets.
_GOOGLE_LANG = {"zh": "zh-CN", "fr": "fr", "es": "es", "de": "de", "ja": "ja"}


# ───────────────────────── structure preservation ─────────────────────────


def preservation_stats(src_text: str, tgt_text: str | None) -> dict[str, int]:
    """Count how many of src's placeholders/math-tokens survive verbatim in tgt."""
    tgt = tgt_text or ""
    ph = PLACEHOLDER_RE.findall(src_text or "")
    math = MATH_TOKEN_RE.findall(src_text or "")
    ph_kept = sum(1 for t in ph if t in tgt)
    math_kept = sum(1 for t in math if t in tgt)
    return {
        "ph_total": len(ph), "ph_kept": ph_kept,
        "math_total": len(math), "math_kept": math_kept,
    }


def aggregate_preservation(pairs: list[tuple[str, str | None]]) -> dict[str, float]:
    """Aggregate placeholder/math preservation rate over many (src, tgt) units."""
    pt = pk = mt = mk = 0
    for src, tgt in pairs:
        s = preservation_stats(src, tgt)
        pt += s["ph_total"]; pk += s["ph_kept"]
        mt += s["math_total"]; mk += s["math_kept"]
    return {
        "placeholder_total": pt,
        "placeholder_preserved": pk,
        "placeholder_rate": (pk / pt) if pt else 1.0,
        "math_total": mt,
        "math_preserved": mk,
        "math_rate": (mk / mt) if mt else 1.0,
    }


# ───────────────────────── naive LLM baseline ─────────────────────────


def naive_system_prompt(language: str) -> str:
    """A plain translate prompt with NO placeholder-preservation contract."""
    return (
        f"Translate the following text into {language_name(language)}. "
        "Output ONLY the translation, with no explanation."
    )


async def naive_translate(provider, text: str, language: str):
    """Translate one unit with the naive prompt. Returns the TranslationResult
    (so the caller can read .text and token counts for cost)."""
    return await provider.generate(prompt=text or "", system=naive_system_prompt(language))


# ───────────────────────── google (best-effort) ─────────────────────────


def google_translate(text: str, language: str, *, retries: int = 3) -> str | None:
    """Free web endpoint; returns None if unavailable (e.g. persistent 429)."""
    tl = _GOOGLE_LANG.get(language, language)
    out: list[str] = []
    for chunk in _chunks(text or "", 4000):
        piece = _google_chunk(chunk, tl, retries)
        if piece is None:
            return None  # endpoint unavailable → caller marks google as N/A
        out.append(piece)
        time.sleep(0.5)  # be polite to the free endpoint
    return "".join(out)


def _chunks(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i:i + n]


# `translate.googleapis.com` 429s from some networks (incl. CN); `translate.google.com`
# and `clients5.google.com` are reachable — try them in order.
_GOOGLE_HOSTS = (
    "https://translate.google.com/translate_a/single",
    "https://translate.googleapis.com/translate_a/single",
)
_GOOGLE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://translate.google.com/",
}


def _google_chunk(text: str, tl: str, retries: int) -> str | None:
    for attempt in range(retries):
        host = _GOOGLE_HOSTS[attempt % len(_GOOGLE_HOSTS)]
        url = host + "?" + urllib.parse.urlencode(
            {"client": "gtx", "sl": "en", "tl": tl, "dt": "t", "q": text}
        )
        try:
            req = urllib.request.Request(url, headers=_GOOGLE_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            return "".join(seg[0] for seg in data[0] if seg and seg[0])
        except Exception as e:  # noqa: BLE001
            logger.debug("google chunk attempt %d (%s) failed: %s", attempt, host, e)
            time.sleep(1.5 * (attempt + 1))
    return None


# ───────────────────────── abstract-only coverage ─────────────────────────


def coverage_fraction(abstract_chars: int, total_chars: int) -> float:
    """Fraction of the document an abstract-only baseline would translate."""
    return (abstract_chars / total_chars) if total_chars else 0.0
