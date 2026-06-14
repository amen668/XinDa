"""Small shared helpers used across CLIs / evaluation / pipeline stages.

Consolidates logic that was copy-pasted in several places (notably the lenient
JSON parsing of LLM responses, which appeared in 5 near-identical forms).
"""

from __future__ import annotations

import json
from typing import Any


def loads_lenient(text: str | None) -> Any | None:
    """Parse JSON from an LLM response, tolerating a ```json … ``` code fence.

    Returns the parsed object (dict/list/…) or None if it cannot be parsed.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def loads_dict(text: str | None) -> dict | None:
    """`loads_lenient` constrained to a JSON object (else None)."""
    obj = loads_lenient(text)
    return obj if isinstance(obj, dict) else None


def parse_translation_array(text: str | None) -> list | None:
    """A batch-translation response: a JSON array, or {"translations": [...]}."""
    obj = loads_lenient(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "translations" in obj:
        return obj["translations"]
    return None
