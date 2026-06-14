"""RUBRIC-MQM v2 judge (Kocmi & Federmann updated 2025, ACL Industry).

Produces per-rubric scores + an aggregated weighted score. Multi-run
aggregation (median of 3+ runs) is recommended in the original paper to
combat single-judgment noise.
"""

from __future__ import annotations

import asyncio
import json
import statistics

from xinda.config import settings
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER

logger = setup_logger(__name__)


RUBRIC = [
    ("accuracy_factual",  0.30, "numbers / citations / method names are accurate"),
    ("accuracy_semantic", 0.20, "meaning is conveyed accurately"),
    ("fluency",           0.15, "fluency in the target language"),
    ("terminology",       0.15, "terminology consistency and correctness"),
    ("structure",         0.10, "discourse logic / paragraph coherence"),
    ("comprehensibility", 0.10, "a target-language reader can understand it"),
]


def _system_prompt() -> str:
    rubric_lines = "\n".join(
        f"{i+1}. {key} (weight {w:.2f}): {desc}" for i, (key, w, desc) in enumerate(RUBRIC)
    )
    return (
        "Evaluate a scientific-paper translation against the rubric below, scoring each item "
        "1-5, then aggregate to a weighted total.\n\n"
        "RUBRIC:\n" + rubric_lines + "\n\n"
        "Give each item a 1-5 score with a brief explanation, then output rubric_score "
        "(weighted total, in the 1-5 range).\n"
        "total = sum(score_i * weight_i).\n\n"
        "Output a JSON object with field `scores` (per-item 1-5 mapping) and `rubric_score` "
        "(weighted total).\n"
        + STRICT_JSON_FOOTER
    )


_RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {key: {"type": "number"} for key, _, _ in RUBRIC},
            "required": [key for key, _, _ in RUBRIC],
            "additionalProperties": False,
        },
        "rubric_score": {"type": "number"},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                    "span_text": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["category", "severity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores", "rubric_score"],
    "additionalProperties": False,
}


async def judge_one(
    src_plain: str,
    tgt_plain: str,
    *,
    section_heading: str | None = None,
    field: str | None = None,
    runs: int = 3,
) -> dict:
    """Run RUBRIC-MQM `runs` times, return aggregated dict.

    Aggregation: median across runs for each numeric field. The raw
    per-run responses are returned under `raw_runs` for audit.
    """
    provider = create_provider(settings.model_judge_rubric)
    prompt = (
        f"<DOMAIN>scientific paper, field={field or 'Other'}</DOMAIN>\n"
        + (f"<CONTEXT>section={section_heading}</CONTEXT>\n" if section_heading else "")
        + f"<SOURCE>{src_plain}</SOURCE>\n"
        + f"<TRANSLATION>{tgt_plain}</TRANSLATION>"
    )
    system = _system_prompt()

    async def one_run() -> dict | None:
        try:
            tr = await provider.generate(
                prompt=prompt, system=system, json_schema=_RUBRIC_SCHEMA,
            )
        except Exception as e:
            logger.warning("rubric judge run failed: %s", e)
            return None
        s = (tr.text or "").strip()
        if s.startswith("```"):
            s = s.strip("`").lstrip("json").strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    raw = await asyncio.gather(*(one_run() for _ in range(runs)))
    valid = [r for r in raw if r and "scores" in r]
    if not valid:
        return {"valid": False, "raw_runs": raw}

    medians: dict[str, float] = {}
    for key, _, _ in RUBRIC:
        vals = [r["scores"].get(key) for r in valid if r["scores"].get(key) is not None]
        if vals:
            medians[key] = statistics.median(vals)

    rubric_scores = [r.get("rubric_score") for r in valid if r.get("rubric_score") is not None]
    rubric_median = statistics.median(rubric_scores) if rubric_scores else None

    all_errors = []
    for r in valid:
        all_errors.extend(r.get("errors") or [])

    return {
        "valid": True,
        "scores": medians,
        "rubric_score": rubric_median,
        "errors": all_errors,
        "raw_runs": raw,
    }
