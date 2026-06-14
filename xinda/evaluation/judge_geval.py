"""G-Eval 4-dim Likert judge (Liu et al., EMNLP 2023).

Chain-of-thought reasoning followed by 1-5 score on each of:
  fluency / adequacy / terminology / structure

Secondary judge for inter-judge agreement testing in meta-eval.
"""

from __future__ import annotations

import json

from xinda.config import settings
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER

logger = setup_logger(__name__)


SYSTEM_PROMPT = (
    "You are evaluating a scientific-paper translation on these 4 dimensions (each 1-5):\n"
    "- fluency: fluency in the target language\n"
    "- adequacy: adequacy of meaning transfer\n"
    "- terminology: terminology consistency and correctness\n"
    "- structure: discourse logic\n\n"
    "For each dimension, reason in 1-2 sentences, then give a 1-5 score.\n\n"
    "Output a JSON object with fields `fluency`/`adequacy`/`terminology`/`structure`, "
    "each being {reasoning, score}.\n"
    + STRICT_JSON_FOOTER
)


SCHEMA = {
    "type": "object",
    "properties": {
        dim: {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        } for dim in ("fluency", "adequacy", "terminology", "structure")
    },
    "required": ["fluency", "adequacy", "terminology", "structure"],
    "additionalProperties": False,
}


async def judge_one(
    src_plain: str,
    tgt_plain: str,
    *,
    section_heading: str | None = None,
    field: str | None = None,
) -> dict:
    provider = create_provider(settings.model_judge_geval)
    prompt = (
        f"<DOMAIN>scientific paper, field={field or 'Other'}</DOMAIN>\n"
        + (f"<CONTEXT>section={section_heading}</CONTEXT>\n" if section_heading else "")
        + f"<SOURCE>{src_plain}</SOURCE>\n"
        + f"<TRANSLATION>{tgt_plain}</TRANSLATION>"
    )
    try:
        tr = await provider.generate(
            prompt=prompt, system=SYSTEM_PROMPT, json_schema=SCHEMA,
        )
    except Exception as e:
        logger.warning("g-eval failed: %s", e)
        return {"valid": False}

    s = (tr.text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {"valid": False}
    return {
        "valid": True,
        "fluency": (obj.get("fluency") or {}).get("score"),
        "adequacy": (obj.get("adequacy") or {}).get("score"),
        "terminology": (obj.get("terminology") or {}).get("score"),
        "structure": (obj.get("structure") or {}).get("score"),
        "raw": obj,
    }
