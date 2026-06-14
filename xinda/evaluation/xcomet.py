"""xCOMET-XXL — reference-free QE + error span detection (2024 SOTA).

Local model, no API cost. Returned scores are headline metrics in plan v3.
The XL variant is used by default to fit on a single 24GB GPU; switch to
XXL via env var if you have an 80GB card.
"""

from __future__ import annotations

import os
from typing import Iterable

from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

_XCOMET_MODEL_NAME = os.environ.get("XCOMET_MODEL", "Unbabel/XCOMET-XL")
_model = None


def _get_model():
    global _model
    if _model is None:
        from comet import download_model, load_from_checkpoint  # noqa: PLC0415
        logger.info("loading xCOMET model %s (first call may download)", _XCOMET_MODEL_NAME)
        path = download_model(_XCOMET_MODEL_NAME)
        _model = load_from_checkpoint(path)
    return _model


def score_pairs(
    pairs: Iterable[tuple[str, str]],
    batch_size: int = 8,
    gpus: int = 0,
) -> list[dict]:
    """Score (src, mt) pairs.

    Returns list of dicts: {"score": float, "errors": [error spans...]}
    (xCOMET produces both an aggregate score AND span-level error tags).
    """
    pairs = list(pairs)
    if not pairs:
        return []
    data = [{"src": s, "mt": t, "ref": ""} for s, t in pairs]
    model = _get_model()
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = out.get("scores", [])
    errors = out.get("metadata", {}).get("error_spans", [[]] * len(scores))
    return [{"score": s, "errors": e} for s, e in zip(scores, errors)]


def score_only(pairs: Iterable[tuple[str, str]], **kwargs) -> list[float]:
    """Convenience: just the float scores."""
    return [r["score"] for r in score_pairs(pairs, **kwargs)]
