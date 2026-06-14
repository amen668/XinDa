"""COMET-Kiwi reference-free quality estimation, per-unit scoring.

Uses Unbabel/wmt22-cometkiwi-da. Wrapped to run on CPU/GPU automatically.
Model is loaded lazily and cached at module level (one ~600MB download).
"""

from __future__ import annotations

from typing import Iterable

from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

_COMET_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
_model = None  # lazy-loaded singleton


def _get_model():
    global _model
    if _model is None:
        from comet import download_model, load_from_checkpoint  # noqa: PLC0415
        logger.info("loading COMET model %s (first call may download)", _COMET_MODEL_NAME)
        path = download_model(_COMET_MODEL_NAME)
        _model = load_from_checkpoint(path)
    return _model


def score_pairs(
    pairs: Iterable[tuple[str, str]],
    batch_size: int = 8,
    gpus: int = 0,
) -> list[float]:
    """Score (src, mt) pairs. Returns per-pair scores in input order.

    Empty mt/src produces 0.0 without invoking the model.
    """
    pairs = list(pairs)
    if not pairs:
        return []
    data = [{"src": s, "mt": t, "ref": ""} for s, t in pairs]
    model = _get_model()
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(out.get("scores", []))
