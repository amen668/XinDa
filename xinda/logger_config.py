"""Module-level loggers with rotating-file handler in logs/."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str | None = None,
    log_file: str | Path = "logs/translation.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
