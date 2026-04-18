from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "info") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    for noisy in ["uvicorn.access"]:
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
