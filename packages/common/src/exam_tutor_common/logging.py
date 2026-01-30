from __future__ import annotations

import logging
import os
from pythonjsonlogger.json import JsonFormatter


def configure_logging() -> None:
    """Configure structured JSON logging for the process."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger()
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    )
    handler.setFormatter(formatter)

    # Replace existing handlers (avoid duplicate logs)
    logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
