"""Logging setup. Modules obtain loggers with ``logging.getLogger(__name__)``."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(threadName)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the package logger once; safe to call again to change the level."""
    root = logging.getLogger("wowfishing")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
