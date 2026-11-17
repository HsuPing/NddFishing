"""Detector interface. Concrete detectors (classic OpenCV first) arrive in Phase 3."""

from __future__ import annotations

from typing import Protocol

from wowfishing.models import Detection, Frame


class BobberDetector(Protocol):
    """Finds the fishing bobber in a single frame.

    Implementations are called from one worker thread at a time and may keep
    internal state (e.g. a background model); ``reset`` clears it, for example
    when the ROI changes or a new cast begins.
    """

    def detect(self, frame: Frame) -> Detection | None:
        """Return the best candidate, or ``None`` when no bobber is visible."""
        ...

    def reset(self) -> None: ...
