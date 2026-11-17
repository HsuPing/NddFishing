"""Classifier interface. Feature extraction and concrete classifiers arrive in Phase 4.

The classifier consumes raw audio chunks rather than a fixed feature struct so a
future ML model can choose its own features; the rule-based classifier will use
the shared ``feature_extractor`` module internally.
"""

from __future__ import annotations

from typing import Protocol

from wowfishing.models import AudioChunk, AudioEvent


class EventClassifier(Protocol):
    """Streaming sound-event classifier.

    Fed consecutive chunks in order from one worker thread; keeps whatever
    history it needs (onset detection needs context) until ``reset``.
    """

    def process(self, chunk: AudioChunk) -> list[AudioEvent]:
        """Return the events that completed within this chunk (usually empty)."""
        ...

    def reset(self) -> None: ...
