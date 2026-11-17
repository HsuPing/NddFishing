"""Owns the running frame source and the hand-off to the UI thread."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from wowfishing.capture.base import AdjustableFrameSource, CaptureError
from wowfishing.capture.recorder import FrameRecorder
from wowfishing.capture.targets import CaptureTarget
from wowfishing.models import Frame, Roi
from wowfishing.utils.latest_slot import LatestSlot
from wowfishing.utils.performance import RateCounter, RollingStats

log = logging.getLogger(__name__)

SourceFactory = Callable[[CaptureTarget, int, Roi], AdjustableFrameSource]
"""Builds a source for ``(target, fps, roi)``."""


class CaptureController(QObject):
    """Starts/stops capture and exposes the newest frame plus statistics.

    Capture callbacks run on the source's thread and only touch thread-safe
    members; the UI polls :attr:`frames` from its own timer. Failures arrive on
    the UI thread through the queued :attr:`failed` signal.
    """

    failed = Signal(str)

    def __init__(self, source_factory: SourceFactory, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._factory = source_factory
        self._source: AdjustableFrameSource | None = None
        self.frames: LatestSlot[Frame] = LatestSlot()
        self.fps = RateCounter()
        self.latency_ms = RollingStats()
        self._recorder: FrameRecorder | None = None

    @property
    def running(self) -> bool:
        return self._source is not None

    def start(self, target: CaptureTarget, fps: int, roi: Roi) -> None:
        """Start capturing. Raises ``CaptureError``; the controller then stays stopped."""
        self.stop()
        self.fps.reset()
        self.latency_ms.reset()
        self.frames.reset()
        source = self._factory(target, fps, roi)
        try:
            source.start(self._on_frame, self._on_error)
        except CaptureError:
            self.frames.clear()  # frames may have arrived before the failure
            raise
        self._source = source

    def stop(self) -> None:
        self.stop_recording()
        source, self._source = self._source, None
        if source is not None:
            source.stop()
        self.frames.clear()

    def set_roi(self, roi: Roi) -> None:
        """Apply ``roi`` to the running source. Raises ``CaptureError`` if rejected."""
        if self._source is not None:
            self._source.set_roi(roi)

    # --- recording ------------------------------------------------------------

    @property
    def recorder(self) -> FrameRecorder | None:
        """The active recorder (for progress read-outs), or ``None``."""
        return self._recorder

    def start_recording(self, session_dir: Path, meta: dict[str, Any]) -> None:
        """Record every delivered frame into ``session_dir``. Raises ``OSError``/``ValueError``."""
        self.stop_recording()
        recorder = FrameRecorder(session_dir)
        recorder.start(meta)
        self._recorder = recorder

    def stop_recording(self) -> FrameRecorder | None:
        """Flush and close the recording, blocking until it is on disk (shutdown path)."""
        recorder = self.detach_recorder()
        if recorder is not None:
            recorder.stop()
        return recorder

    def detach_recorder(self) -> FrameRecorder | None:
        """Stop feeding the recorder and let it finish in the background; never blocks.

        The caller watches ``recorder.finished`` and reports the totals afterwards.
        """
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            recorder.request_stop()
        return recorder

    # --- capture thread -------------------------------------------------------

    def _on_frame(self, frame: Frame) -> None:
        # The timestamp is the frame's display time, which can be a hair in the future.
        self.latency_ms.add(max(0.0, time.monotonic() - frame.timestamp) * 1000.0)
        self.fps.tick()
        recorder = self._recorder  # single read: the UI thread may swap it
        if recorder is not None:
            recorder.write(frame)
        self.frames.put(frame)

    def _on_error(self, error: Exception) -> None:
        log.error("capture failed: %s", error)
        self.failed.emit(str(error))  # queued to the UI thread
