"""Record captured frames to disk as a lossless image sequence.

Session layout::

    <session>/meta.json        what was captured and when
    <session>/frames.jsonl     one line per frame: index, timestamp, roi, scale, file
    <session>/frames/000000.png

PNG (not video) so pixel values are exactly what the detector saw; lossy video
would shift the colours the detector keys on. Timestamps are the frames' own
monotonic timestamps, so an audio recording of the same session lines up.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from wowfishing.models import Frame

log = logging.getLogger(__name__)

INDEX_FILE = "frames.jsonl"
META_FILE = "meta.json"
FRAMES_DIR = "frames"

_PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # speed over size; still lossless


class FrameRecorder:
    """Writes frames on its own thread so the capture thread never waits for the disk.

    ``write`` only enqueues. If the disk cannot keep up the queue fills and
    further frames are dropped (and counted) rather than delaying capture.
    """

    def __init__(self, session_dir: Path, queue_size: int = 64) -> None:
        self.session_dir = session_dir
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._accepting = False
        self._lock = threading.Lock()  # guards the counters and error
        self._written = 0
        self._dropped = 0
        self._error: Exception | None = None

    def start(self, meta: dict[str, Any]) -> None:
        """Create the session directory and begin accepting frames. Raises ``OSError``."""
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        (self.session_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=False)
        payload = {"created_unix": time.time(), "created_monotonic": time.monotonic(), **meta}
        (self.session_dir / META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._thread = threading.Thread(target=self._run, name="frame-recorder", daemon=True)
        self._accepting = True
        self._thread.start()

    def write(self, frame: Frame) -> bool:
        """Queue ``frame`` for writing; never blocks. Returns ``False`` if it was dropped."""
        if not self._accepting:
            return False
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False
        return True

    def stop(self) -> None:
        """Write everything still queued, then close. Idempotent."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._accepting = False
        self._queue.put(None)  # sentinel; blocks only until the writer frees a slot
        thread.join()

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def error(self) -> Exception | None:
        """The failure that stopped the recording early, if any (e.g. disk full)."""
        with self._lock:
            return self._error

    def _run(self) -> None:
        try:
            with (self.session_dir / INDEX_FILE).open("w", encoding="utf-8") as index:
                while True:
                    frame = self._queue.get()
                    if frame is None:
                        return
                    self._write_frame(frame, index)
        except Exception as exc:  # keep the app alive; report through .error
            log.exception("recording to %s failed", self.session_dir)
            self._accepting = False
            with self._lock:
                self._error = exc
            self._drain()

    def _write_frame(self, frame: Frame, index: Any) -> None:
        with self._lock:
            number = self._written
        name = f"{FRAMES_DIR}/{number:06d}.png"
        if not cv2.imwrite(str(self.session_dir / name), frame.image, _PNG_FAST):
            raise OSError(f"cannot write {name}")
        roi = frame.roi
        entry = {
            "index": frame.index,
            "timestamp": frame.timestamp,
            "roi": [roi.x, roi.y, roi.width, roi.height],
            "scale": frame.scale,
            "file": name,
        }
        index.write(json.dumps(entry) + "\n")
        index.flush()  # a crash loses at most the frame being written
        with self._lock:
            self._written = number + 1

    def _drain(self) -> None:
        """After a failure, empty the queue so ``stop`` cannot block on a full queue."""
        while True:
            try:
                if self._queue.get(timeout=0.2) is None:
                    return
            except queue.Empty:
                if self._thread is None:
                    return
