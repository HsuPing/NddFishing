"""Record captured frames to disk as a lossless image sequence.

Session layout::

    <session>/meta.json        what was captured, when, and how the recording ended
    <session>/frames.jsonl     one line per frame: index, timestamp, roi, scale, file
    <session>/frames/000000.png

PNG (not video) so pixel values are exactly what the detector saw; lossy video
would shift the colours the detector keys on. Timestamps are the frames' own
monotonic timestamps, so an audio recording of the same session lines up.

Each image is written before its index line, and the line is flushed, so an
interrupted recording never lists a frame whose image is incomplete.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import cv2

from wowfishing.models import Frame

log = logging.getLogger(__name__)

INDEX_FILE = "frames.jsonl"
META_FILE = "meta.json"
FRAMES_DIR = "frames"

DEFAULT_PENDING_BYTES = 256 * 1024 * 1024
"""Memory the write queue may pin. ~36 frames at 1920x1248, hundreds at ROI size."""

_PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # speed over size; still lossless
_POLL = 0.1  # how often the idle writer checks for a stop request


class FrameRecorder:
    """Writes frames on its own thread so the capture thread never waits for the disk.

    ``write`` only enqueues. If the disk cannot keep up, queued frames accumulate
    up to ``max_pending_bytes``; beyond that frames are dropped (and counted)
    rather than delaying capture or exhausting memory.

    Stopping has two forms: :meth:`stop` blocks until everything queued is on
    disk; :meth:`request_stop` returns immediately and :attr:`finished` turns
    true when the writer is done (for callers that must not block, e.g. a UI).
    """

    def __init__(self, session_dir: Path, max_pending_bytes: int = DEFAULT_PENDING_BYTES) -> None:
        self.session_dir = session_dir
        self._max_pending_bytes = max_pending_bytes
        self._queue: queue.Queue[Frame] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._meta: dict[str, Any] = {}
        # Guards everything below; held only for bookkeeping, never during disk I/O.
        self._lock = threading.Lock()
        self._accepting = False
        self._pending_bytes = 0
        self._written = 0
        self._dropped = 0
        self._error: Exception | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self, meta: dict[str, Any]) -> None:
        """Create the session directory and begin accepting frames.

        Raises ``OSError`` if the directory exists or cannot be written, and
        ``ValueError`` if ``meta`` is not JSON-serialisable. On failure nothing
        is left behind.
        """
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        try:
            json.dumps(meta)
        except TypeError as exc:
            raise ValueError(f"recording meta is not JSON-serialisable: {exc}") from exc
        # Our own keys last, so caller-supplied meta cannot overwrite them.
        self._meta = {**meta, "created_unix": time.time(), "created_monotonic": time.monotonic()}

        self.session_dir.mkdir(parents=True, exist_ok=False)
        try:
            (self.session_dir / FRAMES_DIR).mkdir()
            self._write_meta({"state": "recording"})
            thread = threading.Thread(target=self._run, name="frame-recorder", daemon=True)
            thread.start()
        except BaseException:
            shutil.rmtree(self.session_dir, ignore_errors=True)  # we created it just above
            raise
        self._thread = thread
        with self._lock:
            self._accepting = True

    def write(self, frame: Frame) -> bool:
        """Queue ``frame`` for writing; never blocks. Returns ``False`` if it was not queued."""
        size = frame.image.nbytes
        with self._lock:
            if not self._accepting:
                return False
            # Always allow one frame, however large, so a single huge frame is not unrecordable.
            if self._pending_bytes and self._pending_bytes + size > self._max_pending_bytes:
                self._dropped += 1
                return False
            self._pending_bytes += size
            self._queue.put_nowait(frame)  # unbounded queue: cannot raise or block
        return True

    def request_stop(self) -> None:
        """Stop accepting frames and let the writer finish in the background. Never blocks."""
        with self._lock:
            self._accepting = False
        self._stop_requested.set()

    @property
    def finished(self) -> bool:
        """True once the writer has flushed everything and closed the session."""
        thread = self._thread
        return thread is None or not thread.is_alive()

    def stop(self) -> None:
        """Write everything still queued, then close. Blocks; idempotent."""
        self.request_stop()
        thread = self._thread
        if thread is not None:
            thread.join()

    # ------------------------------------------------------------------ read-outs

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def pending(self) -> int:
        """Frames queued but not yet on disk."""
        return self._queue.qsize()

    @property
    def error(self) -> Exception | None:
        """The failure that stopped the recording early, if any (e.g. disk full)."""
        with self._lock:
            return self._error

    # ------------------------------------------------------------------ writer thread

    def _run(self) -> None:
        try:
            with (self.session_dir / INDEX_FILE).open("w", encoding="utf-8") as index:
                while True:
                    try:
                        frame = self._queue.get(timeout=_POLL)
                    except queue.Empty:
                        if self._stop_requested.is_set():
                            break  # accepting was cleared first, so nothing can arrive now
                        continue
                    self._write_frame(frame, index)
        except Exception as exc:  # keep the app alive; report through .error
            log.exception("recording to %s failed", self.session_dir)
            with self._lock:
                self._accepting = False
                self._error = exc
        self._finalize()

    def _write_frame(self, frame: Frame, index: TextIO) -> None:
        with self._lock:
            number = self._written
        name = f"{FRAMES_DIR}/{number:06d}.png"
        try:
            if not cv2.imwrite(str(self.session_dir / name), frame.image, _PNG_FAST):
                raise OSError(f"cannot write {name}")
            roi = frame.roi
            entry = {
                "index": int(frame.index),
                "timestamp": float(frame.timestamp),
                "roi": [float(roi.x), float(roi.y), float(roi.width), float(roi.height)],
                "scale": float(frame.scale),
                "file": name,
            }
            index.write(json.dumps(entry) + "\n")
            index.flush()  # a crash loses at most the frame being written
        finally:
            with self._lock:
                self._pending_bytes -= frame.image.nbytes
        with self._lock:
            self._written = number + 1

    def _finalize(self) -> None:
        """Record how the session ended, so offline tools need not guess."""
        abandoned = 0  # only non-zero after a failure
        while True:
            try:
                self._queue.get_nowait()  # release the frames' memory
            except queue.Empty:
                break
            abandoned += 1
        with self._lock:
            self._pending_bytes = 0
            summary = {
                "state": "failed" if self._error is not None else "complete",
                "written": self._written,
                "dropped": self._dropped + abandoned,
                "error": repr(self._error) if self._error is not None else None,
                "ended_unix": time.time(),
                "ended_monotonic": time.monotonic(),
            }
            self._dropped += abandoned
        try:
            self._write_meta(summary)
        except OSError as exc:
            log.warning("cannot update %s: %s", self.session_dir / META_FILE, exc)
            with self._lock:
                if self._error is None:
                    self._error = exc

    def _write_meta(self, extra: dict[str, Any]) -> None:
        path = self.session_dir / META_FILE
        tmp = path.with_name(META_FILE + ".tmp")
        tmp.write_text(
            json.dumps({**self._meta, **extra}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
