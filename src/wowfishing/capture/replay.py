"""Read a recorded session back, either all at once or paced like the original capture."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wowfishing.capture.base import CaptureError, EndCallback, ErrorCallback, FrameCallback
from wowfishing.capture.recorder import INDEX_FILE
from wowfishing.models import Frame, Roi

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    """One line of a session index."""

    index: int
    timestamp: float
    roi: Roi
    scale: float
    file: str


def read_index(session_dir: Path) -> list[RecordedFrame]:
    """Parse ``frames.jsonl``. Raises :class:`CaptureError` if missing or malformed.

    A truncated final line (recording interrupted mid-write) is ignored.
    """
    path = session_dir / INDEX_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CaptureError(f"cannot read recording index {path}: {exc}") from exc
    entries: list[RecordedFrame] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            entries.append(
                RecordedFrame(
                    index=int(raw["index"]),
                    timestamp=float(raw["timestamp"]),
                    roi=Roi(*(float(v) for v in raw["roi"])),
                    scale=float(raw["scale"]),
                    file=str(raw["file"]),
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            if number == len(lines):
                log.warning("%s: ignoring truncated last line", path)
                break
            raise CaptureError(f"{path}:{number}: malformed entry: {exc}") from exc
    return entries


def load_frame(session_dir: Path, entry: RecordedFrame) -> Frame:
    """Decode one recorded frame. Raises :class:`CaptureError` if the image is unreadable."""
    image = cv2.imread(str(session_dir / entry.file), cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureError(f"cannot read recorded frame {session_dir / entry.file}")
    return Frame(
        image=np.ascontiguousarray(image, dtype=np.uint8),
        timestamp=entry.timestamp,
        index=entry.index,
        roi=entry.roi,
        scale=entry.scale,
    )


def iter_frames(session_dir: Path) -> Iterator[Frame]:
    """Yield every frame of a session in order, as fast as they decode (offline analysis)."""
    for entry in read_index(session_dir):
        yield load_frame(session_dir, entry)


class ReplayFrameSource:
    """:class:`~wowfishing.capture.base.FrameSource` that plays a recorded session.

    ``speed`` scales the original timing (2.0 = twice as fast); ``0`` delivers
    frames as fast as they can be decoded. Frames keep their recorded timestamps.
    """

    def __init__(self, session_dir: Path, speed: float = 1.0) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self._session_dir = session_dir
        self._speed = speed
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(
        self,
        on_frame: FrameCallback,
        on_error: ErrorCallback,
        on_end: EndCallback | None = None,
    ) -> None:
        if self._thread is not None:
            raise CaptureError("replay already started")
        entries = read_index(self._session_dir)
        if not entries:
            raise CaptureError(f"recording {self._session_dir} contains no frames")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(entries, on_frame, on_error, on_end), name="replay", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        if thread is None:
            return
        if thread is threading.current_thread():
            self._thread = thread
            raise CaptureError("stop() must not be called from a capture callback")
        self._stop_event.set()
        thread.join()

    def _run(
        self,
        entries: list[RecordedFrame],
        on_frame: FrameCallback,
        on_error: ErrorCallback,
        on_end: EndCallback | None,
    ) -> None:
        started = time.monotonic()
        first = entries[0].timestamp
        try:
            for entry in entries:
                if self._speed > 0:
                    due = started + (entry.timestamp - first) / self._speed
                    if self._stop_event.wait(max(0.0, due - time.monotonic())):
                        return
                elif self._stop_event.is_set():
                    return
                frame = load_frame(self._session_dir, entry)
                try:
                    on_frame(frame)
                except Exception:  # contract: consumer callbacks must not raise
                    log.exception("frame callback raised; frame dropped")
        except CaptureError as exc:
            self._report(on_error, exc)
            return
        if on_end is not None and not self._stop_event.is_set():
            try:
                on_end()
            except Exception:
                log.exception("end callback raised")

    @staticmethod
    def _report(on_error: ErrorCallback, error: Exception) -> None:
        try:
            on_error(error)
        except Exception:
            log.exception("error callback raised while reporting: %s", error)
