"""Read a recorded session back, either all at once or paced like the original capture."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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

    The recorder ends every line with a newline, so a final line *without* one is
    a write that was interrupted and is ignored. Any other bad line is an error.
    """
    path = session_dir / INDEX_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CaptureError(f"cannot read recording index {path}: {exc}") from exc
    lines = text.split("\n")
    if lines[-1] != "":
        log.warning("%s: ignoring incomplete last line (recording was interrupted)", path)
    lines.pop()  # either the empty string after the final newline or the partial line
    entries: list[RecordedFrame] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entries.append(_parse_entry(json.loads(line)))
        except (ValueError, KeyError, TypeError) as exc:
            raise CaptureError(f"{path}:{number}: malformed entry: {exc}") from exc
    return entries


def _parse_entry(raw: object) -> RecordedFrame:
    if not isinstance(raw, dict):
        raise TypeError("entry is not an object")
    index, timestamp, scale, file = raw["index"], raw["timestamp"], raw["scale"], raw["file"]
    if type(index) is not int or index < 0:
        raise ValueError(f"index must be a non-negative integer: {index!r}")
    for name, value in (("timestamp", timestamp), ("scale", scale)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number: {value!r}")
    if scale <= 0:
        raise ValueError(f"scale must be positive: {scale!r}")
    if not isinstance(file, str):
        raise TypeError("file must be a string")
    relative = PurePosixPath(file)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"file must be a path inside the session: {file!r}")
    return RecordedFrame(
        index=index,
        timestamp=float(timestamp),
        roi=Roi(*(float(v) for v in raw["roi"])),
        scale=float(scale),
        file=file,
    )


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

    Unlike live capture, replay never drops frames: a slow consumer delays
    delivery and the backlog then arrives without pauses. ``stop`` must be called
    after the end (``on_end``) as well as after an error before starting again.
    """

    def __init__(self, session_dir: Path, speed: float = 1.0) -> None:
        if not (math.isfinite(speed) and speed >= 0):
            raise ValueError("speed must be a finite number >= 0")
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
                if self._stop_event.is_set():
                    return
                frame = load_frame(self._session_dir, entry)  # decode first: not part of the wait
                if self._speed > 0:
                    due = started + (entry.timestamp - first) / self._speed
                    if self._stop_event.wait(max(0.0, due - time.monotonic())):
                        return
                try:
                    on_frame(frame)
                except Exception:  # contract: consumer callbacks must not raise
                    log.exception("frame callback raised; frame dropped")
        except CaptureError as exc:
            self._report(on_error, exc)
            return
        except Exception as exc:  # e.g. cv2.error, MemoryError: never die silently
            self._report(on_error, CaptureError(f"replay failed: {exc!r}"))
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
