"""Integration tests against the real ScreenCaptureKit (skipped without permission)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from wowfishing.capture.base import CaptureError
from wowfishing.models import Frame, Roi

if TYPE_CHECKING:
    from wowfishing.capture.screen_capture import ScreenCaptureSource
    from wowfishing.capture.targets import CaptureTarget

pytestmark = pytest.mark.capture


class Collector:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.received_at: list[float] = []
        self.errors: list[Exception] = []
        self.threads: set[str] = set()
        self._cond = threading.Condition()

    def on_frame(self, frame: Frame) -> None:
        with self._cond:
            self.frames.append(frame)
            self.received_at.append(time.monotonic())
            self.threads.add(threading.current_thread().name)
            self._cond.notify_all()

    def on_error(self, error: Exception) -> None:
        with self._cond:
            self.errors.append(error)
            self._cond.notify_all()

    def wait_for(self, predicate: Callable[[list[Frame]], bool], timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: predicate(self.frames), timeout)


def _source(
    roi: Roi, max_width: int = 8192, fps: int = 30
) -> tuple[ScreenCaptureSource, CaptureTarget]:
    from wowfishing.capture.screen_capture import ScreenCaptureSource, list_targets
    from wowfishing.capture.targets import choose_target

    target = choose_target(list_targets(), None, None)
    return ScreenCaptureSource(target, fps=fps, roi=roi, max_width=max_width), target


def test_targets_include_a_display() -> None:
    from wowfishing.capture.screen_capture import list_targets

    targets = list_targets()
    displays = [t for t in targets if t.kind == "display"]
    assert displays and all(t.width > 0 and t.height > 0 for t in displays)
    assert all(t.owner for t in targets if t.kind == "window")


def test_delivers_roi_sized_frames_off_the_main_thread() -> None:
    roi = Roi(0.0, 0.0, 0.25, 0.25)
    source, target = _source(roi)
    collector = Collector()
    source.start(collector.on_frame, collector.on_error)
    try:
        assert collector.wait_for(lambda frames: len(frames) >= 1), "no frame within 5 s"
    finally:
        source.stop()
    assert not collector.errors
    frame = collector.frames[0]
    assert frame.image.dtype == np.uint8 and frame.image.shape[2] == 3
    assert not frame.image.flags.writeable
    assert frame.roi == roi
    assert frame.index == 0
    assert threading.current_thread().name not in collector.threads
    # Timestamp is on the time.monotonic() clock: received shortly after presentation.
    delay = collector.received_at[0] - frame.timestamp
    assert -0.05 < delay < 1.0, f"timestamp is not on the monotonic clock (delay {delay:.3f}s)"
    # A quarter of the display in each direction, at native pixel density.
    assert frame.scale in (1.0, 2.0, 3.0)  # native pixel density, not downscaled
    assert frame.width == round(target.width * frame.scale * 0.25)
    assert frame.height == round(target.height * frame.scale * 0.25)


def test_no_callbacks_after_stop_and_restart_works() -> None:
    source, _ = _source(Roi(0.0, 0.0, 0.2, 0.2))
    collector = Collector()
    source.start(collector.on_frame, collector.on_error)
    with pytest.raises(CaptureError):
        source.start(collector.on_frame, collector.on_error)
    assert collector.wait_for(lambda frames: len(frames) >= 1)
    source.stop()
    source.stop()  # idempotent
    # Detached from the stream: nothing can be delivered any more, even if the
    # screen changes (a frame count over a static screen would prove nothing).
    assert source._stream is None and source._output is None and not source._running
    count = len(collector.frames)
    time.sleep(0.2)
    assert len(collector.frames) == count

    second = Collector()
    source.start(second.on_frame, second.on_error)
    try:
        assert second.wait_for(lambda frames: len(frames) >= 1)
        assert second.frames[0].index == 0
    finally:
        source.stop()


def test_full_frame_is_downscaled_to_max_width() -> None:
    source, _ = _source(Roi.full(), max_width=640)
    collector = Collector()
    source.start(collector.on_frame, collector.on_error)
    try:
        assert collector.wait_for(lambda frames: len(frames) >= 1)
    finally:
        source.stop()
    assert collector.frames[0].width == 640


def test_roi_change_crops_the_same_pixels_as_the_full_frame() -> None:
    """The cropped stream must show exactly the region the ROI names (Retina-safe)."""
    source, _ = _source(Roi.full())
    collector = Collector()
    source.start(collector.on_frame, collector.on_error)
    try:
        assert collector.wait_for(lambda frames: len(frames) >= 1)
        full = collector.frames[-1]
        # Top-left of the menu bar: static content, so both captures show the same pixels.
        roi = Roi(0.0, 0.0, 0.2, 0.025)
        source.set_roi(roi)
        assert collector.wait_for(lambda frames: frames[-1].roi == roi, timeout=5.0), (
            "no frame after ROI change (static screen?)"
        )
        cropped = collector.frames[-1]
    finally:
        source.stop()
    x, y, w, h = roi.to_pixels(full.width, full.height)
    assert (cropped.width, cropped.height) == (w, h)
    expected = full.image[y : y + h, x : x + w].astype(np.int16)
    difference = np.abs(expected - cropped.image.astype(np.int16)).mean()
    assert difference < 2.0, (
        f"cropped frame differs from full frame region (mean diff {difference})"
    )


def test_callback_exception_does_not_kill_capture() -> None:
    source, _ = _source(Roi(0.0, 0.0, 0.2, 0.2))
    calls: list[int] = []
    errors: list[Exception] = []
    done = threading.Event()

    def bad_callback(frame: Frame) -> None:
        calls.append(frame.index)
        done.set()
        raise RuntimeError("consumer bug")

    source.start(bad_callback, errors.append)
    try:
        assert done.wait(5.0)
    finally:
        source.stop()
    assert calls and not errors  # logged and dropped; the process is still alive


_WINDOW_SCRIPT = """
import threading
from wowfishing.capture.screen_capture import ScreenCaptureSource, list_targets
from wowfishing.models import Roi

from wowfishing.capture.base import CaptureError

windows = [t for t in list_targets() if t.kind == "window"]
if not windows:
    print("NO_WINDOW")
    raise SystemExit(0)
outcome = "TIMEOUT"
for window in windows[:5]:  # a window can vanish between listing and starting
    got = threading.Event()
    source = ScreenCaptureSource(window, fps=30, roi=Roi.full(), max_width=1920)
    try:
        source.start(lambda frame: got.set(), print)
    except CaptureError:
        continue
    ok = got.wait(5.0)
    source.stop()
    if ok:
        outcome = "FRAME"
        break
print(outcome)
"""


def test_window_capture_works_in_a_process_without_a_gui_app() -> None:
    """Regression: a window filter used to abort the process (CGS_REQUIRE_INIT)."""
    result = subprocess.run(
        [sys.executable, "-c", _WINDOW_SCRIPT], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr[-500:]
    if "NO_WINDOW" in result.stdout:
        pytest.skip("no capturable window on screen")
    assert "FRAME" in result.stdout, result.stdout + result.stderr[-300:]


def test_source_dropped_without_stop_is_collected_and_stops_delivering() -> None:
    """The delegate must not keep the source alive through an Objective-C cycle."""
    import gc

    source, _ = _source(Roi(0.0, 0.0, 0.2, 0.2))
    collector = Collector()
    source.start(collector.on_frame, collector.on_error)
    assert collector.wait_for(lambda frames: len(frames) >= 1)
    alive = weakref.ref(source)
    del source
    gc.collect()
    assert alive() is None


def test_stop_during_start_never_leaves_a_silently_dead_source() -> None:
    """Either start() succeeds and frames flow, or it raises; no in-between."""
    source, _ = _source(Roi(0.0, 0.0, 0.2, 0.2))
    collector = Collector()
    outcome: list[str] = []

    def run_start() -> None:
        try:
            source.start(collector.on_frame, collector.on_error)
            outcome.append("started")
        except CaptureError as exc:
            outcome.append(f"error: {exc}")

    starter = threading.Thread(target=run_start)
    starter.start()
    deadline = time.monotonic() + 5.0
    while source._stream is None and starter.is_alive() and time.monotonic() < deadline:
        time.sleep(0.0005)
    source.stop()  # races the start that is waiting for its completion handler
    starter.join(10.0)
    assert not starter.is_alive()
    try:
        if outcome == ["started"]:
            # stop() ran too late or too early to cancel: the source must really work
            assert source._stream is not None and source._running
        else:
            assert "stopped while starting" in outcome[0]
            assert source._stream is None and not source._running
    finally:
        source.stop()
