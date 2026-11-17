"""ScreenCaptureSource state logic, driven through its handlers without a real stream."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from wowfishing.capture.base import CaptureError
from wowfishing.capture.geometry import plan_capture
from wowfishing.capture.targets import CaptureTarget
from wowfishing.models import Frame, Roi

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="needs PyObjC")

TARGET = CaptureTarget("display", 1, "螢幕 1", "", 1280, 832)


class FakeStream:
    """Answers completion handlers like an SCStream; can be told to never answer."""

    def __init__(self) -> None:
        self.answer_updates = True
        self.update_error: Any = None
        self.stopped = 0

    def updateConfiguration_completionHandler_(self, config: Any, handler: Any) -> None:
        if self.answer_updates:
            handler(self.update_error)

    def stopCaptureWithCompletionHandler_(self, handler: Any) -> None:
        self.stopped += 1
        handler(None)


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from wowfishing.capture import screen_capture

        self.module = screen_capture
        self.source = screen_capture.ScreenCaptureSource(
            TARGET, fps=30, roi=Roi.full(), max_width=1920
        )
        self.frames: list[Frame] = []
        self.errors: list[Exception] = []
        self.stream = FakeStream()
        self.next_frame: Frame | Exception | None = self.make_frame()
        monkeypatch.setattr(self.source, "_to_frame", self._to_frame)
        monkeypatch.setattr(screen_capture, "_TIMEOUT", 0.2)
        # Put the source into the state start() leaves it in.
        s = self.source
        s._stream = self.stream
        s._geometry = (1280.0, 832.0, 2.0)
        s._plan = plan_capture(1280, 832, 2.0, Roi.full(), 1920)
        s._on_frame, s._on_error = self.frames.append, self.errors.append
        s._running = True

    @staticmethod
    def make_frame() -> Frame:
        return Frame(image=np.zeros((2, 2, 3), np.uint8), timestamp=time.monotonic(), index=0)

    def _to_frame(self, sample_buffer: Any) -> Frame | None:
        if isinstance(self.next_frame, Exception):
            raise self.next_frame
        return self.next_frame


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(monkeypatch)


def test_frames_are_delivered_while_running(h: Harness) -> None:
    h.source._handle_sample(object())
    assert len(h.frames) == 1


def test_idle_or_stale_samples_are_skipped(h: Harness) -> None:
    h.next_frame = None
    h.source._handle_sample(object())
    assert h.frames == [] and h.errors == []


def test_conversion_failure_reports_once_and_stops_delivery(h: Harness) -> None:
    h.next_frame = RuntimeError("bad buffer")
    h.source._handle_sample(object())
    h.source._handle_sample(object())
    assert len(h.errors) == 1 and "bad buffer" in str(h.errors[0])
    assert isinstance(h.errors[0], CaptureError)
    h.next_frame = h.make_frame()
    h.source._handle_sample(object())
    assert h.frames == []  # failed sources stay silent until restarted


def test_consumer_exception_is_contained_and_capture_continues(h: Harness) -> None:
    calls: list[int] = []

    def bad(frame: Frame) -> None:
        calls.append(1)
        raise RuntimeError("consumer bug")

    h.source._on_frame = bad
    h.source._handle_sample(object())
    h.source._handle_sample(object())  # must not raise, must keep delivering
    assert len(calls) == 2 and h.errors == []
    assert h.source._callback_thread is None


def test_stream_stopped_by_system_reports_once(h: Harness) -> None:
    h.source._handle_stopped(None)
    h.source._handle_stopped(None)
    assert len(h.errors) == 1 and "capture stopped" in str(h.errors[0])


def test_error_callback_exception_is_contained(h: Harness) -> None:
    def bad(error: Exception) -> None:
        raise RuntimeError("handler bug")

    h.source._on_error = bad
    h.source._handle_stopped(None)  # must not raise
    assert not h.source._running


@pytest.mark.parametrize("operation", ["stop", "set_roi", "roi", "start"])
def test_calling_back_into_the_source_raises_instead_of_deadlocking(
    h: Harness, operation: str
) -> None:
    caught: list[Exception] = []

    def reentrant(frame: Frame) -> None:
        try:
            if operation == "stop":
                h.source.stop()
            elif operation == "set_roi":
                h.source.set_roi(Roi(0.0, 0.0, 0.5, 0.5))
            elif operation == "roi":
                _ = h.source.roi
            else:
                h.source.start(h.frames.append, h.errors.append)
        except CaptureError as exc:
            caught.append(exc)

    h.source._on_frame = reentrant
    worker = threading.Thread(target=h.source._handle_sample, args=(object(),))
    worker.start()
    worker.join(2.0)
    assert not worker.is_alive(), "deadlocked"
    assert len(caught) == 1 and "capture callback" in str(caught[0])
    h.source.stop()  # still usable from the outside afterwards


def test_no_delivery_after_stop(h: Harness) -> None:
    h.source.stop()
    h.source.stop()
    h.source._handle_sample(object())
    h.source._handle_stopped(None)
    assert h.frames == [] and h.errors == []
    assert h.stream.stopped == 1


def test_set_roi_updates_plan_and_tags(h: Harness) -> None:
    roi = Roi(0.25, 0.25, 0.5, 0.5)
    h.source.set_roi(roi)
    assert h.source.roi == roi
    assert h.source._plan is not None and h.source._plan.output_size == (1280, 832)
    assert not h.source._roi_changing
    h.source._handle_sample(object())
    assert len(h.frames) == 1


def test_rejected_roi_keeps_the_old_one_and_keeps_running(h: Harness) -> None:
    class Error:
        def localizedDescription(self) -> str:
            return "invalid rect"

    h.stream.update_error = Error()
    with pytest.raises(CaptureError, match="invalid rect"):
        h.source.set_roi(Roi(0.25, 0.25, 0.5, 0.5))
    assert h.source.roi == Roi.full()
    h.source._handle_sample(object())
    assert len(h.frames) == 1 and h.errors == []


def test_roi_change_timeout_fails_the_source(h: Harness) -> None:
    h.stream.answer_updates = False
    with pytest.raises(CaptureError, match="timed out"):
        h.source.set_roi(Roi(0.25, 0.25, 0.5, 0.5))
    assert len(h.errors) == 1  # owner is told; the stream state is unknown
    h.source._handle_sample(object())
    assert h.frames == []


def test_frames_are_withheld_while_the_roi_is_changing(h: Harness) -> None:
    h.source._roi_changing = True
    h.source._handle_sample(object())
    assert h.frames == []


def test_set_roi_before_start_is_remembered() -> None:
    from wowfishing.capture.screen_capture import ScreenCaptureSource

    source = ScreenCaptureSource(TARGET, fps=30, roi=Roi.full(), max_width=1920)
    roi = Roi(0.1, 0.1, 0.2, 0.2)
    source.set_roi(roi)
    assert source.roi == roi
    source.stop()  # stopping a never-started source is a no-op


def test_setup_errors_surface_as_capture_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from wowfishing.capture import screen_capture

    def boom() -> Any:
        raise ValueError("unexpected")

    monkeypatch.setattr(screen_capture, "_shareable_content", boom)
    source = screen_capture.ScreenCaptureSource(TARGET, fps=30, roi=Roi.full(), max_width=1920)
    with pytest.raises(CaptureError, match="cannot set up capture"):
        source.start(lambda f: None, lambda e: None)
    assert source._stream is None and not source._running
