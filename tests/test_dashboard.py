from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from wowfishing.capture.base import CaptureError, EndCallback, ErrorCallback, FrameCallback
from wowfishing.capture.targets import CaptureTarget
from wowfishing.config.settings import PathSettings, Settings
from wowfishing.config.state import load_roi, save_roi
from wowfishing.dashboard.main_window import MainWindow
from wowfishing.dashboard.video_widget import VideoWidget, fit_rect, selection_to_roi
from wowfishing.models import Frame, Roi

TARGETS = [
    CaptureTarget("display", 1, "螢幕 1", "", 1280, 832),
    CaptureTarget("window", 7, "Game", "Game", 800, 600),
]


def make_frame(roi: Roi | None = None, index: int = 0) -> Frame:
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 2] = 200  # red in BGR
    return Frame(image=image, timestamp=time.monotonic(), index=index, roi=roi or Roi.full())


def wait_until(condition: Callable[[], bool], timeout_ms: int = 3000) -> None:
    """Run the Qt event loop until ``condition`` holds."""
    deadline = time.monotonic() + timeout_ms / 1000
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        QTest.qWait(20)


class FakeSource:
    """Stands in for ScreenCaptureSource; the test plays the capture thread."""

    def __init__(self, roi: Roi, fail_start: bool = False) -> None:
        self.roi = roi
        self.fail_start = fail_start
        self.on_frame: FrameCallback | None = None
        self.on_error: ErrorCallback | None = None
        self.stopped = False
        self.reject_roi = False

    def start(
        self, on_frame: FrameCallback, on_error: ErrorCallback, on_end: EndCallback | None = None
    ) -> None:
        if self.fail_start:
            raise CaptureError("permission missing")
        self.on_frame, self.on_error = on_frame, on_error

    def stop(self) -> None:
        self.stopped = True

    def set_roi(self, roi: Roi) -> None:
        if self.reject_roi:
            raise CaptureError("rejected")
        self.roi = roi

    def emit(self, index: int = 0) -> None:
        assert self.on_frame is not None
        self.on_frame(make_frame(self.roi, index))


class Harness:
    def __init__(self, tmp_path: Path, fail_start: bool = False) -> None:
        self.sources: list[FakeSource] = []
        self.settings = Settings(
            paths=PathSettings(
                state_file=tmp_path / "state.json", recordings_dir=tmp_path / "recordings"
            )
        )

        def factory(target: CaptureTarget, fps: int, roi: Roi) -> FakeSource:
            source = FakeSource(roi, fail_start)
            self.sources.append(source)
            return source

        self.window = MainWindow(self.settings, lambda: list(TARGETS), factory)

    @property
    def source(self) -> FakeSource:
        return self.sources[-1]


# ----------------------------------------------------------------------------- pure geometry


class TestFitRect:
    def test_wide_image_is_letterboxed(self) -> None:
        assert fit_rect(800, 800, 200, 100) == (0, 200, 800, 400)

    def test_tall_image_is_pillarboxed(self) -> None:
        assert fit_rect(800, 400, 100, 200) == (300, 0, 200, 400)

    def test_degenerate_sizes(self) -> None:
        assert fit_rect(0, 100, 10, 10) == (0, 0, 0, 0)
        assert fit_rect(100, 100, 0, 10) == (0, 0, 0, 0)


class TestSelectionToRoi:
    RECT = (100, 50, 400, 200)

    def test_selection_inside_the_image(self) -> None:
        roi = selection_to_roi((200, 100), (300, 150), self.RECT)
        assert roi == Roi(0.25, 0.25, 0.25, 0.25)

    def test_drag_direction_does_not_matter(self) -> None:
        assert selection_to_roi((300, 150), (200, 100), self.RECT) == Roi(0.25, 0.25, 0.25, 0.25)

    def test_selection_is_clipped_to_the_image(self) -> None:
        roi = selection_to_roi((0, 0), (300, 150), self.RECT)
        assert roi == Roi(0.0, 0.0, 0.5, 0.5)
        assert selection_to_roi((300, 150), (9999, 9999), self.RECT) == Roi(0.5, 0.5, 0.5, 0.5)

    def test_tiny_or_outside_selections_are_ignored(self) -> None:
        assert selection_to_roi((200, 100), (203, 103), self.RECT) is None
        assert selection_to_roi((0, 0), (90, 40), self.RECT) is None
        assert selection_to_roi((200, 100), (300, 150), (0, 0, 0, 0)) is None


# ----------------------------------------------------------------------------- widgets


def test_video_widget_paints_the_frame(qapp: object) -> None:
    widget = VideoWidget()
    widget.resize(800, 400)
    widget.set_frame(make_frame())
    pixel = widget.grab().toImage().pixelColor(400, 200)
    assert (pixel.red(), pixel.green(), pixel.blue()) == (200, 0, 0)  # BGR shown as RGB correctly
    widget.set_frame(None)
    pixel = widget.grab().toImage().pixelColor(5, 5)
    assert pixel.red() < 40


def test_video_widget_drag_emits_roi_only_while_selecting(qapp: object) -> None:
    widget = VideoWidget()
    widget.resize(800, 400)  # frame is 200x100 -> fills the widget exactly
    widget.set_frame(make_frame())
    assert (widget.width(), widget.height()) == (800, 400)
    received: list[Roi] = []
    widget.roi_selected.connect(received.append)

    def drag() -> None:
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(200, 100))
        QTest.mouseMove(widget, QPoint(400, 200))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(600, 300))

    drag()
    assert received == []
    widget.set_selecting(True)
    drag()
    assert received == [Roi(0.25, 0.25, 0.5, 0.5)]


def test_start_shows_frames_and_stop_releases_the_source(qapp: object, tmp_path: Path) -> None:
    h = Harness(tmp_path)
    w = h.window
    assert w.target_box.count() == 2 and w.start_button.isEnabled()
    assert not w.stop_button.isEnabled() and not w.select_roi_button.isEnabled()

    w.start_button.click()
    assert not w.start_button.isEnabled() and w.stop_button.isEnabled()
    h.source.emit(0)
    h.source.emit(1)  # the UI is slower than capture: only the newest is shown
    w._show_latest_frame()
    assert w.video.frame is not None and w.video.frame.index == 1
    w._update_stats()
    assert w.size_label.text() == "200×100 px"
    assert w.dropped_label.text() == "1"
    assert "ms" in w.latency_label.text()

    w.stop_button.click()
    assert h.source.stopped
    assert w.video.frame is None and w.start_button.isEnabled()
    w.close()


def test_start_failure_is_shown_and_leaves_ui_usable(qapp: object, tmp_path: Path) -> None:
    h = Harness(tmp_path, fail_start=True)
    h.window.start_button.click()
    assert "permission missing" in h.window.error_label.text()
    assert h.window.start_button.isEnabled() and not h.window.stop_button.isEnabled()
    h.window.close()


def test_capture_failure_from_another_thread_stops_cleanly(qapp: object, tmp_path: Path) -> None:
    import threading

    h = Harness(tmp_path)
    h.window.start_button.click()
    source = h.source
    assert source.on_error is not None
    thread = threading.Thread(target=source.on_error, args=(CaptureError("display unplugged"),))
    thread.start()
    thread.join()
    QTest.qWait(50)  # deliver the queued signal on the UI thread
    assert source.stopped
    assert "display unplugged" in h.window.error_label.text()
    assert h.window.start_button.isEnabled()
    h.window.close()


def test_roi_selection_on_a_cropped_frame_maps_to_the_target_and_is_saved(
    qapp: object, tmp_path: Path
) -> None:
    h = Harness(tmp_path)
    w = h.window
    w.start_button.click()
    w._apply_roi(Roi(0.5, 0.5, 0.5, 0.5))
    h.source.emit()
    w._show_latest_frame()

    w.select_roi_button.setChecked(True)
    w.video.roi_selected.emit(Roi(0.5, 0.0, 0.5, 0.5))  # right-top quarter of the crop
    expected = Roi(0.75, 0.5, 0.25, 0.25)
    assert h.source.roi == expected
    assert not w.select_roi_button.isChecked()
    assert load_roi(h.settings.paths.state_file) == expected
    assert w.reset_roi_button.isEnabled()

    w.reset_roi_button.click()
    assert h.source.roi == Roi.full()
    assert load_roi(h.settings.paths.state_file) == Roi.full()
    w.close()


def test_rejected_roi_keeps_the_previous_one(qapp: object, tmp_path: Path) -> None:
    h = Harness(tmp_path)
    w = h.window
    w.start_button.click()
    h.source.reject_roi = True
    w._apply_roi(Roi(0.1, 0.1, 0.2, 0.2))
    assert "無法套用 ROI" in w.error_label.text()
    assert w._roi == Roi.full()
    assert load_roi(h.settings.paths.state_file) is None
    w.close()


def test_saved_roi_is_used_for_the_next_start(qapp: object, tmp_path: Path) -> None:
    saved = Roi(0.2, 0.2, 0.4, 0.4)
    save_roi(tmp_path / "state.json", saved)
    h = Harness(tmp_path)
    h.window.start_button.click()
    assert h.source.roi == saved
    h.window.close()
    assert h.source.stopped  # closing the window releases capture


def test_listing_failure_is_reported(qapp: object, tmp_path: Path) -> None:
    def broken() -> list[CaptureTarget]:
        raise CaptureError("no permission")

    settings = Settings(paths=PathSettings(state_file=tmp_path / "state.json"))
    window = MainWindow(settings, broken, lambda t, f, r: FakeSource(r))
    assert "no permission" in window.error_label.text()
    assert not window.start_button.isEnabled()
    window.close()


@pytest.mark.parametrize("owner", ["Game", None])
def test_configured_target_is_preselected(qapp: object, tmp_path: Path, owner: str | None) -> None:
    from wowfishing.config.settings import CaptureSettings

    settings = Settings(
        capture=CaptureSettings(window_owner=owner),
        paths=PathSettings(state_file=tmp_path / "state.json"),
    )
    window = MainWindow(settings, lambda: list(TARGETS), lambda t, f, r: FakeSource(r))
    assert window.target_box.currentData() == TARGETS[1 if owner else 0]
    window.close()


def test_refresh_keeps_the_selected_target(qapp: object, tmp_path: Path) -> None:
    listings = [list(TARGETS)]

    def listing() -> list[CaptureTarget]:
        return listings[-1]

    settings = Settings(paths=PathSettings(state_file=tmp_path / "state.json"))
    window = MainWindow(settings, listing, lambda t, f, r: FakeSource(r))
    try:
        window.target_box.setCurrentIndex(1)
        # A fresh listing returns new, equal-but-not-identical objects; the window
        # title changed and another window appeared in front.
        listings.append(
            [
                TARGETS[0],
                CaptureTarget("window", 99, "Other", "Other", 300, 300),
                CaptureTarget("window", 7, "Game — new title", "Game", 800, 600),
            ]
        )
        window.refresh_button.click()
        selected = window.target_box.currentData()
        assert (selected.kind, selected.ident) == ("window", 7)

        listings.append([TARGETS[0]])  # the window closed: fall back to the default
        window.refresh_button.click()
        assert window.target_box.currentData() == TARGETS[0]
    finally:
        window.close()


def test_restart_resets_the_undisplayed_counter(qapp: object, tmp_path: Path) -> None:
    h = Harness(tmp_path)
    w = h.window
    try:
        w.start_button.click()
        h.source.emit(0)
        h.source.emit(1)
        w._update_stats()
        assert w.dropped_label.text() == "1"
        w.stop_button.click()
        w.start_button.click()
        w._update_stats()
        assert w.dropped_label.text() == "0"
    finally:
        w.close()


def test_recording_saves_delivered_frames_and_stops_with_capture(
    qapp: object, tmp_path: Path
) -> None:
    from wowfishing.capture.replay import iter_frames

    h = Harness(tmp_path)
    w = h.window
    try:
        assert not w.record_button.isEnabled()  # nothing to record before capture starts
        w.start_button.click()
        h.source.emit(0)  # before recording: must not be saved
        w.record_button.click()
        assert w.record_button.text() == "停止錄製"
        h.source.emit(1)
        h.source.emit(2)
        w.stop_button.click()  # stopping capture also finishes the recording
        assert not w.record_button.isChecked()
        wait_until(lambda: w._saving is None)  # flushed in the background, not on the click
        assert w.record_button.text() == "開始錄製"
        assert "已存 2 張" in w.record_label.text()
        assert str((tmp_path / "recordings").resolve()) in w.record_label.text()
        sessions = list((tmp_path / "recordings").iterdir())
        assert len(sessions) == 1
        assert [f.index for f in iter_frames(sessions[0])] == [1, 2]
    finally:
        w.close()


def test_recording_failure_to_start_is_shown(qapp: object, tmp_path: Path) -> None:
    h = Harness(tmp_path)
    w = h.window
    try:
        (tmp_path / "recordings").write_text("a file where the directory should be")
        w.start_button.click()
        w.record_button.click()
        assert "無法開始錄製" in w.error_label.text()
        assert not w.record_button.isChecked() and w._controller.recorder is None
    finally:
        w.close()


def test_stopping_a_recording_does_not_block_the_ui_on_a_slow_disk(
    qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from wowfishing.capture.recorder import FrameRecorder

    gate = threading.Event()
    original = FrameRecorder._write_frame

    def slow(self: FrameRecorder, frame: Frame, index: object) -> None:
        gate.wait(5.0)
        original(self, frame, index)  # type: ignore[arg-type]

    monkeypatch.setattr(FrameRecorder, "_write_frame", slow)
    h = Harness(tmp_path)
    w = h.window
    try:
        w.start_button.click()
        w.record_button.click()
        for i in range(5):
            h.source.emit(i)
        started = time.monotonic()
        w.record_button.click()  # stop recording while the "disk" is stalled
        assert time.monotonic() - started < 0.5
        assert w.record_button.text() == "儲存中…" and not w.record_button.isEnabled()
        assert "儲存中" in w.record_label.text()
        gate.set()
        wait_until(lambda: w._saving is None)
        assert "已存 5 張" in w.record_label.text()
        assert w.record_button.isEnabled() and w.record_button.text() == "開始錄製"
    finally:
        gate.set()
        w.close()


def test_closing_the_window_while_recording_saves_everything(qapp: object, tmp_path: Path) -> None:
    from wowfishing.capture.replay import iter_frames

    h = Harness(tmp_path)
    w = h.window
    w.start_button.click()
    w.record_button.click()
    for i in range(3):
        h.source.emit(i)
    w.close()
    (session,) = (tmp_path / "recordings").iterdir()
    assert [f.index for f in iter_frames(session)] == [0, 1, 2]
    import json

    assert json.loads((session / "meta.json").read_text(encoding="utf-8"))["state"] == "complete"


def test_two_recordings_in_the_same_second_get_distinct_sessions(
    qapp: object, tmp_path: Path
) -> None:
    h = Harness(tmp_path)
    w = h.window
    try:
        w.start_button.click()
        for _ in range(2):
            w.record_button.click()
            h.source.emit()
            w.record_button.click()
            wait_until(lambda: w._saving is None)
            assert w.error_label.text() == ""
        assert len(list((tmp_path / "recordings").iterdir())) == 2
    finally:
        w.close()


def test_recorder_failure_is_surfaced_and_capture_keeps_running(
    qapp: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wowfishing.capture.recorder import FrameRecorder

    def broken(self: FrameRecorder, frame: Frame, index: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(FrameRecorder, "_write_frame", broken)
    h = Harness(tmp_path)
    w = h.window
    try:
        w.start_button.click()
        w.record_button.click()
        h.source.emit()
        wait_until(lambda: (r := w._controller.recorder) is not None and r.error is not None)
        w._update_stats()  # the periodic read-out notices the failure
        wait_until(lambda: w._saving is None)
        assert "錄製中斷" in w.record_label.text() and "disk full" in w.record_label.text()
        assert w._controller.running and w.stop_button.isEnabled()
    finally:
        w.close()
