from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from wowfishing.capture.base import CaptureError
from wowfishing.capture.recorder import FrameRecorder
from wowfishing.capture.replay import ReplayFrameSource, iter_frames, read_index
from wowfishing.models import Frame, Roi

ROI = Roi(0.25, 0.5, 0.5, 0.25)


def make_frame(index: int, timestamp: float) -> Frame:
    rng = np.random.default_rng(index)
    image = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    return Frame(image=image, timestamp=timestamp, index=index, roi=ROI, scale=2.0)


def record(session: Path, frames: list[Frame]) -> FrameRecorder:
    recorder = FrameRecorder(session)
    recorder.start({"target": "測試", "fps": 30})
    for frame in frames:
        assert recorder.write(frame)
    recorder.stop()
    return recorder


def test_round_trip_is_lossless_and_keeps_metadata(tmp_path: Path) -> None:
    frames = [make_frame(i * 3, 100.0 + i * 0.05) for i in range(5)]
    recorder = record(tmp_path / "s", frames)
    assert (recorder.written, recorder.dropped, recorder.error) == (5, 0, None)

    meta = json.loads((tmp_path / "s" / "meta.json").read_text(encoding="utf-8"))
    assert meta["target"] == "測試" and meta["fps"] == 30 and "created_unix" in meta

    replayed = list(iter_frames(tmp_path / "s"))
    assert len(replayed) == 5
    for original, copy in zip(frames, replayed, strict=True):
        assert np.array_equal(original.image, copy.image)  # exact pixels, not "close"
        assert (copy.index, copy.timestamp, copy.roi, copy.scale) == (
            original.index,
            original.timestamp,
            ROI,
            2.0,
        )
        assert not copy.image.flags.writeable


def test_existing_session_directory_is_not_overwritten(tmp_path: Path) -> None:
    record(tmp_path / "s", [make_frame(0, 0.0)])
    with pytest.raises(OSError):
        FrameRecorder(tmp_path / "s").start({})


def test_write_before_start_and_after_stop_is_refused(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "s")
    assert not recorder.write(make_frame(0, 0.0))
    recorder.start({})
    recorder.stop()
    recorder.stop()  # idempotent
    assert not recorder.write(make_frame(1, 0.0))
    assert recorder.written == 0


def test_slow_disk_drops_frames_instead_of_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    recorder = FrameRecorder(tmp_path / "s", queue_size=2)
    original = recorder._write_frame

    def slow(frame: Frame, index: object) -> None:
        release.wait(5.0)
        original(frame, index)

    monkeypatch.setattr(recorder, "_write_frame", slow)
    recorder.start({})
    started = time.monotonic()
    results = [recorder.write(make_frame(i, float(i))) for i in range(10)]
    assert time.monotonic() - started < 0.5  # the capture thread never waited
    assert results.count(False) >= 7 and recorder.dropped == results.count(False)
    release.set()
    recorder.stop()
    assert recorder.written == results.count(True)
    assert len(read_index(tmp_path / "s")) == recorder.written


def test_write_failure_is_reported_and_stop_still_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = FrameRecorder(tmp_path / "s", queue_size=2)

    def broken(frame: Frame, index: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_frame", broken)
    recorder.start({})
    recorder.write(make_frame(0, 0.0))
    deadline = time.monotonic() + 2.0
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(recorder.error, OSError)
    assert not recorder.write(make_frame(1, 0.0))
    stopper = threading.Thread(target=recorder.stop)
    stopper.start()
    stopper.join(3.0)
    assert not stopper.is_alive()


def test_index_errors(tmp_path: Path) -> None:
    with pytest.raises(CaptureError, match="cannot read"):
        read_index(tmp_path / "missing")
    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0), make_frame(1, 0.1)])
    index = session / "frames.jsonl"
    good = index.read_text(encoding="utf-8")

    index.write_text(good + '{"index": 2, "timest', encoding="utf-8")  # interrupted write
    assert len(read_index(session)) == 2

    index.write_text("garbage\n" + good, encoding="utf-8")
    with pytest.raises(CaptureError, match="malformed"):
        read_index(session)


def test_missing_image_is_a_capture_error(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0)])
    (session / "frames" / "000000.png").unlink()
    with pytest.raises(CaptureError, match="cannot read recorded frame"):
        list(iter_frames(session))


class Sink:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.times: list[float] = []
        self.errors: list[Exception] = []
        self.ended = threading.Event()

    def on_frame(self, frame: Frame) -> None:
        self.frames.append(frame)
        self.times.append(time.monotonic())


def test_replay_source_follows_recorded_timing_and_signals_the_end(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(i, 50.0 + i * 0.1) for i in range(4)])  # 0.3 s long
    sink = Sink()
    source = ReplayFrameSource(session, speed=1.0)
    source.start(sink.on_frame, sink.errors.append, sink.ended.set)
    with pytest.raises(CaptureError):
        source.start(sink.on_frame, sink.errors.append)
    assert sink.ended.wait(3.0)
    source.stop()
    assert [f.index for f in sink.frames] == [0, 1, 2, 3] and not sink.errors
    duration = sink.times[-1] - sink.times[0]
    assert 0.25 < duration < 0.8


def test_replay_at_full_speed_and_restart(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(i, i * 10.0) for i in range(3)])  # 20 s of "real" time
    sink = Sink()
    source = ReplayFrameSource(session, speed=0)
    started = time.monotonic()
    source.start(sink.on_frame, sink.errors.append, sink.ended.set)
    assert sink.ended.wait(3.0) and time.monotonic() - started < 2.0
    source.stop()
    sink.ended.clear()
    source.start(sink.on_frame, sink.errors.append, sink.ended.set)
    assert sink.ended.wait(3.0)
    source.stop()
    assert len(sink.frames) == 6


def test_replay_stop_interrupts_and_suppresses_the_end_signal(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(i, i * 5.0) for i in range(3)])
    sink = Sink()
    source = ReplayFrameSource(session)
    source.start(sink.on_frame, sink.errors.append, sink.ended.set)
    time.sleep(0.1)
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 1.0
    count = len(sink.frames)
    time.sleep(0.1)
    assert len(sink.frames) == count == 1 and not sink.ended.is_set()


def test_replay_reports_unreadable_frames_and_rejects_empty_sessions(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0), make_frame(1, 0.01)])
    (session / "frames" / "000001.png").unlink()
    sink = Sink()
    got_error = threading.Event()

    def on_error(error: Exception) -> None:
        sink.errors.append(error)
        got_error.set()

    source = ReplayFrameSource(session, speed=0)
    source.start(sink.on_frame, on_error, sink.ended.set)
    assert got_error.wait(3.0)
    source.stop()
    assert len(sink.frames) == 1 and not sink.ended.is_set()

    empty = tmp_path / "empty"
    record(empty, [])
    with pytest.raises(CaptureError, match="no frames"):
        ReplayFrameSource(empty).start(sink.on_frame, sink.errors.append)


def test_replay_stop_from_its_own_callback_raises(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0)])
    caught: list[Exception] = []
    done = threading.Event()
    source = ReplayFrameSource(session, speed=0)

    def on_frame(frame: Frame) -> None:
        try:
            source.stop()
        except CaptureError as exc:
            caught.append(exc)

    source.start(on_frame, caught.append, done.set)
    assert done.wait(3.0)
    source.stop()
    assert len(caught) == 1 and "callback" in str(caught[0])
