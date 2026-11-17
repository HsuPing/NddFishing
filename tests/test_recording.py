from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TextIO

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
    with pytest.raises(RuntimeError):
        recorder.start({})
    recorder.stop()
    recorder.stop()  # idempotent
    assert not recorder.write(make_frame(1, 0.0))
    # Refused writes are neither saved nor reported as disk-too-slow drops.
    assert (recorder.written, recorder.dropped) == (0, 0)


FRAME_BYTES = 24 * 32 * 3


def test_slow_disk_drops_frames_instead_of_blocking_or_growing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    recorder = FrameRecorder(tmp_path / "s", max_pending_bytes=2 * FRAME_BYTES)
    original = recorder._write_frame

    def slow(frame: Frame, index: TextIO) -> None:
        release.wait(5.0)
        original(frame, index)

    monkeypatch.setattr(recorder, "_write_frame", slow)
    recorder.start({})
    started = time.monotonic()
    results = [recorder.write(make_frame(i, float(i))) for i in range(10)]
    assert time.monotonic() - started < 0.5  # the capture thread never waited
    # Memory bound: at most two frames' worth pending at any time (one may be in the writer).
    assert results[:2] == [True, True] and results.count(True) <= 3
    assert recorder.dropped == results.count(False)
    release.set()
    recorder.stop()
    assert recorder.written == results.count(True)
    kept = [f.index for f in iter_frames(tmp_path / "s")]
    assert kept == [i for i, ok in enumerate(results) if ok]  # the accepted ones, in order


def test_one_oversized_frame_is_still_recordable(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "s", max_pending_bytes=10)
    recorder.start({})
    assert recorder.write(make_frame(0, 0.0))
    recorder.stop()
    assert recorder.written == 1


def test_write_failure_is_reported_and_stop_returns_with_a_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    recorder = FrameRecorder(tmp_path / "s")

    def broken(frame: Frame, index: TextIO) -> None:
        gate.wait(5.0)
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_frame", broken)
    recorder.start({})
    for i in range(20):  # a backlog is queued behind the failing write
        assert recorder.write(make_frame(i, float(i)))
    gate.set()
    stopper = threading.Thread(target=recorder.stop)
    stopper.start()
    stopper.join(3.0)
    assert not stopper.is_alive()
    assert isinstance(recorder.error, OSError)
    assert not recorder.write(make_frame(99, 0.0))
    assert recorder.pending == 0  # abandoned frames are released, and counted
    assert recorder.written == 0 and recorder.dropped == 19
    meta = json.loads((tmp_path / "s" / "meta.json").read_text(encoding="utf-8"))
    assert meta["state"] == "failed" and "disk full" in meta["error"]
    assert meta["dropped"] == 19


def test_meta_records_how_the_session_ended(tmp_path: Path) -> None:
    session = tmp_path / "s"
    recorder = FrameRecorder(session)
    recorder.start({"target": "x", "created_unix": "spoofed"})
    during = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    assert during["state"] == "recording"
    assert isinstance(during["created_unix"], float)  # caller meta cannot overwrite ours
    recorder.write(make_frame(0, 1.0))
    recorder.stop()
    after = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    assert (after["state"], after["written"], after["dropped"], after["error"]) == (
        "complete",
        1,
        0,
        None,
    )
    assert after["ended_monotonic"] >= after["created_monotonic"]


def test_request_stop_does_not_block_and_finishes_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    recorder = FrameRecorder(tmp_path / "s")
    original = recorder._write_frame

    def slow(frame: Frame, index: TextIO) -> None:
        gate.wait(5.0)
        original(frame, index)

    monkeypatch.setattr(recorder, "_write_frame", slow)
    recorder.start({})
    for i in range(5):
        recorder.write(make_frame(i, float(i)))
    started = time.monotonic()
    recorder.request_stop()
    assert time.monotonic() - started < 0.2
    assert not recorder.finished
    assert not recorder.write(make_frame(9, 9.0))  # refused once stopping
    gate.set()
    recorder.stop()
    assert recorder.finished and recorder.written == 5 and recorder.dropped == 0


def test_no_frame_is_lost_when_write_races_stop(tmp_path: Path) -> None:
    """Every write() that returned True must end up on disk."""
    for round_number in range(20):
        recorder = FrameRecorder(tmp_path / f"s{round_number}")
        recorder.start({})
        accepted: list[int] = []

        def produce(target: FrameRecorder = recorder, sink: list[int] = accepted) -> None:
            for i in range(200):
                if target.write(make_frame(i, float(i))):
                    sink.append(i)

        producer = threading.Thread(target=produce)
        producer.start()
        time.sleep(0.002)
        recorder.stop()
        producer.join()
        assert recorder.written == len(accepted)


def test_failed_start_leaves_nothing_behind(tmp_path: Path) -> None:
    recorder = FrameRecorder(tmp_path / "s")
    with pytest.raises(ValueError, match="JSON"):
        recorder.start({"bad": object()})
    assert not (tmp_path / "s").exists()
    recorder.start({})  # the same recorder and path are usable afterwards
    recorder.stop()
    assert recorder.finished and recorder.error is None


def test_numpy_scalars_in_frame_fields_are_recorded(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 3), np.uint8)
    frame = Frame(
        image=image,
        timestamp=np.float64(1.5),  # np.float64 is a float subclass
        index=np.int64(7),  # type: ignore[arg-type]
        scale=np.float32(2.0),  # type: ignore[arg-type]
    )
    recorder = FrameRecorder(tmp_path / "s")
    recorder.start({})
    recorder.write(frame)
    recorder.stop()
    assert recorder.error is None
    (entry,) = read_index(tmp_path / "s")
    assert (entry.index, entry.timestamp, entry.scale) == (7, 1.5, 2.0)


def test_frames_of_different_sizes_in_one_session(tmp_path: Path) -> None:
    """The ROI can change while recording."""
    small = Frame(image=np.zeros((4, 6, 3), np.uint8), timestamp=0.0, index=0, roi=ROI)
    large = Frame(image=np.zeros((10, 20, 3), np.uint8), timestamp=0.1, index=1)
    recorder = FrameRecorder(tmp_path / "s")
    recorder.start({})
    recorder.write(small)
    recorder.write(large)
    recorder.stop()
    sizes = [(f.width, f.height, f.roi) for f in iter_frames(tmp_path / "s")]
    assert sizes == [(6, 4, ROI), (20, 10, Roi.full())]


def test_index_errors(tmp_path: Path) -> None:
    with pytest.raises(CaptureError, match="cannot read"):
        read_index(tmp_path / "missing")
    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0), make_frame(1, 0.1)])
    index = session / "frames.jsonl"
    good = index.read_text(encoding="utf-8")
    assert good.endswith("\n")

    index.write_text(good + '{"index": 2, "timest', encoding="utf-8")  # interrupted write
    assert len(read_index(session)) == 2

    index.write_text("garbage\n" + good, encoding="utf-8")
    with pytest.raises(CaptureError, match="malformed"):
        read_index(session)

    # A complete (newline-terminated) bad last line is corruption, not an interrupted write.
    index.write_text(good + "garbage\n", encoding="utf-8")
    with pytest.raises(CaptureError, match="malformed"):
        read_index(session)

    index.write_bytes(b"\xff\xfe\n")
    with pytest.raises(CaptureError, match="cannot read"):
        read_index(session)

    index.write_text("garbage-without-newline", encoding="utf-8")
    assert read_index(session) == []


def _entry(**overrides: object) -> str:
    entry: dict[str, object] = {
        "index": 0,
        "timestamp": 1.0,
        "roi": [0, 0, 1, 1],
        "scale": 1.0,
        "file": "frames/000000.png",
    }
    entry.update(overrides)
    return json.dumps(entry) + "\n"


@pytest.mark.parametrize(
    "line",
    [
        _entry(timestamp=float("inf")),
        _entry(timestamp=float("nan")),
        _entry(scale=0),
        _entry(scale=float("inf")),
        _entry(index=1.9),
        _entry(index=True),
        _entry(index=-1),
        _entry(roi=[0, 0, 1]),
        _entry(roi=[0.9, 0, 0.5, 0.5]),
        _entry(file="/etc/hosts"),
        _entry(file="../outside.png"),
        _entry(file="frames/../../x.png"),
        _entry(file=""),
        _entry(file=5),
        "[1, 2, 3]\n",
        '"text"\n',
    ],
)
def test_bad_index_entries_are_rejected(tmp_path: Path, line: str) -> None:
    (tmp_path / "frames.jsonl").write_text(line, encoding="utf-8")
    with pytest.raises(CaptureError, match="malformed"):
        read_index(tmp_path)


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
    # Paced, not dumped: never faster than recorded. (No upper bound: CI may be slow.)
    assert sink.times[-1] - sink.times[0] >= 0.25


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
    first = threading.Event()

    def on_frame(frame: Frame) -> None:
        sink.on_frame(frame)
        first.set()

    source.start(on_frame, sink.errors.append, sink.ended.set)
    assert first.wait(3.0)  # the next frame is 5 s away
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 1.0
    time.sleep(0.1)
    assert len(sink.frames) == 1 and not sink.ended.is_set()


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


def test_unexpected_failure_in_the_replay_thread_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wowfishing.capture import replay

    session = tmp_path / "s"
    record(session, [make_frame(0, 0.0)])

    def explode(session_dir: Path, entry: object) -> Frame:
        raise MemoryError("decode blew up")

    monkeypatch.setattr(replay, "load_frame", explode)
    errors: list[Exception] = []
    got_error = threading.Event()
    ended = threading.Event()

    def on_error(error: Exception) -> None:
        errors.append(error)
        got_error.set()

    source = ReplayFrameSource(session, speed=0)
    source.start(lambda frame: None, on_error, ended.set)
    assert got_error.wait(3.0), "replay thread died silently"
    source.stop()
    assert isinstance(errors[0], CaptureError) and not ended.is_set()


@pytest.mark.parametrize("speed", [-1.0, float("nan"), float("inf")])
def test_replay_rejects_nonsense_speed(tmp_path: Path, speed: float) -> None:
    with pytest.raises(ValueError):
        ReplayFrameSource(tmp_path, speed=speed)


def test_slow_consumer_delays_replay_but_loses_nothing(tmp_path: Path) -> None:
    session = tmp_path / "s"
    record(session, [make_frame(i, i * 0.02) for i in range(5)])
    seen: list[int] = []
    ended = threading.Event()

    def slow(frame: Frame) -> None:
        if frame.index == 0:
            time.sleep(0.15)
        seen.append(frame.index)

    source = ReplayFrameSource(session)
    source.start(slow, lambda error: None, ended.set)
    assert ended.wait(3.0)
    source.stop()
    assert seen == [0, 1, 2, 3, 4]
