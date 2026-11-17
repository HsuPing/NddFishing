from __future__ import annotations

import threading
import time

import pytest

from wowfishing.utils.latest_slot import LatestSlot
from wowfishing.utils.performance import RateCounter, RollingStats


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TestRateCounter:
    def test_steady_rate(self) -> None:
        clock = FakeClock()
        counter = RateCounter(window=2.0, clock=clock)
        for _ in range(61):  # 30 events per second for 2 s
            counter.tick()
            clock.now += 1 / 30
        clock.now -= 1 / 30
        assert counter.rate() == pytest.approx(30.0, rel=1e-6)

    def test_zero_before_two_events(self) -> None:
        counter = RateCounter(clock=FakeClock())
        assert counter.rate() == 0.0
        counter.tick()
        assert counter.rate() == 0.0

    def test_rate_falls_to_zero_when_events_stop(self) -> None:
        clock = FakeClock()
        counter = RateCounter(window=2.0, clock=clock)
        for _ in range(10):
            counter.tick()
            clock.now += 0.1
        assert counter.rate() > 0
        clock.now += 5.0
        assert counter.rate() == 0.0

    def test_memory_is_bounded_by_the_window(self) -> None:
        clock = FakeClock()
        counter = RateCounter(window=1.0, clock=clock)
        for _ in range(10_000):
            counter.tick()
            clock.now += 0.01
        assert len(counter._times) <= 102

    def test_reset(self) -> None:
        clock = FakeClock()
        counter = RateCounter(clock=clock)
        counter.tick()
        clock.now += 0.1
        counter.tick()
        counter.reset()
        assert counter.rate() == 0.0

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            RateCounter(window=0)


class TestRollingStats:
    def test_empty(self) -> None:
        assert RollingStats().snapshot() is None

    def test_statistics_cover_only_the_recent_samples(self) -> None:
        stats = RollingStats(size=3)
        for value in (100.0, 1.0, 2.0, 6.0):  # 100 falls out of the window
            stats.add(value)
        snap = stats.snapshot()
        assert snap is not None
        assert (snap.count, snap.last, snap.maximum) == (3, 6.0, 6.0)
        assert snap.mean == pytest.approx(3.0)

    def test_reset(self) -> None:
        stats = RollingStats()
        stats.add(1.0)
        stats.reset()
        assert stats.snapshot() is None


class TestLatestSlot:
    def test_take_returns_newest_and_empties(self) -> None:
        slot: LatestSlot[int] = LatestSlot()
        assert slot.take() is None
        slot.put(1)
        slot.put(2)
        slot.put(3)
        assert slot.take() == 3
        assert slot.take() is None
        assert slot.dropped == 2

    def test_take_waits_for_a_producer(self) -> None:
        slot: LatestSlot[str] = LatestSlot()
        threading.Timer(0.05, slot.put, args=("frame",)).start()
        started = time.monotonic()
        assert slot.take(timeout=2.0) == "frame"
        assert time.monotonic() - started < 1.0

    def test_take_times_out(self) -> None:
        slot: LatestSlot[int] = LatestSlot()
        started = time.monotonic()
        assert slot.take(timeout=0.05) is None
        assert time.monotonic() - started >= 0.04

    def test_clear(self) -> None:
        slot: LatestSlot[int] = LatestSlot()
        slot.put(1)
        slot.clear()
        assert slot.take() is None

    def test_reset_zeroes_the_dropped_counter(self) -> None:
        slot: LatestSlot[int] = LatestSlot()
        slot.put(1)
        slot.put(2)
        slot.reset()
        assert slot.take() is None and slot.dropped == 0

    def test_concurrent_producers_never_lose_the_slot(self) -> None:
        slot: LatestSlot[int] = LatestSlot()
        taken: list[int] = []

        def produce(base: int) -> None:
            for i in range(2000):
                slot.put(base + i)

        threads = [threading.Thread(target=produce, args=(n * 10_000,)) for n in range(4)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            item = slot.take()
            if item is not None:
                taken.append(item)
        for t in threads:
            t.join()
        last = slot.take()
        if last is not None:
            taken.append(last)
        assert len(taken) + slot.dropped == 8000
