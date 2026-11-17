"""Thread-safe rate and rolling statistics for FPS and latency read-outs."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


class RateCounter:
    """Events per second over a sliding time window (e.g. frames -> FPS).

    ``tick`` is called from the producing thread, ``rate`` from any thread.
    """

    def __init__(self, window: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._window = window
        self._clock = clock
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = self._clock()
        with self._lock:
            self._times.append(now)
            self._prune(now)

    def rate(self) -> float:
        now = self._clock()
        with self._lock:
            self._prune(now)
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        with self._lock:
            self._times.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()


@dataclass(frozen=True, slots=True)
class StatsSnapshot:
    count: int
    last: float
    mean: float
    maximum: float


class RollingStats:
    """Last / mean / max over the most recent ``size`` samples (e.g. latency in ms)."""

    def __init__(self, size: int = 120) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._values: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    def snapshot(self) -> StatsSnapshot | None:
        """Return current statistics, or ``None`` before the first sample."""
        with self._lock:
            if not self._values:
                return None
            values = tuple(self._values)
        return StatsSnapshot(len(values), values[-1], sum(values) / len(values), max(values))

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
