"""Single-item hand-off between a producer thread and a slower consumer."""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Holds only the newest item; a new ``put`` replaces an item nobody took.

    This is the frame hand-off: when the consumer is slower than the producer,
    stale frames are dropped instead of queueing, so latency cannot build up.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._dropped = 0

    def put(self, item: T) -> None:
        with self._cond:
            if self._item is not None:
                self._dropped += 1
            self._item = item
            self._cond.notify()

    def take(self, timeout: float | None = 0.0) -> T | None:
        """Remove and return the item.

        ``timeout=0`` (default) never blocks; a positive value waits that long for
        an item; ``None`` waits until one arrives. Returns ``None`` on timeout.
        """
        with self._cond:
            if self._item is None and timeout != 0.0:
                self._cond.wait_for(lambda: self._item is not None, timeout)
            item, self._item = self._item, None
            return item

    def clear(self) -> None:
        """Discard the held item (not counted as dropped)."""
        with self._cond:
            self._item = None

    def reset(self) -> None:
        """Discard the held item and zero the dropped counter (new session)."""
        with self._cond:
            self._item = None
            self._dropped = 0

    @property
    def dropped(self) -> int:
        """How many items were replaced before anyone took them."""
        with self._cond:
            return self._dropped
