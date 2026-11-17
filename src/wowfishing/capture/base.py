"""Interfaces for frame and audio sources.

Live capture (ScreenCaptureKit) and offline replay (recorded files) implement the
same protocols, so everything downstream runs unchanged on recorded data.

Contract (applies to both protocols)
------------------------------------
* ``start`` returns once delivery has begun; it does not wait for frames. (Live
  capture needs roughly 0.1-0.2 s to set up.) Callbacks run on the source's own
  thread, never on the caller's. They must be quick and must not raise; a source
  logs and drops an exception escaping a callback rather than dying.
* ``on_error`` reports a failure after a successful ``start``. The source has
  stopped delivering by then; the owner must still call ``stop`` to release it.
* ``on_end`` is called once when a finite source (replay) runs out. Live sources
  never call it. As after an error, the owner must still call ``stop``.
* ``stop`` is idempotent and may be called from any thread. Once it returns, no
  further callbacks arrive.
* No method of a source may be called from inside one of its own callbacks
  (``stop`` waits for in-flight callbacks, so that would deadlock). Sources
  detect this and raise :class:`CaptureError` instead of hanging.
* ``start`` after ``stop`` is allowed; ``start`` while running raises
  :class:`CaptureError`.

The two protocols share method names on purpose (uniform lifecycle), so a single
object cannot be both. A ScreenCaptureKit session that produces video and audio
from one stream exposes two small source objects instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from wowfishing.models import AudioChunk, Frame, Roi

FrameCallback = Callable[[Frame], None]
AudioCallback = Callable[[AudioChunk], None]
ErrorCallback = Callable[[Exception], None]
EndCallback = Callable[[], None]


class CaptureError(RuntimeError):
    """A capture source could not start or failed while running."""


class FrameSource(Protocol):
    def start(
        self,
        on_frame: FrameCallback,
        on_error: ErrorCallback,
        on_end: EndCallback | None = None,
    ) -> None:
        """Begin delivering frames. Raises :class:`CaptureError` if it cannot start."""
        ...

    def stop(self) -> None: ...


class AdjustableFrameSource(FrameSource, Protocol):
    """A frame source whose region can be changed while it runs (live capture)."""

    def set_roi(self, roi: Roi) -> None:
        """Switch to ``roi``. Raises :class:`CaptureError` if the change is rejected."""
        ...


class AudioSource(Protocol):
    def start(
        self,
        on_audio: AudioCallback,
        on_error: ErrorCallback,
        on_end: EndCallback | None = None,
    ) -> None:
        """Begin delivering audio chunks. Raises :class:`CaptureError` if it cannot start."""
        ...

    def stop(self) -> None: ...
