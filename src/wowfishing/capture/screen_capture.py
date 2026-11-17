"""Live screen capture through ScreenCaptureKit (macOS 13+).

Frames are cropped to the ROI and scaled by the capture system itself, so Python
only ever touches ROI-sized buffers. Frame timestamps are the presentation times
reported by ScreenCaptureKit, which are on the same clock as ``time.monotonic()``
(verified on macOS 27.2).

PyObjC note: an exception escaping a Python callback invoked from Objective-C
terminates the process, as does a completion handler returning anything but
``None``. Every callback in this module is therefore a guarded function that
returns ``None``.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

import CoreMedia
import dispatch
import objc
import Quartz
import ScreenCaptureKit as SCK
from Foundation import NSObject

from wowfishing.capture.base import (
    CaptureError,
    EndCallback,
    ErrorCallback,
    FrameCallback,
)
from wowfishing.capture.geometry import CapturePlan, bgra_buffer_to_bgr, plan_capture
from wowfishing.capture.targets import CaptureTarget
from wowfishing.models import Frame, Roi

log = logging.getLogger(__name__)

_TIMEOUT = 5.0
"""Seconds to wait for a ScreenCaptureKit completion handler."""
_MIN_WINDOW_SIZE = 100.0
"""Windows smaller than this (points) are helper windows, not capture targets."""
_GET_CONTENT = "getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_"

PERMISSION_HINT = (
    "Screen recording permission is missing. Enable the app that launched Python under "
    "System Settings > Privacy & Security > Screen & System Audio Recording, then restart it."
)


def _await_result(start: Callable[[Callable[..., None]], None], what: str) -> tuple[Any, ...]:
    """Run an async ScreenCaptureKit call and wait for its completion handler arguments."""
    done = threading.Event()
    result: list[tuple[Any, ...]] = []

    def handler(*args: Any) -> None:
        result.append(args)
        done.set()

    start(handler)
    if not done.wait(_TIMEOUT):
        raise CaptureError(f"{what} timed out after {_TIMEOUT:g}s")
    return result[0]


def _ignore_result(*_args: Any) -> None:
    """Completion handler for best-effort calls whose outcome no longer matters."""


def _stop_stream_quietly(stream: Any) -> None:
    """Ask a stream to stop without waiting (clean-up of streams nobody owns any more)."""
    try:
        stream.stopCaptureWithCompletionHandler_(_ignore_result)
    except Exception:
        log.debug("best-effort stopCapture failed", exc_info=True)


def _shareable_content() -> Any:
    # Connects this process to the window server. Without it, building a window
    # filter in a process that has no NSApplication aborts with CGS_REQUIRE_INIT.
    Quartz.CGMainDisplayID()
    if not Quartz.CGPreflightScreenCaptureAccess():
        Quartz.CGRequestScreenCaptureAccess()  # shows the system prompt once
        raise CaptureError(PERMISSION_HINT)
    content, error = _await_result(
        lambda handler: getattr(SCK.SCShareableContent, _GET_CONTENT)(True, True, handler),
        "listing capture targets",
    )
    if error is not None or content is None:
        raise CaptureError(f"cannot list capture targets: {_describe(error)}")
    return content


def _describe(error: Any) -> str:
    if error is None:
        return "unknown error"
    try:
        return str(error.localizedDescription())
    except Exception:
        return repr(error)


def list_targets() -> list[CaptureTarget]:
    """Main display first, then other displays, then normal application windows.

    May block for up to a few seconds if the system is slow to answer.
    Raises :class:`CaptureError`.
    """
    content = _shareable_content()
    main_display = int(Quartz.CGMainDisplayID())
    displays = sorted(content.displays(), key=lambda d: int(d.displayID()) != main_display)
    targets: list[CaptureTarget] = []
    for number, display in enumerate(displays, start=1):
        targets.append(
            CaptureTarget(
                kind="display",
                ident=int(display.displayID()),
                label=f"螢幕 {number}（{int(display.width())}×{int(display.height())}）",
                owner="",
                width=float(display.width()),
                height=float(display.height()),
            )
        )
    for window in content.windows():
        size = window.frame().size
        app = window.owningApplication()
        owner = str(app.applicationName()) if app is not None else ""
        if window.windowLayer() != 0 or not owner:
            continue
        if size.width < _MIN_WINDOW_SIZE or size.height < _MIN_WINDOW_SIZE:
            continue
        title = str(window.title() or "")
        targets.append(
            CaptureTarget(
                kind="window",
                ident=int(window.windowID()),
                label=f"{owner} — {title}" if title else owner,
                owner=owner,
                width=float(size.width),
                height=float(size.height),
            )
        )
    return targets


_PROTOCOLS = [objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]


class _StreamOutput(NSObject, protocols=_PROTOCOLS):  # type: ignore[misc, call-arg]
    """Objective-C delegate that forwards to the owning source.

    Holds the source weakly: a strong reference would form a cycle through an
    Objective-C object, which Python's garbage collector cannot see.
    """

    owner: weakref.ref[ScreenCaptureSource] | None = None

    def stream_didOutputSampleBuffer_ofType_(
        self, stream: Any, sample_buffer: Any, kind: int
    ) -> None:
        source = self.owner() if self.owner is not None else None
        if source is not None and kind == SCK.SCStreamOutputTypeScreen:
            source._handle_sample(sample_buffer)

    def stream_didStopWithError_(self, stream: Any, error: Any) -> None:
        source = self.owner() if self.owner is not None else None
        if source is not None:
            source._handle_stopped(error)


class ScreenCaptureSource:
    """:class:`~wowfishing.capture.base.AdjustableFrameSource` backed by one ``SCStream``.

    Only frames with new content are delivered: while the captured region is
    static, ScreenCaptureKit sends idle notifications that are skipped, so the
    measured frame rate drops below the configured one. That is expected.

    ``start`` typically blocks 80-160 ms and ``set_roi`` about 50 ms while
    ScreenCaptureKit answers (worst case: the 5 s timeout).

    Known limitation: the target's size is measured at ``start``. If a captured
    window is resized afterwards, stop and start again.
    """

    def __init__(
        self,
        target: CaptureTarget,
        *,
        fps: int,
        roi: Roi,
        max_width: int,
        show_cursor: bool = False,
    ) -> None:
        self._target = target
        self._fps = fps
        self._max_width = max_width
        self._show_cursor = show_cursor

        # Guards everything below. Held while user callbacks run, which is what
        # makes "no callbacks after stop() returns" true.
        self._lock = threading.Lock()
        self._roi = roi
        self._plan: CapturePlan | None = None
        self._geometry: tuple[float, float, float] | None = None  # width, height (pt), scale
        self._stream: Any = None
        self._output: Any = None
        self._queue: Any = None
        self._finalizer: weakref.finalize[Any, Any] | None = None
        self._on_frame: FrameCallback | None = None
        self._on_error: ErrorCallback | None = None
        self._running = False
        self._roi_changing = False
        self._index = 0
        self._callback_failures = 0
        self._callback_thread: int | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(
        self,
        on_frame: FrameCallback,
        on_error: ErrorCallback,
        on_end: EndCallback | None = None,  # live capture never ends on its own
    ) -> None:
        self._reject_reentry("start")
        with self._lock:
            if self._stream is not None:
                raise CaptureError("capture already started")
        try:
            stream = self._open_stream(on_frame, on_error)
        except CaptureError:
            raise
        except Exception as exc:  # PyObjC / geometry errors must not leak as other types
            raise CaptureError(f"cannot set up capture: {exc!r}") from exc

        try:
            (error,) = _await_result(stream.startCaptureWithCompletionHandler_, "starting capture")
        except CaptureError:
            self._release()
            _stop_stream_quietly(stream)  # it may still come up after the timeout
            raise
        if error is not None:
            self._release()
            raise CaptureError(f"cannot start capture: {_describe(error)}")
        with self._lock:
            cancelled = self._stream is not stream
        if cancelled:  # stop() ran on another thread while we were waiting
            _stop_stream_quietly(stream)
            raise CaptureError("capture was stopped while starting")
        log.info("capturing %s at %d fps", self._target.label, self._fps)

    def _open_stream(self, on_frame: FrameCallback, on_error: ErrorCallback) -> Any:
        content = _shareable_content()
        content_filter = self._make_filter(content)
        geometry = self._measure(content_filter)

        with self._lock:
            if self._stream is not None:
                raise CaptureError("capture already started")
            plan = plan_capture(geometry[0], geometry[1], geometry[2], self._roi, self._max_width)
            output = _StreamOutput.alloc().init()
            output.owner = weakref.ref(self)
            stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
                content_filter, self._configuration(plan), output
            )
            queue = dispatch.dispatch_queue_create(b"wowfishing.capture.frames", None)
            added, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                output, SCK.SCStreamOutputTypeScreen, queue, None
            )
            if not added:
                raise CaptureError(f"cannot attach stream output: {_describe(error)}")
            self._stream, self._output, self._queue = stream, output, queue
            # If this object is dropped without stop(), still stop the system stream.
            self._finalizer = weakref.finalize(self, _stop_stream_quietly, stream)
            self._plan, self._geometry = plan, geometry
            self._on_frame, self._on_error = on_frame, on_error
            self._index = 0
            self._callback_failures = 0
            self._roi_changing = False
            self._running = True
            return stream

    def stop(self) -> None:
        self._reject_reentry("stop")
        stream = self._release()
        if stream is None:
            return
        try:
            (error,) = _await_result(stream.stopCaptureWithCompletionHandler_, "stopping capture")
        except CaptureError as exc:
            log.warning("%s", exc)
            return
        if error is not None:
            # Typical when the stream already stopped by itself (display unplugged, ...).
            log.debug("stopCapture reported: %s", _describe(error))

    def _release(self) -> Any:
        """Detach from the stream so no further callbacks are delivered; return it."""
        with self._lock:
            stream, output, finalizer = self._stream, self._output, self._finalizer
            self._stream = self._output = self._queue = self._finalizer = None
            self._on_frame = self._on_error = None
            self._running = False
        if output is not None:
            output.owner = None
        if finalizer is not None:
            finalizer.detach()
        return stream

    def _reject_reentry(self, operation: str) -> None:
        """Calling back into the source from its own callback would deadlock on the lock."""
        if self._callback_thread == threading.get_ident():
            raise CaptureError(f"{operation}() must not be called from a capture callback")

    # ------------------------------------------------------------------ ROI

    @property
    def roi(self) -> Roi:
        self._reject_reentry("roi")
        with self._lock:
            return self._roi

    def set_roi(self, roi: Roi) -> None:
        """Change the captured region without restarting the stream.

        Frames are withheld while ScreenCaptureKit applies the change, so no
        frame is delivered with content from one region and the tag of another.
        If the change times out the stream is in an unknown state: the source
        reports failure through ``on_error`` and must be stopped.
        """
        self._reject_reentry("set_roi")
        with self._lock:
            stream, geometry = self._stream, self._geometry
            if stream is None or geometry is None:
                self._roi = roi
                return
            self._roi_changing = True
        plan = plan_capture(geometry[0], geometry[1], geometry[2], roi, self._max_width)
        try:
            (error,) = _await_result(
                lambda h: stream.updateConfiguration_completionHandler_(
                    self._configuration(plan), h
                ),
                "changing ROI",
            )
        except CaptureError as exc:
            with self._lock:
                if self._stream is stream and self._running:
                    self._fail_locked(exc)
            raise
        with self._lock:
            if self._stream is not stream:
                return  # stopped meanwhile
            self._roi_changing = False
            if error is not None:
                raise CaptureError(f"cannot change ROI: {_describe(error)}")
            self._roi, self._plan = roi, plan

    # ------------------------------------------------------------------ setup helpers

    def _make_filter(self, content: Any) -> Any:
        target = self._target
        if target.kind == "display":
            for display in content.displays():
                if int(display.displayID()) == target.ident:
                    return SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                        display, []
                    )
        else:
            for window in content.windows():
                if int(window.windowID()) == target.ident:
                    return SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
        raise CaptureError(f"capture target is no longer available: {target.label}")

    def _measure(self, content_filter: Any) -> tuple[float, float, float]:
        """Return the target's ``(width, height)`` in points and its native pixel scale."""
        if content_filter.respondsToSelector_("pointPixelScale"):  # macOS 14+
            size = content_filter.contentRect().size
            return float(size.width), float(size.height), float(content_filter.pointPixelScale())
        # macOS 13 fallback (not verified on real hardware): use the display mode's ratio.
        display_id = (
            self._target.ident if self._target.kind == "display" else Quartz.CGMainDisplayID()
        )
        mode = Quartz.CGDisplayCopyDisplayMode(display_id)
        points = Quartz.CGDisplayModeGetWidth(mode) if mode is not None else 0
        scale = Quartz.CGDisplayModeGetPixelWidth(mode) / points if points else 1.0
        return self._target.width, self._target.height, float(scale)

    def _configuration(self, plan: CapturePlan) -> Any:
        config = SCK.SCStreamConfiguration.alloc().init()
        config.setSourceRect_(Quartz.CGRectMake(*plan.source_rect))
        config.setWidth_(plan.output_size[0])
        config.setHeight_(plan.output_size[1])
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self._fps))
        config.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
        config.setShowsCursor_(self._show_cursor)
        config.setQueueDepth_(3)  # ScreenCaptureKit's minimum; we keep no backlog
        return config

    # ------------------------------------------------------------------ callbacks (capture queue)

    def _handle_sample(self, sample_buffer: Any) -> None:
        with self._lock:
            if not self._running or self._roi_changing:
                return
            try:
                frame = self._to_frame(sample_buffer)
            except Exception as exc:  # never let an exception reach Objective-C
                self._fail_locked(CaptureError(f"frame conversion failed: {exc!r}"))
                return
            if frame is None or self._on_frame is None:
                return
            self._callback_thread = threading.get_ident()
            try:
                self._on_frame(frame)
            except Exception:  # contract: consumer callbacks must not raise
                self._callback_failures += 1
                if self._callback_failures == 1 or self._callback_failures % 100 == 0:
                    log.exception(
                        "frame callback raised (%d so far); frame dropped",
                        self._callback_failures,
                    )
            finally:
                self._callback_thread = None

    def _handle_stopped(self, error: Any) -> None:
        with self._lock:
            if self._running:
                self._fail_locked(CaptureError(f"capture stopped: {_describe(error)}"))

    def _fail_locked(self, error: CaptureError) -> None:
        """Stop delivering and report ``error`` once. Caller holds the lock."""
        self._running = False
        on_error = self._on_error
        if on_error is None:
            return
        self._callback_thread = threading.get_ident()
        try:
            on_error(error)
        except Exception:
            log.exception("error callback raised while reporting: %s", error)
        finally:
            self._callback_thread = None

    def _to_frame(self, sample_buffer: Any) -> Frame | None:
        """Convert a sample buffer; ``None`` for idle frames and frames of a stale size."""
        attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample_buffer, False)
        if not attachments:
            return None
        if attachments[0].get(SCK.SCStreamFrameInfoStatus) != SCK.SCFrameStatusComplete:
            return None
        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        plan = self._plan
        if pixel_buffer is None or plan is None:
            return None

        timestamp = CoreMedia.CMTimeGetSeconds(
            CoreMedia.CMSampleBufferGetPresentationTimeStamp(sample_buffer)
        )
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            if (width, height) != plan.output_size:
                return None  # produced before the latest ROI change took effect
            bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            image = bgra_buffer_to_bgr(
                base.as_buffer(bytes_per_row * height), width, height, bytes_per_row
            )
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly)

        frame = Frame(
            image=image, timestamp=timestamp, index=self._index, roi=self._roi, scale=plan.scale
        )
        self._index += 1
        return frame
