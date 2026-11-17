"""Pure geometry and pixel conversion for screen capture (no OS dependencies)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wowfishing.models import ImageBGR, Roi


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """How to configure the capture system for one target + ROI."""

    source_rect: tuple[float, float, float, float]
    """``(x, y, width, height)`` to capture, in display points relative to the target."""
    output_size: tuple[int, int]
    """``(width, height)`` of delivered frames, in pixels."""
    scale: float
    """Delivered pixels per display point."""


def plan_capture(
    target_width: float, target_height: float, point_scale: float, roi: Roi, max_width: int
) -> CapturePlan:
    """Translate a normalised ROI into a point-space source rect and a pixel output size.

    ``target_width/height`` are in points and ``point_scale`` is native pixels per
    point (2.0 on Retina). The ROI is snapped to whole native pixels; output wider
    than ``max_width`` is scaled down, keeping the aspect ratio.
    """
    if target_width <= 0 or target_height <= 0 or point_scale <= 0:
        raise ValueError(
            f"invalid target geometry: {target_width}x{target_height} @ {point_scale}x"
        )
    native_w = max(1, round(target_width * point_scale))
    native_h = max(1, round(target_height * point_scale))
    x, y, w, h = roi.to_pixels(native_w, native_h)
    source = (x / point_scale, y / point_scale, w / point_scale, h / point_scale)
    if w > max_width:
        out_w = max_width
        out_h = max(1, round(h * max_width / w))
    else:
        out_w, out_h = w, h
    return CapturePlan(source, (out_w, out_h), out_w / source[2])


def bgra_buffer_to_bgr(
    buffer: memoryview | bytes, width: int, height: int, bytes_per_row: int
) -> ImageBGR:
    """Copy a (possibly row-padded) BGRA pixel buffer into a tight BGR image.

    This is the one unavoidable copy per frame: the capture system reuses its
    buffer as soon as the callback returns.
    """
    if bytes_per_row < width * 4:
        raise ValueError(f"bytes_per_row {bytes_per_row} too small for width {width}")
    needed = bytes_per_row * height
    if len(buffer) < needed:
        raise ValueError(f"buffer has {len(buffer)} bytes, need {needed}")
    rows = np.frombuffer(buffer, dtype=np.uint8, count=needed).reshape(height, bytes_per_row)
    bgra = rows[:, : width * 4].reshape(height, width, 4)
    # asarray only narrows the static type; cvtColor already returns contiguous uint8.
    return np.asarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR), dtype=np.uint8)
