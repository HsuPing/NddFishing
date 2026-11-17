"""Data types shared between modules.

Kept dependency-free (NumPy only) so every other module can import it without
pulling in capture, UI, or OS-specific code.

Conventions
-----------
* Timestamps are seconds on the host monotonic clock (the clock behind
  ``time.monotonic()``), for both video and audio, so the two streams can be
  aligned. Sources whose native clock differs must convert before emitting.
* Image coordinates are pixels relative to the frame's top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ImageBGR = NDArray[np.uint8]
"""``(height, width, 3)`` uint8 image in BGR channel order (OpenCV's native order)."""

Samples = NDArray[np.float32]
"""``(n_samples,)`` mono float32 audio in the range [-1.0, 1.0]."""


@dataclass(frozen=True, slots=True)
class Roi:
    """Region of interest, normalised to the capture target (all values in 0..1).

    Normalised coordinates stay valid when the display resolution or the
    Retina scale factor changes; convert with :meth:`to_pixels` at the point of use.
    """

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.width, self.height)):
            raise ValueError(f"ROI values must be finite: {self}")
        if not (0.0 <= self.x < 1.0 and 0.0 <= self.y < 1.0):
            raise ValueError(f"ROI origin must be within [0, 1): {self}")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(f"ROI size must be positive: {self}")
        # Small tolerance: values usually come from float arithmetic on a dragged rectangle.
        if self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9:
            raise ValueError(f"ROI must lie inside the capture target: {self}")

    @classmethod
    def full(cls) -> Roi:
        return cls(0.0, 0.0, 1.0, 1.0)

    def sub(self, inner: Roi) -> Roi:
        """Return ``inner`` (normalised to this ROI) expressed in this ROI's own target.

        Used when the user selects a region on a frame that is itself a crop.
        """
        x = self.x + inner.x * self.width
        y = self.y + inner.y * self.height
        width = min(inner.width * self.width, 1.0 - x)
        height = min(inner.height * self.height, 1.0 - y)
        return Roi(x, y, width, height)

    def to_pixels(self, target_width: int, target_height: int) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` in pixels, inside the target, at least 1x1.

        Both edges are rounded (half up) rather than origin and size, so adjacent
        ROIs tile without gaps and an ROI ending at 1.0 reaches the last pixel.
        """
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"target size must be positive: {target_width}x{target_height}")
        x, w = _span(self.x, self.width, target_width)
        y, h = _span(self.y, self.height, target_height)
        return x, y, w, h


def _span(start: float, length: float, size: int) -> tuple[int, int]:
    """Map a normalised 1-D interval onto ``size`` pixels as ``(offset, extent)``."""
    lo = min(math.floor(start * size + 0.5), size - 1)
    hi = min(math.floor((start + length) * size + 0.5), size)
    return lo, max(1, hi - lo)


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One captured video frame (already cropped to the ROI).

    The same array is shared by the detector, the recorder and the preview to
    avoid per-consumer copies, so construction marks ``image`` read-only.
    Frames compare by identity.
    """

    image: ImageBGR
    timestamp: float
    index: int
    roi: Roi = Roi(0.0, 0.0, 1.0, 1.0)
    """The part of the capture target this frame covers, so results map back to the screen."""
    scale: float = 1.0
    """Frame pixels per display point (2.0 on Retina, less when downscaled). Informational."""

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"image must have shape (h, w, 3), got {self.image.shape}")
        if not self.image.flags.c_contiguous:
            # Consumers wrap the memory directly (e.g. QImage); a sliced view cannot be.
            raise ValueError("image must be C-contiguous; use np.ascontiguousarray")
        self.image.setflags(write=False)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class Detection:
    """A bobber candidate reported by a detector for a single frame."""

    x: float
    """Centre x, in frame pixels."""
    y: float
    """Centre y, in frame pixels."""
    bbox: tuple[int, int, int, int]
    """``(x, y, width, height)`` in frame pixels."""
    confidence: float
    """0.0 (no evidence) .. 1.0 (certain). Detector-specific but always in this range."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1]: {self.confidence}")


@dataclass(frozen=True, slots=True, eq=False)
class AudioChunk:
    """A block of mono audio. ``timestamp`` is the time of the first sample.

    ``samples`` is marked read-only on construction. Chunks compare by identity.
    """

    samples: Samples
    sample_rate: int
    timestamp: float
    first_sample_index: int = 0
    """Running sample count since the source started; a jump means samples were lost."""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive: {self.sample_rate}")
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be mono 1-D, got shape {self.samples.shape}")
        self.samples.setflags(write=False)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioEvent:
    """A sound event reported by a classifier (e.g. the bobber splash)."""

    label: str
    timestamp: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within [0, 1]: {self.confidence}")
