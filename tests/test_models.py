from __future__ import annotations

import numpy as np
import pytest

from wowfishing.models import AudioChunk, AudioEvent, Detection, Frame, Roi


class TestRoi:
    def test_full_covers_whole_target(self) -> None:
        assert Roi.full().to_pixels(2560, 1664) == (0, 0, 2560, 1664)

    def test_to_pixels_scales_with_resolution(self) -> None:
        roi = Roi(0.25, 0.5, 0.5, 0.25)
        assert roi.to_pixels(1000, 800) == (250, 400, 500, 200)
        # Same normalised ROI on a Retina-sized target covers the same area.
        assert roi.to_pixels(2000, 1600) == (500, 800, 1000, 400)

    def test_to_pixels_never_exceeds_target(self) -> None:
        x, y, w, h = Roi(0.999, 0.999, 0.001, 0.001).to_pixels(100, 100)
        assert w >= 1 and h >= 1
        assert x + w <= 100 and y + h <= 100

    def test_region_ending_at_one_reaches_the_last_pixel(self) -> None:
        # 0.5 * 5 = 2.5: independent rounding of origin and size used to drop column 4.
        assert Roi(0.5, 0.0, 0.5, 1.0).to_pixels(5, 5) == (3, 0, 2, 5)

    @pytest.mark.parametrize("size", [3, 5, 101, 2560])
    def test_adjacent_regions_tile_without_gap_or_overlap(self, size: int) -> None:
        left_x, _, left_w, _ = Roi(0.0, 0.0, 0.5, 1.0).to_pixels(size, size)
        right_x, _, right_w, _ = Roi(0.5, 0.0, 0.5, 1.0).to_pixels(size, size)
        assert left_x == 0
        assert left_x + left_w == right_x
        assert right_x + right_w == size

    @pytest.mark.parametrize(
        "args",
        [
            (-0.1, 0.0, 0.5, 0.5),
            (0.0, 1.0, 0.5, 0.5),
            (0.0, 0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5, -0.1),
            (0.6, 0.0, 0.5, 0.5),
            (0.0, 0.6, 0.5, 0.5),
            (0.0, 0.0, float("nan"), 0.5),
            (float("nan"), 0.0, 0.5, 0.5),
            (0.0, 0.0, 0.5, float("inf")),
        ],
    )
    def test_rejects_invalid_regions(self, args: tuple[float, float, float, float]) -> None:
        with pytest.raises(ValueError):
            Roi(*args)

    def test_accepts_float_rounding_at_the_edge(self) -> None:
        roi = Roi(0.1, 0.1, 0.9 + 1e-12, 0.9)
        assert roi.to_pixels(100, 100) == (10, 10, 90, 90)

    def test_sub_maps_a_selection_inside_a_crop_back_to_the_target(self) -> None:
        outer = Roi(0.2, 0.4, 0.5, 0.5)
        inner = outer.sub(Roi(0.5, 0.5, 0.5, 0.25))
        assert (inner.x, inner.y) == pytest.approx((0.45, 0.65))
        assert (inner.width, inner.height) == pytest.approx((0.25, 0.125))

    def test_sub_of_full_is_identity_both_ways(self) -> None:
        roi = Roi(0.1, 0.2, 0.3, 0.4)
        assert Roi.full().sub(roi) == roi
        assert roi.sub(Roi.full()) == roi

    def test_sub_at_the_far_edge_stays_valid(self) -> None:
        outer = Roi(0.7, 0.7, 0.3, 0.3)
        inner = outer.sub(Roi(0.9, 0.9, 0.1, 0.1))
        assert inner.x + inner.width <= 1.0 + 1e-9
        assert inner.y + inner.height <= 1.0 + 1e-9

    def test_to_pixels_rejects_empty_target(self) -> None:
        with pytest.raises(ValueError):
            Roi.full().to_pixels(0, 100)


def test_frame_reports_size_from_image() -> None:
    frame = Frame(image=np.zeros((120, 160, 3), np.uint8), timestamp=1.0, index=0)
    assert (frame.width, frame.height) == (160, 120)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_a_probability(confidence: float) -> None:
    with pytest.raises(ValueError):
        Detection(x=1.0, y=1.0, bbox=(0, 0, 2, 2), confidence=confidence)
    with pytest.raises(ValueError):
        AudioEvent(label="splash", timestamp=0.0, confidence=confidence)


def test_audio_chunk_duration() -> None:
    chunk = AudioChunk(samples=np.zeros(24_000, np.float32), sample_rate=48_000, timestamp=0.0)
    assert chunk.duration == pytest.approx(0.5)


def test_frame_image_is_read_only_and_identity_compared() -> None:
    image = np.zeros((4, 4, 3), np.uint8)
    frame = Frame(image=image, timestamp=0.0, index=0)
    with pytest.raises(ValueError):
        frame.image[0, 0, 0] = 1
    other = Frame(image=image, timestamp=0.0, index=0)
    assert frame == frame and frame != other
    assert len({frame, other}) == 2  # hashable, usable in sets


def test_frame_rejects_non_bgr_shape() -> None:
    with pytest.raises(ValueError):
        Frame(image=np.zeros((4, 4, 4), np.uint8), timestamp=0.0, index=0)


def test_audio_chunk_validation() -> None:
    with pytest.raises(ValueError):
        AudioChunk(samples=np.zeros(8, np.float32), sample_rate=0, timestamp=0.0)
    with pytest.raises(ValueError):
        AudioChunk(samples=np.zeros((8, 2), np.float32), sample_rate=48_000, timestamp=0.0)
    chunk = AudioChunk(samples=np.zeros(8, np.float32), sample_rate=48_000, timestamp=0.0)
    with pytest.raises(ValueError):
        chunk.samples[0] = 1.0


def test_frame_requires_contiguous_memory() -> None:
    wide = np.zeros((4, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="contiguous"):
        Frame(image=wide[:, ::2], timestamp=0.0, index=0)
    Frame(image=np.ascontiguousarray(wide[:, ::2]), timestamp=0.0, index=0)
