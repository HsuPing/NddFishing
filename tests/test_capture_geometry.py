from __future__ import annotations

import numpy as np
import pytest

from wowfishing.capture.base import CaptureError
from wowfishing.capture.geometry import bgra_buffer_to_bgr, plan_capture
from wowfishing.capture.targets import CaptureTarget, choose_target
from wowfishing.models import Roi


class TestPlanCapture:
    def test_full_retina_display_is_downscaled_to_max_width(self) -> None:
        plan = plan_capture(1280, 832, 2.0, Roi.full(), max_width=1920)
        assert plan.source_rect == (0.0, 0.0, 1280.0, 832.0)
        assert plan.output_size == (1920, 1248)
        assert plan.scale == pytest.approx(1.5)

    def test_roi_is_delivered_at_native_resolution(self) -> None:
        plan = plan_capture(1280, 832, 2.0, Roi(0.25, 0.25, 0.5, 0.5), max_width=1920)
        assert plan.source_rect == (320.0, 208.0, 640.0, 416.0)
        assert plan.output_size == (1280, 832)  # 640x416 pt at 2x
        assert plan.scale == pytest.approx(2.0)

    def test_non_retina_display(self) -> None:
        plan = plan_capture(1920, 1080, 1.0, Roi(0.5, 0.0, 0.5, 1.0), max_width=1920)
        assert plan.source_rect == (960.0, 0.0, 960.0, 1080.0)
        assert plan.output_size == (960, 1080)

    def test_source_rect_stays_inside_target(self) -> None:
        plan = plan_capture(1280, 832, 2.0, Roi(0.999, 0.999, 0.001, 0.001), max_width=1920)
        x, y, w, h = plan.source_rect
        assert x + w <= 1280 and y + h <= 832
        assert plan.output_size[0] >= 1 and plan.output_size[1] >= 1

    def test_downscale_keeps_aspect_ratio(self) -> None:
        plan = plan_capture(1920, 1080, 2.0, Roi.full(), max_width=1280)
        assert plan.output_size == (1280, 720)

    @pytest.mark.parametrize("args", [(0, 832, 2.0), (1280, 0, 2.0), (1280, 832, 0.0)])
    def test_rejects_degenerate_targets(self, args: tuple[float, float, float]) -> None:
        with pytest.raises(ValueError):
            plan_capture(*args, Roi.full(), max_width=1920)


class TestBufferConversion:
    def test_drops_alpha_and_keeps_channel_order(self) -> None:
        bgra = np.zeros((2, 3, 4), np.uint8)
        bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
        bgr = bgra_buffer_to_bgr(bgra.tobytes(), 3, 2, 12)
        assert bgr.shape == (2, 3, 3)
        assert bgr.dtype == np.uint8
        assert (bgr == [10, 20, 30]).all()

    def test_row_padding_is_ignored(self) -> None:
        # ScreenCaptureKit pads rows (seen: 400 px wide -> 1664 bytes per row, not 1600).
        width, height, bytes_per_row = 3, 2, 20
        raw = np.full((height, bytes_per_row), 99, np.uint8)  # 99 = padding garbage
        pixels = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width * 4)
        raw[:, : width * 4] = pixels
        bgr = bgra_buffer_to_bgr(raw.tobytes(), width, height, bytes_per_row)
        expected = pixels.reshape(height, width, 4)[..., :3]
        assert (bgr == expected).all()

    def test_result_does_not_alias_the_source_buffer(self) -> None:
        raw = bytearray(4 * 4)
        bgr = bgra_buffer_to_bgr(memoryview(raw), 2, 2, 8)
        raw[:] = b"\xff" * len(raw)  # capture system reuses its buffer
        assert not bgr.any()

    def test_rejects_inconsistent_sizes(self) -> None:
        with pytest.raises(ValueError):
            bgra_buffer_to_bgr(bytes(16), 2, 2, 4)  # stride smaller than a row
        with pytest.raises(ValueError):
            bgra_buffer_to_bgr(bytes(8), 2, 2, 8)  # buffer too short


def _targets() -> list[CaptureTarget]:
    return [
        CaptureTarget("display", 1, "螢幕 1", "", 1280, 832),
        CaptureTarget("display", 3, "螢幕 2", "", 1920, 1080),
        CaptureTarget("window", 77, "Game — main", "Game", 800, 600),
    ]


class TestChooseTarget:
    def test_default_is_first_display(self) -> None:
        assert choose_target(_targets(), None, None).ident == 1

    def test_display_by_id(self) -> None:
        assert choose_target(_targets(), 3, None).ident == 3

    def test_window_owner_wins_over_display(self) -> None:
        assert choose_target(_targets(), 3, "Game").ident == 77

    @pytest.mark.parametrize(("display_id", "owner"), [(9, None), (None, "Nope")])
    def test_missing_target_is_an_error(self, display_id: int | None, owner: str | None) -> None:
        with pytest.raises(CaptureError):
            choose_target(_targets(), display_id, owner)

    def test_no_displays(self) -> None:
        with pytest.raises(CaptureError):
            choose_target([], None, None)
