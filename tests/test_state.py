from __future__ import annotations

from pathlib import Path

import pytest

from wowfishing.config.state import load_roi, save_roi
from wowfishing.models import Roi


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    roi = Roi(0.1, 0.2, 0.3, 0.4)
    save_roi(path, roi)
    assert load_roi(path) == roi
    assert not path.with_name("state.json.tmp").exists()


def test_missing_file_means_no_roi(tmp_path: Path) -> None:
    assert load_roi(tmp_path / "state.json") is None


def test_saving_none_removes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    save_roi(path, Roi.full())
    save_roi(path, None)
    assert not path.exists()
    save_roi(path, None)  # idempotent


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        "{}",
        '{"roi": 5}',
        '{"roi": {"x": 0.1}}',
        '{"roi": {"x": 0.9, "y": 0, "width": 0.5, "height": 0.5}}',
        '{"roi": {"x": "a", "y": 0, "width": 0.5, "height": 0.5}}',
        '{"roi": {"x": NaN, "y": 0, "width": 0.5, "height": 0.5}}',
    ],
)
def test_corrupt_state_is_ignored_not_fatal(tmp_path: Path, content: str) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")
    assert load_roi(path) is None
