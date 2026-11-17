from __future__ import annotations

from pathlib import Path

import pytest

from wowfishing.config.settings import (
    AudioSettings,
    CaptureSettings,
    Settings,
    SettingsError,
    load_settings,
    settings_from_dict,
)
from wowfishing.models import Roi


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_file() -> None:
    settings = load_settings(None)
    assert settings == Settings()
    assert settings.capture.roi == Roi.full()
    assert settings.audio.source == "system"


def test_file_overrides_only_given_keys(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [capture]
        fps = 20
        window_owner = "World of Warcraft"
        roi = { x = 0.25, y = 0.2, width = 0.5, height = 0.5 }

        [audio]
        source = "device"
        device_name = "BlackHole 2ch"

        [paths]
        recordings_dir = "/tmp/rec"
        """,
    )
    settings = load_settings(path)
    assert settings.capture.fps == 20
    assert settings.capture.window_owner == "World of Warcraft"
    assert settings.capture.roi == Roi(0.25, 0.2, 0.5, 0.5)
    assert settings.capture.show_cursor is False  # untouched default
    assert settings.audio.source == "device"
    assert settings.audio.sample_rate == 48_000
    assert settings.paths.recordings_dir == Path("/tmp/rec")
    assert settings.paths.datasets_dir == Path("datasets")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[capture]\nfpss = 20\n", "unknown key"),
        ("[vison]\nx = 1\n", "unknown section"),
        ("capture = 3\n", "must be a table"),
        ("[capture]\nfps = 0\n", "fps"),
        ("[capture]\nroi = { x = 0.9, y = 0, width = 0.5, height = 0.5 }\n", "roi"),
        ("[capture]\nroi = { x = 0.1 }\n", "roi"),
        ("[capture]\nroi = 5\n", "roi"),
        ('[audio]\nsource = "loopback"\n', "source"),
        ("[audio]\nblock_size = 1000\n", "block_size"),
        ("[audio]\nsample_rate = 12345\n", "sample_rate"),
        ("[capture\nfps = 1\n", "config.toml"),
        # Wrong types must be rejected with a message naming the key.
        ("[capture]\nfps = true\n", "capture.fps: must be an integer"),
        ("[capture]\nfps = 30.5\n", "capture.fps: must be an integer"),
        ('[capture]\nfps = "30"\n', "capture.fps: must be an integer"),
        ("[capture]\nshow_cursor = 3\n", "capture.show_cursor"),
        ('[capture]\nshow_cursor = "no"\n', "capture.show_cursor"),
        ("[capture]\nwindow_owner = 5\n", "capture.window_owner"),
        ('[capture]\ndisplay_id = "main"\n', "capture.display_id"),
        ("[capture]\nroi = { x = 0, y = 0, width = nan, height = 0.5 }\n", "capture.roi"),
        ("[capture]\nroi = { x = true, y = 0, width = 0.5, height = 0.5 }\n", "capture.roi"),
        ("[audio]\ndevice_name = 3\n", "audio.device_name"),
        ("[audio]\nsample_rate = 48000.0\n", "audio.sample_rate"),
        ("[audio]\nblock_size = 1024.0\n", "audio.block_size"),
        ('[paths]\nrecordings_dir = ""\n', "paths.recordings_dir"),
        ("[paths]\ndatabase = 3\n", "paths.database"),
    ],
)
def test_invalid_files_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(SettingsError, match=message):
        load_settings(write(tmp_path, text))


def test_missing_file_is_a_settings_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="cannot read"):
        load_settings(tmp_path / "nope.toml")


def test_non_utf8_file_is_a_settings_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b'[capture]\nwindow_owner = "caf\xe9"\n')
    with pytest.raises(SettingsError, match="UTF-8"):
        load_settings(path)


def test_integer_roi_values_are_accepted(tmp_path: Path) -> None:
    path = write(tmp_path, "[capture]\nroi = { x = 0, y = 0, width = 1, height = 1 }\n")
    assert load_settings(path).capture.roi == Roi.full()


def test_direct_construction_validates() -> None:
    with pytest.raises(SettingsError):
        CaptureSettings(fps=500)
    with pytest.raises(SettingsError):
        AudioSettings(block_size=48)


def test_settings_from_empty_dict_equals_defaults() -> None:
    assert settings_from_dict({}) == Settings()
