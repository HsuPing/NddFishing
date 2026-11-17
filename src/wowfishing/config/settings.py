"""Central place for every tunable parameter.

Defaults live in the dataclasses below; a TOML file can override any subset::

    [capture]
    fps = 20
    window_owner = "World of Warcraft"
    roi = { x = 0.25, y = 0.2, width = 0.5, height = 0.5 }

    [audio]
    source = "system"

Unknown sections or keys are rejected so typos do not silently fall back to defaults.
Vision / classifier / state-machine sections are added in the phases that implement them.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from wowfishing.models import Roi

AudioSourceKind = Literal["system", "device"]


class SettingsError(ValueError):
    """The settings file is malformed or contains invalid values."""


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    fps: int = 30
    display_id: int | None = None
    """CoreGraphics display id; ``None`` = main display. Ignored when ``window_owner`` is set."""
    window_owner: str | None = None
    """Capture the first on-screen window owned by this application name instead of a display."""
    roi: Roi = field(default_factory=Roi.full)
    show_cursor: bool = False
    max_width: int = 1920
    """Frames wider than this (pixels) are downscaled by the capture system, keeping aspect."""

    def __post_init__(self) -> None:
        if not 1 <= self.fps <= 120:
            raise SettingsError(f"capture.fps must be within 1..120, got {self.fps}")
        if not 64 <= self.max_width <= 8192:
            raise SettingsError(f"capture.max_width must be within 64..8192, got {self.max_width}")


@dataclass(frozen=True, slots=True)
class AudioSettings:
    source: AudioSourceKind = "system"
    """``system``: app/system output via ScreenCaptureKit. ``device``: a CoreAudio input device."""
    device_name: str | None = None
    """Input device for ``source = "device"`` (e.g. "BlackHole 2ch"); ``None`` = default input."""
    sample_rate: int = 48_000
    block_size: int = 1024
    """Samples per analysis block (~21 ms at 48 kHz)."""

    def __post_init__(self) -> None:
        if self.source not in get_args(AudioSourceKind):
            raise SettingsError(f"audio.source must be one of {get_args(AudioSourceKind)}")
        if self.sample_rate not in (16_000, 24_000, 44_100, 48_000):
            raise SettingsError(f"audio.sample_rate unsupported: {self.sample_rate}")
        if self.block_size < 64 or self.block_size & (self.block_size - 1):
            raise SettingsError(f"audio.block_size must be a power of two >= 64: {self.block_size}")


@dataclass(frozen=True, slots=True)
class PathSettings:
    recordings_dir: Path = Path("recordings")
    datasets_dir: Path = Path("datasets")
    database: Path = Path("datasets/research.sqlite3")
    state_file: Path = Path("state.json")
    """Values changed from the UI (currently the ROI). Overrides ``capture.roi`` when present."""


@dataclass(frozen=True, slots=True)
class Settings:
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    paths: PathSettings = field(default_factory=PathSettings)


Converter = Callable[[object], object]


def _typed(expected: type, description: str) -> Converter:
    """Accept only values of exactly ``expected`` (so ``true`` is not an int)."""

    def convert(value: object) -> object:
        if type(value) is not expected:
            raise TypeError(f"must be {description}, got {type(value).__name__}")
        return value

    return convert


_INT = _typed(int, "an integer")
_STR = _typed(str, "a string")
_BOOL = _typed(bool, "true or false")


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("must be a non-empty path string")
    return Path(value)


def _roi(value: object) -> Roi:
    keys = ("x", "y", "width", "height")
    if not isinstance(value, dict) or set(value) != set(keys):
        raise TypeError("must be a table with exactly x, y, width, height")
    numbers: list[float] = []
    for key in keys:
        item = value[key]
        if type(item) not in (int, float):
            raise TypeError(f"{key} must be a number, got {type(item).__name__}")
        numbers.append(float(item))
    return Roi(*numbers)


# One converter per accepted key: the single source of truth for what a file may contain.
_CAPTURE: dict[str, Converter] = {
    "fps": _INT,
    "display_id": _INT,
    "window_owner": _STR,
    "roi": _roi,
    "show_cursor": _BOOL,
    "max_width": _INT,
}
_AUDIO: dict[str, Converter] = {
    "source": _STR,
    "device_name": _STR,
    "sample_rate": _INT,
    "block_size": _INT,
}
_PATHS: dict[str, Converter] = {
    "recordings_dir": _path,
    "datasets_dir": _path,
    "database": _path,
    "state_file": _path,
}


def _table(section: str, raw: object, spec: dict[str, Converter]) -> dict[str, Any]:
    """Validate one TOML table against ``spec`` and return constructor keyword arguments."""
    if not isinstance(raw, dict):
        raise SettingsError(f"[{section}] must be a table")
    unknown = sorted(set(raw) - set(spec))
    if unknown:
        raise SettingsError(f"[{section}] unknown key(s): {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            values[key] = spec[key](value)
        except (TypeError, ValueError) as exc:
            raise SettingsError(f"{section}.{key}: {exc}") from exc
    return values


def settings_from_dict(data: dict[str, Any]) -> Settings:
    unknown = sorted(set(data) - {"capture", "audio", "paths"})
    if unknown:
        raise SettingsError(f"unknown section(s): {', '.join(unknown)}")
    return Settings(
        capture=CaptureSettings(**_table("capture", data.get("capture", {}), _CAPTURE)),
        audio=AudioSettings(**_table("audio", data.get("audio", {}), _AUDIO)),
        paths=PathSettings(**_table("paths", data.get("paths", {}), _PATHS)),
    )


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``path``; with ``None`` return the defaults."""
    if path is None:
        return Settings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise SettingsError(f"cannot read settings file {path}: {exc}") from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise SettingsError(f"{path}: not valid UTF-8 TOML: {exc}") from exc
    return settings_from_dict(data)
