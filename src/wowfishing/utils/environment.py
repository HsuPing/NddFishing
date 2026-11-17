"""Runtime environment diagnostics (``python -m wowfishing --check``).

Each check reports what was actually observed on this machine. Nothing here
triggers a permission prompt or starts a capture.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

MIN_PYTHON = (3, 11)
MIN_MACOS = (13, 0)  # ScreenCaptureKit audio capture needs macOS 13+

_PACKAGES: tuple[tuple[str, str], ...] = (
    # (import name, distribution name)
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python-headless"),
    ("PySide6.QtWidgets", "PySide6-Essentials"),
    ("sounddevice", "sounddevice"),
    ("psutil", "psutil"),
    ("ScreenCaptureKit", "pyobjc-framework-ScreenCaptureKit"),
    ("Quartz", "pyobjc-framework-Quartz"),
    ("CoreMedia", "pyobjc-framework-CoreMedia"),
    ("dispatch", "pyobjc-framework-libdispatch"),
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_python() -> CheckResult:
    version = sys.version_info[:3]
    return CheckResult(
        "python",
        version >= MIN_PYTHON,
        f"{'.'.join(map(str, version))} ({platform.machine()})",
    )


def check_macos() -> CheckResult:
    if sys.platform != "darwin":
        return CheckResult("macos", False, f"unsupported platform: {sys.platform}")
    release = platform.mac_ver()[0]
    try:
        numbers = [int(p) for p in release.split(".")[:2]]
    except ValueError:
        return CheckResult("macos", False, f"cannot parse macOS version {release!r}")
    major, minor = [*numbers, 0][:2]
    return CheckResult("macos", (major, minor) >= MIN_MACOS, f"macOS {release}")


def check_package(import_name: str, dist_name: str) -> CheckResult:
    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        return CheckResult(dist_name, False, f"import failed: {exc}")
    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown version"
    return CheckResult(dist_name, True, version)


def check_screen_recording_permission() -> CheckResult:
    """Report the Screen Recording permission of the process that launched Python."""
    name = "screen-recording permission"
    try:
        import Quartz
    except ImportError as exc:
        return CheckResult(name, False, f"Quartz unavailable: {exc}")
    granted = bool(Quartz.CGPreflightScreenCaptureAccess())
    detail = (
        "granted"
        if granted
        else "not granted: System Settings > Privacy & Security > Screen & System Audio Recording"
    )
    return CheckResult(name, granted, detail)


def check_displays() -> CheckResult:
    name = "displays"
    try:
        import Quartz
    except ImportError as exc:
        return CheckResult(name, False, f"Quartz unavailable: {exc}")
    error, display_ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if error != 0 or count == 0:
        return CheckResult(name, False, f"no active display (CGError {error})")
    parts: list[str] = []
    for display_id in display_ids[:count]:
        mode = Quartz.CGDisplayCopyDisplayMode(display_id)
        if mode is None:
            parts.append(f"id={display_id} (no display mode)")
            continue
        points_w = Quartz.CGDisplayModeGetWidth(mode)
        pixels_w = Quartz.CGDisplayModeGetPixelWidth(mode)
        pixels_h = Quartz.CGDisplayModeGetPixelHeight(mode)
        scale = pixels_w / points_w if points_w else 0.0
        parts.append(f"id={display_id} {pixels_w}x{pixels_h}px scale={scale:g}")
    return CheckResult(name, True, "; ".join(parts))


def check_audio_inputs() -> CheckResult:
    """List CoreAudio input devices.

    Informational: inputs are only needed for ``audio.source = "device"``, so
    having none is not a failure.
    """
    name = "audio input devices"
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # OSError: PortAudio library failed to load
        return CheckResult(name, False, f"sounddevice unavailable: {exc}")
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as exc:
        return CheckResult(name, False, f"PortAudio error: {exc}")
    inputs = [str(d["name"]) for d in devices if d["max_input_channels"] > 0]
    detail = ", ".join(inputs) or 'none (only needed for audio.source = "device")'
    return CheckResult(name, True, detail)


def _guarded(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    """Run one check; a crashing check becomes a failed line instead of aborting the report."""
    try:
        return check()
    except Exception as exc:  # diagnostics must survive a broken native package
        return CheckResult(name, False, f"check crashed: {exc!r}")


def run_checks() -> list[CheckResult]:
    checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("python", check_python),
        ("macos", check_macos),
        *((dist, partial(check_package, imp, dist)) for imp, dist in _PACKAGES),
        ("screen-recording permission", check_screen_recording_permission),
        ("displays", check_displays),
        ("audio input devices", check_audio_inputs),
    ]
    return [_guarded(name, check) for name, check in checks]


def format_report(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = [f"[{'OK' if r.ok else '!!'}] {r.name:<{width}}  {r.detail}" for r in results]
    return "\n".join(lines)
