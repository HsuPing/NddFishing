from __future__ import annotations

from pathlib import Path

import pytest

from wowfishing.main import main
from wowfishing.utils import environment
from wowfishing.utils.environment import (
    CheckResult,
    check_audio_inputs,
    check_macos,
    check_package,
    format_report,
    run_checks,
)


def test_missing_package_is_reported_not_raised() -> None:
    result = check_package("definitely_not_installed_xyz", "definitely-not-installed-xyz")
    assert not result.ok
    assert "import failed" in result.detail


def test_installed_package_reports_version() -> None:
    result = check_package("numpy", "numpy")
    assert result.ok
    assert result.detail[0].isdigit()


@pytest.mark.parametrize(
    ("release", "ok"),
    [("13", True), ("13.0", True), ("27.2", True), ("12.7.6", False), ("", False), ("x.y", False)],
)
def test_macos_version_parsing(monkeypatch: pytest.MonkeyPatch, release: str, ok: bool) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.mac_ver", lambda: (release, ("", "", ""), ""))
    assert check_macos().ok is ok


def test_non_macos_platform_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    result = check_macos()
    assert not result.ok and "linux" in result.detail


def test_a_crashing_check_does_not_abort_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> CheckResult:
        raise RuntimeError("ABI mismatch")

    monkeypatch.setattr(environment, "check_displays", boom)
    results = {r.name: r for r in run_checks()}
    assert not results["displays"].ok
    assert "ABI mismatch" in results["displays"].detail
    assert results["python"].ok  # the other checks still ran


def test_no_audio_input_is_informational(monkeypatch: pytest.MonkeyPatch) -> None:
    import sounddevice

    monkeypatch.setattr(sounddevice, "query_devices", lambda: [])
    result = check_audio_inputs()
    assert result.ok and "none" in result.detail


def test_report_marks_failures() -> None:
    report = format_report([CheckResult("a", True, "fine"), CheckResult("bb", False, "broken")])
    lines = report.splitlines()
    assert lines[0].startswith("[OK] a ")
    assert lines[1].startswith("[!!] bb")


def test_main_rejects_bad_config(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "missing.toml")]) == 2
    assert "cannot read settings file" in capfd.readouterr().err


def test_main_check_prints_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--check"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "python" in out and "screen-recording permission" in out
    assert (code == 0) == ("[!!]" not in out)
