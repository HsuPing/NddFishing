"""Command-line entry point.

Without options it opens the monitoring dashboard; ``--check`` runs diagnostics only.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from wowfishing import __version__
from wowfishing.config.settings import Settings, SettingsError, load_settings
from wowfishing.utils.environment import format_report, run_checks
from wowfishing.utils.logger import setup_logging

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wowfishing", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="TOML settings file (default: built-in values)")
    parser.add_argument("--check", action="store_true", help="run environment diagnostics and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        settings = load_settings(args.config)
    except SettingsError as exc:
        log.error("%s", exc)
        return 2
    log.debug("settings: %s", settings)

    if args.check:
        results = run_checks()
        print(format_report(results))
        return 0 if all(r.ok for r in results) else 1

    return run_dashboard(settings)


def run_dashboard(settings: Settings) -> int:
    """Open the monitoring window. Imports Qt and PyObjC lazily so ``--check`` needs neither."""
    from PySide6.QtWidgets import QApplication

    from wowfishing.capture.screen_capture import ScreenCaptureSource, list_targets
    from wowfishing.capture.targets import CaptureTarget
    from wowfishing.dashboard.main_window import MainWindow
    from wowfishing.models import Roi

    def make_source(target: CaptureTarget, fps: int, roi: Roi) -> ScreenCaptureSource:
        return ScreenCaptureSource(
            target,
            fps=fps,
            roi=roi,
            max_width=settings.capture.max_width,
            show_cursor=settings.capture.show_cursor,
        )

    app = QApplication([])
    window = MainWindow(settings, list_targets, make_source)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
