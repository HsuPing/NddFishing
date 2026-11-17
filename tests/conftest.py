from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest

# Qt must not open real windows during tests; set before PySide6 is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp() -> Iterator[object]:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _screen_capture_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Quartz
    except ImportError:
        return False
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _screen_capture_available():
        return
    skip = pytest.mark.skip(reason="screen recording permission / macOS display not available")
    for item in items:
        if "capture" in item.keywords:
            item.add_marker(skip)
