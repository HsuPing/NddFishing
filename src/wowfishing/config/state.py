"""Small JSON state file for values the user changes from the UI (currently the ROI).

Kept apart from the TOML settings so saving from the UI never rewrites a
hand-edited config file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from wowfishing.models import Roi

log = logging.getLogger(__name__)


def load_roi(path: Path) -> Roi | None:
    """Return the saved ROI, or ``None`` when there is none or the file is unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("ignoring unreadable state file %s: %s", path, exc)
        return None
    try:
        raw = data["roi"]
        return Roi(float(raw["x"]), float(raw["y"]), float(raw["width"]), float(raw["height"]))
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("ignoring invalid ROI in state file %s: %s", path, exc)
        return None


def save_roi(path: Path, roi: Roi | None) -> None:
    """Persist ``roi``; ``None`` removes the saved value. Raises ``OSError`` on failure."""
    if roi is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"roi": asdict(roi)}, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file
