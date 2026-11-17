"""Capture target description and selection (no OS dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wowfishing.capture.base import CaptureError


@dataclass(frozen=True, slots=True)
class CaptureTarget:
    """A display or window that can be captured."""

    kind: Literal["display", "window"]
    ident: int
    """CoreGraphics display id or window id."""
    label: str
    owner: str
    """Owning application name for windows; empty for displays."""
    width: float
    """Size in display points."""
    height: float


def choose_target(
    targets: list[CaptureTarget], display_id: int | None, window_owner: str | None
) -> CaptureTarget:
    """Pick the target named by the settings; default is the first display listed.

    ``list_targets`` puts the main display first.
    """
    if window_owner is not None:
        for target in targets:
            if target.kind == "window" and target.owner == window_owner:
                return target
        raise CaptureError(f"no on-screen window owned by {window_owner!r}")
    displays = [t for t in targets if t.kind == "display"]
    if display_id is not None:
        for target in displays:
            if target.ident == display_id:
                return target
        raise CaptureError(f"display {display_id} not found")
    if not displays:
        raise CaptureError("no display available")
    return displays[0]
