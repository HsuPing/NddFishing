"""Preview widget: paints the newest frame and lets the user drag out an ROI."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from wowfishing.models import Frame, Roi

_MIN_SELECTION = 8
"""Drags smaller than this (widget pixels) are treated as accidental clicks."""


def fit_rect(
    area_width: int, area_height: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Largest ``(x, y, w, h)`` with the image's aspect ratio, centred in the area."""
    if min(area_width, area_height, image_width, image_height) <= 0:
        return 0, 0, 0, 0
    scale = min(area_width / image_width, area_height / image_height)
    width = max(1, round(image_width * scale))
    height = max(1, round(image_height * scale))
    return (area_width - width) // 2, (area_height - height) // 2, width, height


def selection_to_roi(
    start: tuple[int, int], end: tuple[int, int], image_rect: tuple[int, int, int, int]
) -> Roi | None:
    """Convert a drag (widget pixels) into an ROI normalised to the displayed image.

    The selection is clipped to the image; returns ``None`` when what remains is
    too small to be intentional.
    """
    ix, iy, iw, ih = image_rect
    if iw <= 0 or ih <= 0:
        return None
    left = max(min(start[0], end[0]), ix)
    top = max(min(start[1], end[1]), iy)
    right = min(max(start[0], end[0]), ix + iw)
    bottom = min(max(start[1], end[1]), iy + ih)
    if right - left < _MIN_SELECTION or bottom - top < _MIN_SELECTION:
        return None
    return Roi((left - ix) / iw, (top - iy) / ih, (right - left) / iw, (bottom - top) / ih)


class VideoWidget(QWidget):
    roi_selected = Signal(object)
    """Emits a :class:`Roi` normalised to the frame that was on screen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self._frame: Frame | None = None
        self._image: QImage | None = None
        self._selecting = False
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None

    @property
    def frame(self) -> Frame | None:
        return self._frame

    def set_frame(self, frame: Frame | None) -> None:
        self._frame = frame
        if frame is None:
            self._image = None
        else:
            # Wraps the frame's memory without copying; self._frame keeps it alive.
            self._image = QImage(
                frame.image.data,
                frame.width,
                frame.height,
                frame.width * 3,
                QImage.Format.Format_BGR888,
            )
        self.update()

    def set_selecting(self, enabled: bool) -> None:
        self._selecting = enabled
        self._drag_start = self._drag_end = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def _image_rect(self) -> tuple[int, int, int, int]:
        if self._frame is None:
            return 0, 0, 0, 0
        return fit_rect(self.width(), self.height(), self._frame.width, self._frame.height)

    # --- Qt events ------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 20))
        if self._image is None:
            painter.setPen(QColor(130, 130, 135))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "尚未開始擷取")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(*self._image_rect()), self._image)
        if self._selecting and self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(255, 200, 0, 40))
            painter.drawRect(QRect(self._drag_start, self._drag_end).normalized())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._selecting and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._drag_end = event.position().toPoint()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting and self._drag_start is not None:
            self._drag_end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._selecting or self._drag_start is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        start, end = self._drag_start, event.position().toPoint()
        self._drag_start = self._drag_end = None
        self.update()
        roi = selection_to_roi((start.x(), start.y()), (end.x(), end.y()), self._image_rect())
        if roi is not None:
            self.roi_selected.emit(roi)
