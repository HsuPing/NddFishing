"""Main monitoring window (Phase 2: preview, capture controls, ROI, statistics)."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psutil
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from wowfishing.capture.base import CaptureError
from wowfishing.capture.recorder import FrameRecorder
from wowfishing.capture.targets import CaptureTarget, choose_target
from wowfishing.config.settings import Settings
from wowfishing.config.state import load_roi, save_roi
from wowfishing.dashboard.capture_controller import CaptureController, SourceFactory
from wowfishing.dashboard.video_widget import VideoWidget
from wowfishing.models import Roi

log = logging.getLogger(__name__)

_DISPLAY_INTERVAL_MS = 15
_STATS_INTERVAL_MS = 1000
_SAVING_POLL_MS = 100

STYLE = """
QWidget { background: #1c1c1f; color: #dcdce0; font-size: 13px; }
QGroupBox { border: 1px solid #34343a; border-radius: 6px; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #9a9aa2; }
QPushButton { background: #2c2c32; border: 1px solid #44444c; border-radius: 4px;
  padding: 5px 10px; }
QPushButton:hover { background: #36363e; }
QPushButton:checked { background: #6b5200; border-color: #c79a00; }
QPushButton:disabled { color: #66666c; }
QComboBox, QSpinBox { background: #26262b; border: 1px solid #44444c; border-radius: 4px;
  padding: 3px; }
QLabel#value { font-family: Menlo; color: #f0f0f4; }
QLabel#error { color: #ff7b72; }
"""


def new_session_dir(recordings_dir: Path) -> Path:
    """A not-yet-existing ``<recordings>/<date>-<time>[-n]`` path."""
    base = recordings_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate, suffix = base, 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def free_gigabytes(path: Path) -> float:
    """Free space on the volume holding ``path``; 0.0 if it cannot be determined."""
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def format_roi(roi: Roi) -> str:
    return f"x {roi.x:.3f}  y {roi.y:.3f}\nw {roi.width:.3f}  h {roi.height:.3f}"


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: Settings,
        list_targets: Callable[[], list[CaptureTarget]],
        source_factory: SourceFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._list_targets = list_targets
        self._roi = load_roi(settings.paths.state_file) or settings.capture.roi
        self._process = psutil.Process()
        self._process.cpu_percent(None)  # prime: the first call always returns 0

        self._controller = CaptureController(source_factory, self)
        self._saving: FrameRecorder | None = None  # recording still flushing to disk
        self._saving_timer = QTimer(self)
        self._saving_timer.timeout.connect(self._check_saving)
        # Always queued: the handler stops the source, which must never happen
        # inside the source's own error callback.
        self._controller.failed.connect(self._on_capture_failed, Qt.ConnectionType.QueuedConnection)

        self.setWindowTitle("WoW Fishing Research")
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh_targets()
        self._update_controls()

        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._show_latest_frame)
        self._display_timer.start(_DISPLAY_INTERVAL_MS)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(_STATS_INTERVAL_MS)

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self) -> None:
        self.video = VideoWidget()
        self.video.roi_selected.connect(self._on_roi_selected)

        self.target_box = QComboBox()
        self.refresh_button = QPushButton("重新整理")
        self.refresh_button.clicked.connect(self._refresh_targets)
        self.fps_box = QSpinBox()
        self.fps_box.setRange(1, 120)
        self.fps_box.setValue(self._settings.capture.fps)
        self.fps_box.setSuffix(" FPS")
        self.start_button = QPushButton("開始研究")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self._stop)

        source_group = QGroupBox("擷取來源")
        source_layout = QVBoxLayout(source_group)
        source_layout.addWidget(self.target_box)
        row = QHBoxLayout()
        row.addWidget(self.fps_box)
        row.addWidget(self.refresh_button)
        source_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        source_layout.addLayout(row)

        self.select_roi_button = QPushButton("框選 ROI")
        self.select_roi_button.setCheckable(True)
        self.select_roi_button.toggled.connect(self.video.set_selecting)
        self.reset_roi_button = QPushButton("顯示全畫面")
        self.reset_roi_button.clicked.connect(lambda: self._apply_roi(Roi.full()))
        self.roi_label = self._value_label()

        roi_group = QGroupBox("ROI")
        roi_layout = QVBoxLayout(roi_group)
        row = QHBoxLayout()
        row.addWidget(self.select_roi_button)
        row.addWidget(self.reset_roi_button)
        roi_layout.addLayout(row)
        roi_layout.addWidget(self.roi_label)

        self.record_button = QPushButton("開始錄製")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self._toggle_recording)
        self.record_label = self._value_label()
        self.record_label.setWordWrap(True)
        record_group = QGroupBox("錄製樣本")
        record_layout = QVBoxLayout(record_group)
        record_layout.addWidget(self.record_button)
        record_layout.addWidget(self.record_label)

        self.fps_label = self._value_label()
        self.latency_label = self._value_label()
        self.size_label = self._value_label()
        self.dropped_label = self._value_label()
        self.cpu_label = self._value_label()
        stats_group = QGroupBox("統計")
        stats_layout = QFormLayout(stats_group)
        stats_layout.addRow("FPS", self.fps_label)
        stats_layout.addRow("擷取延遲", self.latency_label)
        stats_layout.addRow("畫面大小", self.size_label)
        stats_layout.addRow("未顯示幀", self.dropped_label)
        stats_layout.addRow("CPU", self.cpu_label)

        self.error_label = QLabel()
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)

        side = QVBoxLayout()
        side.addWidget(source_group)
        side.addWidget(roi_group)
        side.addWidget(record_group)
        side.addWidget(stats_group)
        side.addWidget(self.error_label)
        side.addStretch(1)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(280)

        layout = QHBoxLayout()
        layout.addWidget(self.video, 1)
        layout.addWidget(side_widget)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1180, 680)

    @staticmethod
    def _value_label() -> QLabel:
        label = QLabel("—")
        label.setObjectName("value")
        return label

    # ------------------------------------------------------------------ actions

    def _refresh_targets(self) -> None:
        previous: CaptureTarget | None = self.target_box.currentData()
        self.target_box.clear()
        try:
            targets = self._list_targets()
        except CaptureError as exc:
            self._show_error(str(exc))
            self._update_controls()
            return
        for target in targets:
            self.target_box.addItem(target.label, target)
        # Keep the current selection across a refresh; identity is (kind, id)
        # because labels and sizes change with the window title and geometry.
        wanted = None
        if previous is not None:
            key = (previous.kind, previous.ident)
            wanted = next((t for t in targets if (t.kind, t.ident) == key), None)
        if wanted is None:
            capture = self._settings.capture
            try:
                wanted = choose_target(targets, capture.display_id, capture.window_owner)
            except CaptureError as exc:
                self._show_error(str(exc))
        if wanted is not None:
            self.target_box.setCurrentIndex(targets.index(wanted))
        self._update_controls()

    def _start(self) -> None:
        target = self.target_box.currentData()
        if target is None:
            self._show_error("沒有可用的擷取來源")
            return
        try:
            self._controller.start(target, self.fps_box.value(), self._roi)
        except CaptureError as exc:
            self._show_error(str(exc))
        else:
            self._show_error("")
        self._update_controls()

    def _stop(self) -> None:
        self._finish_recording()
        self._controller.stop()
        self.select_roi_button.setChecked(False)
        self.video.set_frame(None)
        self._update_controls()

    def _toggle_recording(self, checked: bool) -> None:
        if not checked:
            self._finish_recording()
            return
        target: CaptureTarget | None = self.target_box.currentData()
        meta = {"target": target.label if target else "", "fps": self.fps_box.value()}
        try:
            session = new_session_dir(self._settings.paths.recordings_dir)
            self._controller.start_recording(session, meta)
        except (OSError, ValueError) as exc:
            self.record_button.setChecked(False)
            self._show_error(f"無法開始錄製：{exc}")
            return
        self._show_error("")
        self.record_button.setText("停止錄製")
        self.record_label.setText("錄製中…")

    def _finish_recording(self) -> None:
        """Stop feeding the recorder; it flushes in the background and is polled."""
        recorder = self._controller.detach_recorder()
        self.record_button.setChecked(False)
        if recorder is None:
            return
        self._saving = recorder
        self.record_button.setText("儲存中…")
        self._saving_timer.start(_SAVING_POLL_MS)
        self._check_saving()
        self._update_controls()

    def _check_saving(self) -> None:
        recorder = self._saving
        if recorder is None:
            self._saving_timer.stop()
            return
        if not recorder.finished:
            self.record_label.setText(f"儲存中… 剩 {recorder.pending} 張")
            return
        self._saving = None
        self._saving_timer.stop()
        self.record_button.setText("開始錄製")
        summary = f"已存 {recorder.written} 張"
        if recorder.dropped:
            summary += f"（略過 {recorder.dropped} 張）"
        if recorder.error is not None:
            summary += f"\n錄製中斷：{recorder.error}"
        self.record_label.setText(f"{summary}\n{recorder.session_dir.resolve()}")
        self._update_controls()

    def _on_capture_failed(self, message: str) -> None:
        self._stop()
        self._show_error(message)

    def _on_roi_selected(self, selection: Roi) -> None:
        self.select_roi_button.setChecked(False)
        frame = self.video.frame
        if frame is not None:
            self._apply_roi(frame.roi.sub(selection))

    def _apply_roi(self, roi: Roi) -> None:
        try:
            self._controller.set_roi(roi)
        except (CaptureError, ValueError) as exc:
            self._show_error(f"無法套用 ROI：{exc}")
            return
        self._roi = roi
        self._show_error("")
        try:
            # Saved even when full, so "show everything" survives a restart
            # when the settings file names a smaller default ROI.
            save_roi(self._settings.paths.state_file, roi)
        except OSError as exc:
            self._show_error(f"ROI 已套用但無法儲存：{exc}")
        self._update_controls()

    # ------------------------------------------------------------------ periodic updates

    def _show_latest_frame(self) -> None:
        frame = self._controller.frames.take()
        if frame is not None:
            self.video.set_frame(frame)

    def _update_stats(self) -> None:
        self.cpu_label.setText(f"{self._process.cpu_percent(None):.0f} %")
        recorder = self._controller.recorder
        if recorder is not None:
            if recorder.error is not None:
                self._finish_recording()
            else:
                self.record_label.setText(
                    f"錄製中… {recorder.written} 張（略過 {recorder.dropped}）\n"
                    f"磁碟剩餘 {free_gigabytes(recorder.session_dir):.1f} GB"
                )
        if not self._controller.running:
            for label in (self.fps_label, self.latency_label, self.size_label, self.dropped_label):
                label.setText("—")
            return
        self.fps_label.setText(f"{self._controller.fps.rate():.1f}")
        latency = self._controller.latency_ms.snapshot()
        if latency is not None:
            self.latency_label.setText(f"{latency.mean:.1f} ms（最大 {latency.maximum:.1f}）")
        frame = self.video.frame
        if frame is not None:
            self.size_label.setText(f"{frame.width}×{frame.height} px")
        self.dropped_label.setText(str(self._controller.frames.dropped))

    def _update_controls(self) -> None:
        running = self._controller.running
        self.start_button.setEnabled(not running and self.target_box.count() > 0)
        self.stop_button.setEnabled(running)
        self.target_box.setEnabled(not running)
        self.fps_box.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.select_roi_button.setEnabled(running)
        self.record_button.setEnabled(running and self._saving is None)
        self.reset_roi_button.setEnabled(self._roi != Roi.full())
        self.roi_label.setText(format_roi(self._roi))

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._display_timer.stop()
        self._stats_timer.stop()
        self._saving_timer.stop()
        self._controller.stop()  # blocks until an active recording is on disk
        if self._saving is not None:
            self._saving.stop()  # likewise for one that was already finishing
            self._saving = None
        super().closeEvent(event)
