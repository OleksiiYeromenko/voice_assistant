"""Tool status strip — shown when a tool call is in progress or just finished."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme

_SPINNER_FRAMES = (".", "..", "...")
_SPINNER_INTERVAL_MS = 450
_AUTO_HIDE_MS = 3000


class ToolStrip(QWidget):
    """One-line strip displayed below the chat view during tool execution.

    Hidden when IDLE. Shows tool name + animated spinner while executing,
    then turns green on completion and auto-hides after 3 seconds.
    """

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setObjectName("tool_strip")
        self.setStyleSheet(
            f"#tool_strip {{ background-color: {theme.BG_ELEVATED}; "
            f"border-top: 1px solid #2A2A45; border-bottom: 1px solid #2A2A45; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(6)

        # Wrench icon
        self._icon = QLabel("⚙", self)
        self._icon.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 14px;")

        # Tool name
        self._name = QLabel("", self)
        self._name.setStyleSheet(
            f"color: #FFB300; font-family: 'DejaVu Sans Mono'; font-size: 13px;"
        )

        # Bullet separator
        self._sep = QLabel("•", self)
        self._sep.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 13px;")

        # Status / spinner
        self._status = QLabel("", self)
        self._status.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-family: 'DejaVu Sans Mono'; font-size: 13px;"
        )

        layout.addWidget(self._icon)
        layout.addWidget(self._name)
        layout.addWidget(self._sep)
        layout.addWidget(self._status)
        layout.addStretch()

        # Spinner timer
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_INTERVAL_MS)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frame = 0

        # Auto-hide timer
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_AUTO_HIDE_MS)
        self._hide_timer.timeout.connect(self.hide)

        self.hide()
        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.tool_started.connect(self._on_tool_started)
        bus.tool_done.connect(self._on_tool_done)

    def _on_tool_started(self, tool_name: str):
        self._hide_timer.stop()
        self._name.setText(tool_name)
        self._status.setText("EXECUTING")
        self._status.setStyleSheet(
            f"color: #90CAF9; font-family: 'DejaVu Sans Mono'; font-size: 13px;"
        )
        self._spinner_frame = 0
        self._spinner_timer.start()
        self.show()

    def _on_tool_done(self, tool_name: str, result: str):
        self._spinner_timer.stop()
        self._name.setText(tool_name)
        short = result[:60] + "…" if len(result) > 60 else result
        self._status.setText(f"DONE  {short}" if short else "DONE")
        self._status.setStyleSheet(
            f"color: #69F0AE; font-family: 'DejaVu Sans Mono'; font-size: 13px;"
        )
        self._hide_timer.start()

    def _tick_spinner(self):
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        dots = _SPINNER_FRAMES[self._spinner_frame]
        self._status.setText(f"EXECUTING{dots}")
