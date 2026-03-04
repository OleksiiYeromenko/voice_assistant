"""Ambient idle screen: large clock, date, slim system strip at bottom."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.ui import theme


class IdleScreen(QWidget):
    """Full-screen ambient display shown when the assistant is IDLE.

    Layout (800×480):
      ┌─────────────────────────────────────┐
      │                                     │  ↑ flex spacer
      │              18:25                  │  ← 120px clock
      │         Tuesday, 4 March            │  ← 28px date
      │                                     │  ↓ flex spacer
      ├─────────────────────────────────────┤
      │  ● IDLE              47°C  CPU 12%  │  ← 40px bottom strip
      └─────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Center: clock + date ─────────────────────────────────────────────
        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.setSpacing(6)

        self._clock_label = QLabel(self)
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clock_label.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY};"
            f"font-size: {theme.IDLE_CLOCK_FONT_SIZE}px;"
        )

        self._date_label = QLabel(self)
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._date_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.IDLE_DATE_FONT_SIZE}px;"
        )

        center_layout.addWidget(self._clock_label)
        center_layout.addWidget(self._date_label)

        # ── Bottom strip: state pill + sys stats ─────────────────────────────
        bottom = QWidget(self)
        bottom.setFixedHeight(40)
        bottom.setStyleSheet(f"background-color: {theme.BG_ELEVATED};")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(12, 0, 12, 0)
        bottom_layout.setSpacing(16)

        self._state_label = QLabel("● IDLE", self)
        idle_fg = theme.STATE_COLORS["IDLE"][1]
        self._state_label.setStyleSheet(
            f"color: {idle_fg}; font-size: 13px; font-weight: bold;"
        )

        self._temp_label = QLabel("", self)
        self._temp_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._temp_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 13px;"
            f"font-family: 'DejaVu Sans Mono';"
        )

        self._cpu_label = QLabel("", self)
        self._cpu_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._cpu_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 13px;"
            f"font-family: 'DejaVu Sans Mono';"
        )

        bottom_layout.addWidget(self._state_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self._temp_label)
        bottom_layout.addWidget(self._cpu_label)

        outer.addWidget(center, stretch=1)
        outer.addWidget(bottom)

        # ── Clock timer (every 60 seconds) ───────────────────────────────────
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(60_000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()
        self._update_clock()

    # ── Public slots ─────────────────────────────────────────────────────────

    def update_resources(self, cpu_pct: float, temp: float | None):
        """Update the bottom strip with current CPU and temperature."""
        if temp is not None:
            color = theme.temp_color(temp)
            self._temp_label.setText(f"{temp:.0f}°C")
            self._temp_label.setStyleSheet(
                f"color: {color}; font-size: 13px;"
                f"font-family: 'DejaVu Sans Mono';"
            )

        cpu_color = theme.cpu_color(cpu_pct)
        self._cpu_label.setText(f"CPU {cpu_pct:.0f}%")
        self._cpu_label.setStyleSheet(
            f"color: {cpu_color}; font-size: 13px;"
            f"font-family: 'DejaVu Sans Mono';"
        )

    def on_state_changed(self, state_name: str):
        """Update bottom strip state pill (e.g. SESSION_CHECK before going active)."""
        _, fg = theme.STATE_COLORS.get(state_name, theme.STATE_COLORS["IDLE"])
        self._state_label.setText(f"● {state_name}")
        self._state_label.setStyleSheet(
            f"color: {fg}; font-size: 13px; font-weight: bold;"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_clock(self):
        now = datetime.now()
        self._clock_label.setText(now.strftime("%H:%M"))
        self._date_label.setText(
            now.strftime("%A, ") + str(now.day) + now.strftime(" %B")
        )
