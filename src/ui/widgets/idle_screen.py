"""Ambient idle screen: large clock, date, slim system strip at bottom."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme

_RECIPE_STEP_FONT_SIZE = 36    # single-step display
_RECIPE_ALL_FONT_SIZE = 22     # all-steps display (≤ 5 steps)
_RECIPE_NUM_FONT_SIZE = 24     # "Step N of M" label


class IdleScreen(QWidget):
    """Full-screen ambient display shown when the assistant is IDLE.

    Center area uses a QStackedWidget:
      Page 0 — clock + date (default)
      Page 1 — recipe step display (Active Recipe)

    Layout (800×480):
      ┌─────────────────────────────────────┐
      │                                     │  ↑ flex spacer
      │         [clock or recipe step]      │  ← center (stacked)
      │                                     │  ↓ flex spacer
      ├─────────────────────────────────────┤
      │  ● IDLE    47°C  CPU 12%  3.6GB RAM │  ← 52px bottom strip
      └─────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Center: stacked clock / recipe ───────────────────────────────────
        self._center = QStackedWidget(self)

        # Page 0: clock + date
        clock_panel = QWidget()
        clock_layout = QVBoxLayout(clock_panel)
        clock_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clock_layout.setSpacing(6)

        self._clock_label = QLabel()
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clock_label.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY};font-size: {theme.IDLE_CLOCK_FONT_SIZE}px;"
        )

        self._date_label = QLabel()
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._date_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.IDLE_DATE_FONT_SIZE}px;"
        )

        clock_layout.addWidget(self._clock_label)
        clock_layout.addWidget(self._date_label)
        self._center.addWidget(clock_panel)

        # Page 1: recipe step display
        recipe_panel = QWidget()
        recipe_layout = QVBoxLayout(recipe_panel)
        recipe_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recipe_layout.setSpacing(8)
        recipe_layout.setContentsMargins(24, 16, 24, 16)

        self._step_num_label = QLabel()
        self._step_num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_num_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {_RECIPE_NUM_FONT_SIZE}px;"
        )

        self._step_label = QLabel()
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setWordWrap(True)
        self._step_label.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {_RECIPE_STEP_FONT_SIZE}px;"
        )

        recipe_layout.addWidget(self._step_num_label)
        recipe_layout.addWidget(self._step_label)
        self._center.addWidget(recipe_panel)

        self._center.setCurrentIndex(0)

        # ── Bottom strip: state pill + sys stats ─────────────────────────────
        bottom = QWidget(self)
        bottom.setFixedHeight(52)
        bottom.setStyleSheet(f"background-color: {theme.BG_ELEVATED};")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(12, 0, 12, 0)
        bottom_layout.setSpacing(16)

        self._state_label = QLabel("● IDLE", self)
        idle_fg = theme.STATE_COLORS["IDLE"][1]
        self._state_label.setStyleSheet(f"color: {idle_fg}; font-size: 18px; font-weight: bold;")

        self._radio_label = QLabel("", self)
        self._radio_label.setStyleSheet(
            "color: #26C6DA; font-size: 18px; font-family: 'DejaVu Sans';"
        )
        self._radio_label.setVisible(False)

        self._temp_label = QLabel("", self)
        self._temp_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._temp_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
        )

        self._cpu_label = QLabel("", self)
        self._cpu_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._cpu_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
        )

        self._ram_label = QLabel("", self)
        self._ram_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._ram_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
        )

        bottom_layout.addWidget(self._state_label)
        bottom_layout.addWidget(self._radio_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self._temp_label)
        bottom_layout.addWidget(self._cpu_label)
        bottom_layout.addWidget(self._ram_label)

        outer.addWidget(self._center, stretch=1)
        outer.addWidget(bottom)

        # ── Clock timer (every 60 seconds) ───────────────────────────────────
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(60_000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()
        self._update_clock()

    # ── Public slots ─────────────────────────────────────────────────────────

    def update_resources(
        self,
        cpu_pct: float,
        ram_used_mb: float,
        ram_total_mb: float,
        temp: float | None,
    ):
        """Update the bottom strip with current CPU, RAM, and temperature."""
        if temp is not None:
            color = theme.temp_color(temp)
            self._temp_label.setText(f"{temp:.0f}°C")
            self._temp_label.setStyleSheet(
                f"color: {color}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
            )

        cpu_color = theme.cpu_color(cpu_pct)
        self._cpu_label.setText(f"CPU {cpu_pct:.0f}%")
        self._cpu_label.setStyleSheet(
            f"color: {cpu_color}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
        )

        ram_pct = (ram_used_mb / ram_total_mb * 100) if ram_total_mb > 0 else 0
        ram_color = theme.cpu_color(ram_pct)
        self._ram_label.setText(f"{ram_used_mb / 1024:.1f}GB RAM")
        self._ram_label.setStyleSheet(
            f"color: {ram_color}; font-size: 18px;font-family: 'DejaVu Sans Mono';"
        )

    def on_radio_changed(self, station: str):
        """Show or hide the radio indicator in the bottom strip."""
        if station:
            self._radio_label.setText(f"♪ {station}")
            self._radio_label.setVisible(True)
        else:
            self._radio_label.setVisible(False)

    def on_state_changed(self, state_name: str):
        """Update bottom strip state pill (e.g. SESSION_CHECK before going active)."""
        _, fg = theme.STATE_COLORS.get(state_name, theme.STATE_COLORS["IDLE"])
        self._state_label.setText(f"● {state_name}")
        self._state_label.setStyleSheet(f"color: {fg}; font-size: 18px; font-weight: bold;")

    def on_recipe_step(self, step_text: str, step_num: int, total_steps: int):
        """Show recipe step display or return to clock.

        step_num == 0 → step_text is all steps (recipe has ≤ 5 steps total)
        step_num > 0  → step_text is one step
        step_text ""  → clear and return to clock
        """
        if not step_text:
            self._center.setCurrentIndex(0)
            return

        if step_num > 0:
            self._step_num_label.setText(f"Step {step_num} of {total_steps}")
            self._step_num_label.setVisible(True)
            self._step_label.setStyleSheet(
                f"color: {theme.TEXT_PRIMARY}; font-size: {_RECIPE_STEP_FONT_SIZE}px;"
            )
        else:
            self._step_num_label.setVisible(False)
            self._step_label.setStyleSheet(
                f"color: {theme.TEXT_PRIMARY}; font-size: {_RECIPE_ALL_FONT_SIZE}px;"
            )

        self._step_label.setText(step_text)
        self._center.setCurrentIndex(1)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_clock(self):
        now = datetime.now()
        self._clock_label.setText(now.strftime("%H:%M"))
        self._date_label.setText(now.strftime("%A, ") + str(now.day) + now.strftime(" %B"))
