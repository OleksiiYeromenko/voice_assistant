"""Retro-theme idle screen: pixel clock, ASCII bars, CRT scanlines."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme as T

_FONT = "DejaVu Sans Mono"
_RECIPE_FONT_PX = 26
_RECIPE_NUM_FONT_PX = 18


def _px(n: int) -> str:
    return f"{n}px"


class ScanlineOverlay(QWidget):
    """Transparent overlay that paints heavy CRT scanlines (every other row)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, _event):
        p = QPainter(self)
        dark = QColor(0, 0, 0, 90)
        y = 1
        while y < self.height():
            p.fillRect(0, y, self.width(), 1, dark)
            y += 2
        p.end()


def _ascii_bar(pct: float, length: int = 12) -> str:
    filled = round(pct / 100 * length)
    return "█" * filled + "░" * (length - filled)


def _metric_color(val: float, warn: float, crit: float) -> str:
    if val >= crit:
        return "#ff4444"
    if val >= warn:
        return "#ffcc00"
    return T.RETRO_TEXT_PRIMARY


class RetroIdleScreen(QWidget):
    """Retro idle screen — replaces IdleScreen when display.theme == 'retro'.

    Public interface is identical to IdleScreen:
      update_resources(), on_radio_changed(), on_state_changed(), on_recipe_step()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {T.RETRO_BG};")

        self._cpu: float = 0.0
        self._temp: float | None = None
        self._ram_used: float = 0.0
        self._ram_total: float = 1.0
        self._blink = True
        self._radio_station = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(
            f"background-color: {T.RETRO_ELEVATED}; "
            f"border-bottom: 2px solid {T.RETRO_BORDER};"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)

        self._sys_title = QLabel("[ POONDYK.SYS ]")
        self._sys_title.setStyleSheet(
            f"color: {T.RETRO_ACCENT}; font-family: '{_FONT}'; "
            f"font-size: 18px; letter-spacing: 3px;"
        )
        self._date_label = QLabel()
        self._date_label.setStyleSheet(
            f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
            f"font-size: 16px; letter-spacing: 2px;"
        )
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        h_layout.addWidget(self._sys_title)
        h_layout.addStretch()
        h_layout.addWidget(self._date_label)

        # ── Center stacked: clock/recipe ────────────────────────────────────
        self._center = QStackedWidget()

        # Page 0: clock + status block
        clock_panel = QWidget()
        clock_panel.setStyleSheet(f"background-color: {T.RETRO_BG};")
        cp_layout = QVBoxLayout(clock_panel)
        cp_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cp_layout.setSpacing(24)

        self._clock_label = QLabel()
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clock_label.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
            f"font-size: 176px; font-weight: bold; letter-spacing: 12px;"
        )

        # Status block (PixelBox equivalent)
        status_box = QFrame()
        status_box.setStyleSheet(
            f"QFrame {{ background-color: #040414; border: 2px solid {T.RETRO_BORDER}; }}"
        )
        sb_layout = QVBoxLayout(status_box)
        sb_layout.setContentsMargins(36, 14, 36, 14)
        sb_layout.setSpacing(12)

        sys_title_lbl = QLabel("SYSTEM STATUS")
        sys_title_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
            f"font-size: 13px; letter-spacing: 4px; border: none;"
        )
        sb_layout.addWidget(sys_title_lbl)

        self._cpu_bar_lbl = QLabel()
        self._ram_bar_lbl = QLabel()
        self._temp_bar_lbl = QLabel()

        for lbl in (self._cpu_bar_lbl, self._ram_bar_lbl, self._temp_bar_lbl):
            lbl.setStyleSheet(
                f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
                f"font-size: 20px; letter-spacing: 2px; border: none;"
            )
            sb_layout.addWidget(lbl)

        # Prompt
        self._prompt_lbl = QLabel()
        self._prompt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prompt_lbl.setStyleSheet(
            f"color: {T.RETRO_ACCENT}; font-family: '{_FONT}'; "
            f"font-size: 22px; letter-spacing: 4px;"
        )

        cp_layout.addWidget(self._clock_label)
        cp_layout.addWidget(status_box, 0, Qt.AlignmentFlag.AlignHCenter)
        cp_layout.addWidget(self._prompt_lbl)
        self._center.addWidget(clock_panel)

        # Page 1: recipe step
        recipe_panel = QWidget()
        recipe_panel.setStyleSheet(f"background-color: {T.RETRO_BG};")
        rp_layout = QVBoxLayout(recipe_panel)
        rp_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rp_layout.setSpacing(8)
        rp_layout.setContentsMargins(24, 16, 24, 16)

        self._step_num_label = QLabel()
        self._step_num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_num_label.setStyleSheet(
            f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
            f"font-size: {_RECIPE_NUM_FONT_PX}px; letter-spacing: 2px;"
        )

        self._step_label = QLabel()
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setWordWrap(True)
        self._step_label.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
            f"font-size: {_RECIPE_FONT_PX}px;"
        )
        rp_layout.addWidget(self._step_num_label)
        rp_layout.addWidget(self._step_label)
        self._center.addWidget(recipe_panel)

        self._center.setCurrentIndex(0)

        # ── Bottom bar ───────────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(44)
        bottom.setStyleSheet(
            f"background-color: {T.RETRO_ELEVATED}; "
            f"border-top: 2px solid {T.RETRO_BORDER};"
        )
        bot_layout = QHBoxLayout(bottom)
        bot_layout.setContentsMargins(20, 0, 20, 0)
        bot_layout.setSpacing(0)

        _s = (
            f"color: {T.RETRO_TEXT_SECONDARY}; font-family: '{_FONT}'; "
            "font-size: 15px; letter-spacing: 1px;"
        )
        self._bot_cpu = QLabel()
        self._bot_cpu.setStyleSheet(_s)
        self._bot_temp = QLabel()
        self._bot_temp.setStyleSheet(_s)
        self._bot_ram = QLabel()
        self._bot_ram.setStyleSheet(_s)
        self._bot_radio = QLabel()
        self._bot_radio.setStyleSheet(
            f"color: #4fc3f7; font-family: '{_FONT}'; font-size: 15px; letter-spacing: 1px;"
        )
        self._bot_radio.setVisible(False)

        self._bot_state = QLabel()
        self._bot_state.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
            f"font-size: 15px; letter-spacing: 2px;"
        )

        def _pipe():
            lbl = QLabel("│")
            lbl.setStyleSheet(
                f"color: {T.RETRO_BORDER}; font-family: '{_FONT}'; "
                "font-size: 15px; margin: 0 10px;"
            )
            return lbl

        bot_layout.addWidget(self._bot_cpu)
        bot_layout.addWidget(_pipe())
        bot_layout.addWidget(self._bot_temp)
        bot_layout.addWidget(_pipe())
        bot_layout.addWidget(self._bot_ram)
        bot_layout.addWidget(self._bot_radio)
        bot_layout.addStretch()
        bot_layout.addWidget(_pipe())
        bot_layout.addWidget(self._bot_state)

        outer.addWidget(header)
        outer.addWidget(self._center, stretch=1)
        outer.addWidget(bottom)

        # ── Scanline overlay (added last so it sits on top) ──────────────────
        self._scanlines = ScanlineOverlay(self)

        # ── Timers ───────────────────────────────────────────────────────────
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._tick_blink)
        self._blink_timer.start()

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1_000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()

        self._update_clock()
        self._update_bars()
        self._update_state_dot("IDLE")

    # ── Layout ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scanlines.setGeometry(self.rect())
        self._scanlines.raise_()

    # ── Public slots (same interface as IdleScreen) ──────────────────────────

    def update_resources(
        self,
        cpu_pct: float,
        ram_used_mb: float,
        ram_total_mb: float,
        temp: float | None,
    ):
        self._cpu = cpu_pct
        self._ram_used = ram_used_mb
        self._ram_total = max(ram_total_mb, 1.0)
        self._temp = temp
        self._update_bars()

    def on_radio_changed(self, station: str):
        self._radio_station = station
        if station:
            self._bot_radio.setText(f"  │  ♪ {station[:22]}")
            self._bot_radio.setVisible(True)
        else:
            self._bot_radio.setVisible(False)

    def on_state_changed(self, state_name: str):
        self._update_state_dot(state_name)

    def on_recipe_step(self, step_text: str, step_num: int, total_steps: int):
        if not step_text:
            self._center.setCurrentIndex(0)
            return
        if step_num > 0:
            self._step_num_label.setText(f"STEP {step_num} OF {total_steps}:")
            self._step_num_label.setVisible(True)
            self._step_label.setStyleSheet(
                f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
                f"font-size: {_RECIPE_FONT_PX}px;"
            )
        else:
            self._step_num_label.setVisible(False)
            self._step_label.setStyleSheet(
                f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; font-size: 22px;"
            )
        self._step_label.setText(step_text)
        self._center.setCurrentIndex(1)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _tick_blink(self):
        self._blink = not self._blink
        hh = datetime.now().strftime("%H")
        mm = datetime.now().strftime("%M")
        colon = ":" if self._blink else " "
        self._clock_label.setText(f"{hh}{colon}{mm}")

        cursor_on = "█" if self._blink else " "
        cursor_off = " " if self._blink else "█"
        self._prompt_lbl.setText(f"{cursor_on} AWAITING INPUT {cursor_off}")

        dot = "■" if self._blink else "□"
        self._bot_state.setText(f"{dot} IDLE")

    def _update_clock(self):
        now = datetime.now()
        date_str = now.strftime("%a, %d %b %Y").upper()
        self._date_label.setText(date_str)

    def _update_bars(self):
        cpu_c = _metric_color(self._cpu, 65, 85)
        temp_c = _metric_color(self._temp or 0, 60, 72)
        ram_pct = self._ram_used / self._ram_total * 100
        ram_c = T.RETRO_TEXT_PRIMARY

        def _row(label: str, pct: float, unit: str, color: str) -> str:
            return f"{label} {_ascii_bar(pct)}  {pct:.0f}{unit}"

        self._cpu_bar_lbl.setText(_row("CPU ", self._cpu, "%", cpu_c))
        def _bar_style(color: str) -> str:
            return (
                f"color: {color}; font-family: '{_FONT}'; "
                "font-size: 20px; letter-spacing: 2px; border: none;"
            )

        self._cpu_bar_lbl.setStyleSheet(_bar_style(cpu_c))
        temp_pct = min(self._temp or 0, 100)
        self._temp_bar_lbl.setText(_row("TEMP", temp_pct, "C", temp_c))
        self._temp_bar_lbl.setStyleSheet(_bar_style(temp_c))
        self._ram_bar_lbl.setText(_row("RAM ", ram_pct, "%", ram_c))
        self._ram_bar_lbl.setStyleSheet(_bar_style(ram_c))

        self._bot_cpu.setText(f"CPU {self._cpu:.0f}%")
        self._bot_temp.setText(f"TEMP {self._temp:.0f}°C" if self._temp is not None else "TEMP —")
        self._bot_ram.setText(f"RAM {ram_pct:.0f}%")

    def _update_state_dot(self, state_name: str):
        _, fg, lbl = T.RETRO_STATE.get(state_name, ("", T.RETRO_TEXT_PRIMARY, state_name))
        self._bot_state.setStyleSheet(
            f"color: {fg}; font-family: '{_FONT}'; font-size: 15px; letter-spacing: 2px;"
        )
        self._bot_state.setText(f"■ {lbl}")
