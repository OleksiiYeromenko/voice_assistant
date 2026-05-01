"""Retro-theme idle screen: pixel clock, two-column system status, CRT scanlines.

Spec: docs/design/design_handoff_retro_ui/README.md
"""
from __future__ import annotations

import socket
import time
from datetime import datetime

import psutil
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme as T
from src.ui.backend_monitor import BackendMonitor, BackendStatus

_FONT = "Press Start 2P"
_FB = "DejaVu Sans Mono"  # fallback

# Font pixel sizes (spec px → readable Qt px, scaled ~1.4× for RPi 133dpi display)
_FZ_TINY = 12    # spec 6px  — sub-model, badge
_FZ_SMALL = 14   # spec 7px  — labels, ASCII bars, status text
_FZ_BODY = 16    # spec 8px  — header, prompt text
_FZ_MED = 13     # spec 9px  — time in state bar
_FZ_RESP = 15    # spec 11px — response text
_FZ_CLOCK = 100  # spec 64px — clock (scaled for RPi physical display)

_HEADER_H = 44    # was 40 hardcoded
_BOTTOM_H = 36    # was 32 hardcoded
_BOX_W = 600      # was 480 hardcoded

_RECIPE_FONT_PX = 22
_RECIPE_NUM_FONT_PX = 14


def _ss(color: str, size: int, spacing: int = 1, bold: bool = False) -> str:
    """Build a stylesheet string for retro text labels."""
    w = "bold" if bold else "normal"
    return (
        f"color: {color}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {size}px; letter-spacing: {spacing}px; font-weight: {w}; "
        f"background: transparent; border: none;"
    )


def _get_local_ip() -> str:
    try:
        for _iface, addr_list in psutil.net_if_addrs().items():
            for addr in addr_list:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    return addr.address
    except Exception:
        pass
    return "---"


def _get_uptime() -> str:
    try:
        elapsed = time.time() - psutil.boot_time()
        h = int(elapsed // 3600)
        return f"{h:02d}H"
    except Exception:
        return "--H"


def _metric_color(val: float, warn: float, crit: float) -> str:
    if val >= crit:
        return "#ff4444"
    if val >= warn:
        return "#ffcc00"
    return T.RETRO_TEXT_PRIMARY


# ── Scanline overlay ─────────────────────────────────────────────────────────

class ScanlineOverlay(QWidget):
    """Transparent overlay painting horizontal CRT scanlines every 2px."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, _event):
        p = QPainter(self)
        # rgba(0,0,0,0.35) → alpha ≈ 89
        dark = QColor(0, 0, 0, 120)
        y = 1
        while y < self.height():
            p.fillRect(0, y, self.width(), 1, dark)
            y += 2
        p.end()


# ── Clock with purple CRT chromatic-aberration glow ─────────────────────────

class CRTClockLabel(QLabel):
    """Draws purple offset shadow layers underneath the standard cyan clock text."""

    def paintEvent(self, event):
        font = QFont(self.font())
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8.0)
        rect = self.rect()
        align = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, False)
        p.setFont(font)
        # Wide purple halo — multi-radius blur simulation
        for r, alpha in ((10, 8), (7, 14), (5, 20), (3, 28)):
            h2 = max(1, r // 2)
            for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r),
                           (-r, -h2), (r, -h2), (-r, h2), (r, h2)):
                p.setPen(QColor(140, 0, 255, alpha))
                p.drawText(rect.translated(dx, dy), align, self.text())
        # Outer cyan glow ring — simulate 0 0 40px #00ffcc30
        for dx, dy in ((-5, 0), (5, 0), (0, -5), (0, 5), (-4, -4), (4, -4), (-4, 4), (4, 4)):
            p.setPen(QColor(0, 255, 204, 50))
            p.drawText(rect.translated(dx, dy), align, self.text())
        # Inner cyan glow ring — simulate 0 0 20px #00ffcc80
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1),
                       (-2, -2), (2, -2), (-2, 2), (2, 2)):
            p.setPen(QColor(0, 255, 204, 100))
            p.drawText(rect.translated(dx, dy), align, self.text())
        # Chromatic aberration — right+down diagonal spread (no left-side offset)
        p.setPen(QColor(170, 0, 255, 220))
        p.drawText(rect.translated(3, 2), align, self.text())
        p.setPen(QColor(140, 0, 255, 120))
        p.drawText(rect.translated(5, 3), align, self.text())
        p.setPen(QColor(110, 0, 255, 60))
        p.drawText(rect.translated(7, 4), align, self.text())
        p.end()
        super().paintEvent(event)  # sharp cyan text on top


# ── Status box widgets ───────────────────────────────────────────────────────

class HardwareRow(QWidget):
    """LABEL [filled▓▓░░░░░] VALUE  — one CPU/TEMP/RAM row."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 8)
        lay.setSpacing(6)

        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(72)
        self._lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_SMALL, 2))

        # Single label with HTML spans for lit/dim colors
        self._bar_lbl = QLabel()
        self._bar_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._bar_lbl.setStyleSheet(
            f"background: transparent; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ_SMALL}px; letter-spacing: 2px;"
        )

        self._val = QLabel()
        self._val.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_SMALL, 1))

        lay.addWidget(self._lbl)
        lay.addWidget(self._bar_lbl)
        lay.addSpacing(8)
        lay.addWidget(self._val)

    def update_metric(self, pct: float, value_str: str, color: str):
        length = 10
        filled = max(0, min(length, round(pct / 100 * length)))
        lit = "█" * filled
        dim = "█" * (length - filled)
        self._bar_lbl.setText(
            f'<span style="color:{color};">{lit}</span>'
            f'<span style="color:{T.RETRO_TEXT_MUTED};">{dim}</span>'
        )
        self._val.setText(value_str)
        self._val.setStyleSheet(_ss(color, _FZ_SMALL, 1))


class BackendRow(QWidget):
    """LABEL [●] ONLINE/OFFLIN  [ACT]"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 8)
        lay.setSpacing(6)

        self._lbl = QLabel(label)
        self._lbl.setFixedWidth(92)
        self._lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_SMALL, 1))

        self._dot = QLabel("■")
        self._dot.setStyleSheet(_ss("#ff4444", _FZ_SMALL - 3, 0))

        self._status = QLabel("OFFLN")
        self._status.setFixedWidth(92)
        self._status.setStyleSheet(_ss("#ff4444", _FZ_SMALL, 1))

        self._act = QLabel("ACT")
        self._act.setStyleSheet(
            f"color: {T.RETRO_ACCENT}; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ_TINY}px; letter-spacing: 1px; "
            f"border: 1px solid {T.RETRO_ACCENT}; padding: 1px 5px; background: transparent;"
        )
        glow = QGraphicsDropShadowEffect()
        glow.setBlurRadius(5)
        glow.setColor(QColor(255, 68, 221, 80))
        glow.setOffset(0, 0)
        self._act.setGraphicsEffect(glow)
        self._act.setVisible(False)

        lay.addWidget(self._lbl)
        lay.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self._status)
        lay.addStretch(1)
        lay.addWidget(self._act)

    def set_status(self, online: bool, sub_model: str, active: bool):
        color = "#00ff66" if online else "#ff4444"
        self._dot.setStyleSheet(_ss(color, _FZ_SMALL - 3, 0))
        self._status.setText("ONLN" if online else "OFFLN")
        self._status.setStyleSheet(_ss(color, _FZ_SMALL, 1))
        self._act.setVisible(active)


# ── Main idle screen ─────────────────────────────────────────────────────────

class RetroIdleScreen(QWidget):
    """Retro idle screen. Public interface matches IdleScreen:
      update_resources(), on_radio_changed(), on_state_changed(),
      on_recipe_step(), on_model_changed()
    """

    def __init__(self, cfg: dict | None = None, parent=None):
        super().__init__(parent)
        cfg = cfg or {}
        self.setStyleSheet(f"background-color: {T.RETRO_BG};")

        self._cpu: float = 0.0
        self._temp: float | None = None
        self._ram_used: float = 0.0
        self._ram_total: float = 1.0
        self._blink = True
        self._radio_blink = True
        self._radio_station = ""
        self._net_online = False
        self._ip = _get_local_ip()
        self._state_name = "IDLE"
        self._active_backend = "local"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar (h:44) ────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet(
            f"background-color: {T.RETRO_ELEVATED}; "
            f"border-bottom: 2px solid {T.RETRO_BORDER};"
        )
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        sys_title = QLabel("[ POONDYK.SYS ]")
        sys_title.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_BODY, 2))

        self._date_lbl = QLabel()
        self._date_lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_SMALL, 1))
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        h_lay.addWidget(sys_title)
        h_lay.addStretch()
        h_lay.addWidget(self._date_lbl)

        # ── Center stacked area ──────────────────────────────────────────────
        self._center = QStackedWidget()

        # Page 0: clock + status
        clock_panel = QWidget()
        clock_panel.setStyleSheet(f"background-color: {T.RETRO_BG};")
        cp_lay = QVBoxLayout(clock_panel)
        cp_lay.setSpacing(0)
        cp_lay.setContentsMargins(0, 0, 0, 0)

        # Big clock (88px, with cyan glow + purple chromatic aberration)
        self._clock_lbl = CRTClockLabel("00:00")
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clock_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ_CLOCK}px; letter-spacing: 8px; background: transparent;"
        )
        _clock_glow = QGraphicsDropShadowEffect()
        _clock_glow.setBlurRadius(40)
        _clock_glow.setColor(QColor(0, 255, 204, 120))
        _clock_glow.setOffset(0, 0)
        self._clock_lbl.setGraphicsEffect(_clock_glow)
        # Status box wrapper (w:660, centered)
        box_wrapper = QWidget()
        box_wrapper.setFixedWidth(660)
        box_wrapper.setStyleSheet("background: transparent;")
        bw_lay = QVBoxLayout(box_wrapper)
        bw_lay.setContentsMargins(0, 0, 0, 0)
        bw_lay.setSpacing(0)

        # Status box (fully bordered; prompt row is a separate box below)
        status_box = QFrame()
        status_box.setObjectName("statusBox")
        status_box.setStyleSheet(
            f"#statusBox {{ background-color: #040414; "
            f"border: 2px solid {T.RETRO_BORDER}; }}"
        )
        sb_lay = QVBoxLayout(status_box)
        sb_lay.setContentsMargins(24, 20, 24, 18)
        sb_lay.setSpacing(12)

        # "SYSTEM STATUS" header
        hdr = QLabel("SYSTEM STATUS")
        hdr.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_SMALL, 3))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_lay.addWidget(hdr)

        # Two-column layout
        cols_lay = QHBoxLayout()
        cols_lay.setContentsMargins(0, 6, 0, 0)
        cols_lay.setSpacing(0)

        # Left column — hardware metrics
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        self._hw_cpu = HardwareRow("CPU")
        self._hw_temp = HardwareRow("TEMP")
        self._hw_ram = HardwareRow("RAM")
        left_col.addWidget(self._hw_cpu)
        left_col.addWidget(self._hw_temp)
        left_col.addWidget(self._hw_ram)

        # Divider between metrics and NET
        div_line = QFrame()
        div_line.setFrameShape(QFrame.Shape.HLine)
        div_line.setFixedHeight(1)
        div_line.setStyleSheet(f"background-color: {T.RETRO_BORDER}; border: none; margin: 2px 0;")
        left_col.addWidget(div_line)

        # NET row
        net_row = QWidget()
        net_row.setStyleSheet("background: transparent;")
        net_lay = QHBoxLayout(net_row)
        net_lay.setContentsMargins(0, 1, 0, 1)
        net_lay.setSpacing(4)
        net_lbl = QLabel("NET")
        net_lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_SMALL, 2))
        net_lbl.setFixedWidth(64)
        self._net_dot = QLabel("■")
        self._net_dot.setStyleSheet(_ss("#ff4444", _FZ_SMALL - 3, 0))
        self._net_status_lbl = QLabel("OFFLN")
        self._net_status_lbl.setStyleSheet(_ss("#ff4444", _FZ_SMALL, 1))
        net_glow = QGraphicsDropShadowEffect()
        net_glow.setBlurRadius(5)
        net_glow.setColor(QColor(0, 255, 102, 144))
        net_glow.setOffset(0, 0)
        self._net_dot.setGraphicsEffect(net_glow)
        net_lay.addWidget(net_lbl)
        net_lay.addWidget(self._net_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        net_lay.addWidget(self._net_status_lbl)
        net_lay.addStretch()
        left_col.addWidget(net_row)

        # Vertical divider
        v_div = QFrame()
        v_div.setFrameShape(QFrame.Shape.VLine)
        v_div.setFixedWidth(1)
        v_div.setStyleSheet(f"background-color: {T.RETRO_BORDER}; border: none;")

        # Right column — backend status
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setContentsMargins(20, 0, 0, 0)
        self._be_rows: dict[str, BackendRow] = {}
        for key, lbl in (("local", "RPI"), ("remote", "GPU"), ("claude", "CLAUDE"), ("gemini", "GEMINI")):
            row = BackendRow(lbl)
            self._be_rows[key] = row
            right_col.addWidget(row)
        cols_lay.addLayout(left_col, 1)
        cols_lay.addWidget(v_div)
        cols_lay.addLayout(right_col, 1)
        sb_lay.addLayout(cols_lay)

        # Prompt row (attached below box, shares left/right/bottom border)
        prompt_row = QWidget()
        prompt_row.setStyleSheet(
            f"background-color: #020210; "
            f"border: 2px solid {T.RETRO_BORDER};"
        )
        pr_lay = QHBoxLayout(prompt_row)
        pr_lay.setContentsMargins(20, 6, 20, 6)
        pr_lay.setSpacing(0)

        prompt_gt = QLabel(">")
        prompt_gt.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_BODY, 0))
        prompt_gt.setFixedWidth(22)

        self._prompt_text = QLabel("AWAITING INPUT")
        self._prompt_text.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_BODY, 3))

        self._prompt_cursor = QLabel("█")
        self._prompt_cursor.setFixedWidth(20)
        self._prompt_cursor.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_BODY, 0))

        pr_lay.addWidget(prompt_gt)
        pr_lay.addSpacing(8)
        pr_lay.addWidget(self._prompt_text)
        pr_lay.addSpacing(6)
        pr_lay.addWidget(self._prompt_cursor)
        pr_lay.addStretch()

        bw_lay.addWidget(status_box)
        bw_lay.addWidget(prompt_row)

        cp_lay.addStretch(1)
        cp_lay.addWidget(self._clock_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
        cp_lay.addSpacing(28)
        cp_lay.addWidget(box_wrapper, 0, Qt.AlignmentFlag.AlignHCenter)
        cp_lay.addStretch(1)
        self._center.addWidget(clock_panel)

        # Page 1: recipe step
        recipe_panel = QWidget()
        recipe_panel.setStyleSheet(f"background-color: {T.RETRO_BG};")
        rp_lay = QVBoxLayout(recipe_panel)
        rp_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rp_lay.setSpacing(8)
        rp_lay.setContentsMargins(24, 16, 24, 16)

        self._step_num_lbl = QLabel()
        self._step_num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_num_lbl.setStyleSheet(_ss(T.RETRO_TEXT_MUTED, _RECIPE_NUM_FONT_PX, 2))

        self._step_lbl = QLabel()
        self._step_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_lbl.setWordWrap(True)
        self._step_lbl.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _RECIPE_FONT_PX, 1))

        rp_lay.addWidget(self._step_num_lbl)
        rp_lay.addWidget(self._step_lbl)
        self._center.addWidget(recipe_panel)
        self._center.setCurrentIndex(0)

        # ── Bottom bar (h:36) ────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(36)
        bottom.setStyleSheet(
            f"background-color: {T.RETRO_ELEVATED}; "
            f"border-top: 2px solid {T.RETRO_BORDER};"
        )
        bot_lay = QHBoxLayout(bottom)
        bot_lay.setContentsMargins(16, 0, 16, 0)
        bot_lay.setSpacing(0)

        def _kv_pair(key: str) -> tuple[QLabel, QLabel]:
            k = QLabel(key)
            k.setStyleSheet(_ss(T.RETRO_TEXT_MUTED, _FZ_SMALL, 2))
            v = QLabel("---")
            v.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_SMALL, 2))
            return k, v

        k_ip, self._ip_val = _kv_pair("IP:")
        k_up, self._uptime_val = _kv_pair("UPTIME:")

        self._radio_lbl = QLabel()
        self._radio_lbl.setStyleSheet(
            f"color: #4fc3f7; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ_SMALL}px; letter-spacing: 1px; background: transparent;"
        )
        self._radio_lbl.setVisible(False)

        self._state_dot = QLabel("█")
        self._state_dot_lbl = QLabel("IDLE")
        self._state_dot.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_SMALL, 0))
        self._state_dot_lbl.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_SMALL, 2))

        def _pipe():
            lbl = QLabel("│")
            lbl.setStyleSheet(
                f"color: {T.RETRO_BORDER}; font-family: '{_FONT}', '{_FB}'; "
                f"font-size: {_FZ_SMALL}px; margin: 0 12px; background: transparent;"
            )
            return lbl

        bot_lay.addWidget(k_ip)
        bot_lay.addSpacing(4)
        bot_lay.addWidget(self._ip_val)
        bot_lay.addWidget(_pipe())
        bot_lay.addWidget(k_up)
        bot_lay.addSpacing(4)
        bot_lay.addWidget(self._uptime_val)
        bot_lay.addStretch()
        bot_lay.addWidget(_pipe())
        bot_lay.addWidget(self._state_dot)
        bot_lay.addSpacing(4)
        bot_lay.addWidget(self._state_dot_lbl)

        outer.addWidget(header)
        outer.addWidget(self._center, stretch=1)
        outer.addWidget(bottom)

        # ── Scanline overlay (added last, sits above everything) ─────────────
        self._scanlines = ScanlineOverlay(self)

        # ── Backend monitor ──────────────────────────────────────────────────
        self._monitor = BackendMonitor(cfg, self)
        self._monitor.status_updated.connect(self._on_backends_updated)
        self._monitor.net_updated.connect(self._on_net_updated)

        # ── Timers ───────────────────────────────────────────────────────────
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._tick_blink)
        self._blink_timer.start()

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1_000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()

        self._uptime_timer = QTimer(self)
        self._uptime_timer.setInterval(60_000)
        self._uptime_timer.timeout.connect(self._update_uptime)
        self._uptime_timer.start()

        # Initial renders
        self._update_clock()
        self._update_bars()
        self._update_state_dot("IDLE")
        self._ip_val.setText(self._ip)
        self._update_uptime()

    # ── Layout ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scanlines.setGeometry(self.rect())
        self._scanlines.raise_()

    # ── Public slots ─────────────────────────────────────────────────────────

    def update_resources(self, cpu_pct: float, ram_used_mb: float, ram_total_mb: float, temp):
        self._cpu = cpu_pct
        self._ram_used = ram_used_mb
        self._ram_total = max(ram_total_mb, 1.0)
        self._temp = temp
        self._update_bars()

    def on_radio_changed(self, station: str):
        self._radio_station = station
        if station:
            self._state_dot.setText("♪")
            self._state_dot.setStyleSheet(_ss("#4fc3f7", _FZ_SMALL, 0))
            self._state_dot_lbl.setText(station)
            self._state_dot_lbl.setStyleSheet(_ss("#4fc3f7", _FZ_SMALL, 2))
        else:
            self._update_state_dot(self._state_name)

    def on_state_changed(self, state_name: str):
        self._state_name = state_name
        if not self._radio_station:
            self._update_state_dot(state_name)

    def on_model_changed(self, backend_key: str, _model_name: str):
        self._active_backend = backend_key
        self._monitor.set_active(backend_key)

    def on_recipe_step(self, step_text: str, step_num: int, total_steps: int):
        if not step_text:
            self._center.setCurrentIndex(0)
            return
        if step_num > 0:
            self._step_num_lbl.setText(f"STEP {step_num} OF {total_steps}:")
            self._step_num_lbl.setVisible(True)
        else:
            self._step_num_lbl.setVisible(False)
        self._step_lbl.setText(step_text)
        self._center.setCurrentIndex(1)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _tick_blink(self):
        self._blink = not self._blink

        # Toggle color instead of text to avoid triggering a layout recalculation
        cursor_color = T.RETRO_ACCENT if self._blink else "transparent"
        self._prompt_cursor.setStyleSheet(_ss(cursor_color, _FZ_BODY, 0))

        # Radio ♪ blink
        if self._radio_station and self._radio_lbl.isVisible():
            self._radio_blink = not self._radio_blink
            note = "♪" if self._radio_blink else " "
            self._radio_lbl.setText(f"  │  {note} {self._radio_station[:20]}")

    def _update_clock(self):
        now = datetime.now()
        self._clock_lbl.setText(now.strftime("%H:%M"))
        self._date_lbl.setText(now.strftime("%a, %b %d, %Y").upper())

    def _update_uptime(self):
        self._uptime_val.setText(_get_uptime())

    def _update_bars(self):
        cpu_c = _metric_color(self._cpu, 65, 85)
        temp_c = _metric_color(self._temp or 0, 60, 72)
        ram_pct = self._ram_used / self._ram_total * 100

        self._hw_cpu.update_metric(self._cpu, f"{self._cpu:.0f}%", cpu_c)
        temp_val = f"{self._temp:.0f}C" if self._temp is not None else "--C"
        temp_pct = max(0.0, min(100.0, ((self._temp or 0) - 40) / 55 * 100))
        self._hw_temp.update_metric(temp_pct, temp_val, temp_c)
        self._hw_ram.update_metric(ram_pct, f"{ram_pct:.0f}%", T.RETRO_TEXT_PRIMARY)

    def _update_state_dot(self, state_name: str):
        _, fg, lbl = T.RETRO_STATE.get(state_name, ("", T.RETRO_TEXT_PRIMARY, state_name))
        self._state_dot.setStyleSheet(_ss(fg, _FZ_SMALL, 0))
        self._state_dot_lbl.setText(lbl.replace("*", "").strip())
        self._state_dot_lbl.setStyleSheet(_ss(fg, _FZ_SMALL, 2))

    def _on_backends_updated(self, statuses: list[BackendStatus]):
        for s in statuses:
            row = self._be_rows.get(s.key)
            if row:
                row.set_status(s.online, s.sub_model, s.active)

    def _on_net_updated(self, online: bool):
        self._net_online = online
        color = "#00ff66" if online else "#ff4444"
        self._net_dot.setStyleSheet(_ss(color, _FZ_SMALL - 3, 0))
        self._net_status_lbl.setText("ONLN" if online else "OFFLN")
        self._net_status_lbl.setStyleSheet(_ss(color, _FZ_SMALL, 1))
