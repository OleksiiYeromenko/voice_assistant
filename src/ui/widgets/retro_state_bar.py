"""Retro-theme top status bar (h:40): state tag, model badge, time.

Spec: docs/design/design_handoff_retro_ui/README.md — Screen 2 / Top Status Bar
"""
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QWidget

from src.ui import theme as T

_FONT = "Press Start 2P"
_FB = "DejaVu Sans Mono"
_PULSE_MS = 900  # softpulse period


def _ss(color: str, size: int, spacing: int = 1) -> str:
    return (
        f"color: {color}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {size}px; letter-spacing: {spacing}px; background: transparent;"
    )


class RetroStateBar(QWidget):
    """40px top bar with state tag, model badge, and clock.

    Public slots: on_state_changed, on_model_changed, on_radio_changed.
    """

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setObjectName("retro_state_bar")

        self._state = "IDLE"
        self._model_key = "local"
        self._pulse_on = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)

        # State tag: bordered box, color changes per state
        self._state_tag = QLabel()
        self._state_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_tag.setFixedHeight(34)
        self._state_tag.setContentsMargins(10, 0, 10, 0)

        # Gap
        self._spacer = QLabel("  ")
        self._spacer.setStyleSheet("background: transparent;")

        # Model badge
        self._model_tag = QLabel()
        self._model_tag.setFixedHeight(30)
        self._model_tag.setContentsMargins(8, 0, 8, 0)
        self._model_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Radio tag (optional)
        self._radio_tag = QLabel()
        self._radio_tag.setFixedHeight(30)
        self._radio_tag.setContentsMargins(8, 0, 8, 0)
        self._radio_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._radio_tag.setVisible(False)

        # Time (right-aligned)
        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, 16, 4))
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._state_tag)
        layout.addWidget(self._spacer)
        layout.addWidget(self._model_tag)
        layout.addStretch()
        layout.addWidget(self._radio_tag)
        layout.addSpacing(12)
        layout.addWidget(self._clock_lbl)

        # Softpulse timer for LISTENING / THINKING
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(_PULSE_MS)
        self._pulse_timer.timeout.connect(self._pulse_tick)

        # Clock timer
        clock_timer = QTimer(self)
        clock_timer.setInterval(30_000)
        clock_timer.timeout.connect(self._update_clock)
        clock_timer.start()

        self._apply_state("IDLE")
        self._apply_model("local", "")
        self._update_clock()
        self._refresh_bar_style()

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.state_changed.connect(self.on_state_changed)
        bus.model_changed.connect(self.on_model_changed)
        bus.radio_changed.connect(self.on_radio_changed)

    # ── Public slots ─────────────────────────────────────────────────────────

    def on_state_changed(self, state_name: str):
        self._state = state_name
        self._apply_state(state_name)
        self._refresh_bar_style()

    def on_model_changed(self, backend_key: str, model_name: str):
        self._model_key = backend_key
        self._apply_model(backend_key, model_name)

    def on_radio_changed(self, station: str):
        if station:
            self._radio_tag.setText(f"♪  {station[:18]}")
            self._radio_tag.setStyleSheet(
                f"color: #4fc3f7; font-family: '{_FONT}', '{_FB}'; "
                f"font-size: 12px; letter-spacing: 1px; border: 1px solid #4fc3f760; "
                f"background: transparent;"
            )
            self._radio_tag.setVisible(True)
        else:
            self._radio_tag.setVisible(False)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _apply_state(self, state_name: str):
        self._pulse_timer.stop()
        self._pulse_on = True
        bg, fg, lbl = T.RETRO_STATE.get(state_name, (T.RETRO_BG, T.RETRO_TEXT_PRIMARY, state_name))
        self._state_fg = fg
        self._state_bg = bg
        self._state_lbl = lbl
        self._draw_state_tag(fg, bg, lbl)
        if state_name in ("LISTENING", "THINKING", "SPEAKING"):
            self._pulse_timer.start()

    def _draw_state_tag(self, fg: str, bg: str, lbl: str):
        self._state_tag.setText(lbl)
        self._state_tag.setStyleSheet(
            f"color: {fg}; background-color: {bg}; "
            f"font-family: '{_FONT}', '{_FB}'; "
            f"font-size: 13px; letter-spacing: 2px; font-weight: bold; "
            f"border: 2px solid {fg}; padding: 2px 10px;"
        )
        # Glow on state tag
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(10)
        eff.setColor(QColor(fg))
        eff.setOffset(0, 0)
        self._state_tag.setGraphicsEffect(eff)

    def _apply_model(self, backend_key: str, model_name: str):
        fg = T.RETRO_MODEL_FG.get(backend_key, T.RETRO_TEXT_SECONDARY)
        label = T.MODEL_LABELS.get(backend_key, backend_key.upper())
        sub = T.RETRO_MODEL_SUB.get(backend_key, "")
        tag = model_name.split("/")[-1][:14] if model_name else sub
        self._model_tag.setText(f"{label} / {tag.upper()}")
        self._model_tag.setStyleSheet(
            f"color: {fg}; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: 12px; letter-spacing: 2px; "
            f"border: 1px solid {fg}60; padding: 2px 8px; background: transparent;"
        )

    def _refresh_bar_style(self):
        _, fg, _ = T.RETRO_STATE.get(self._state, ("", T.RETRO_TEXT_PRIMARY, ""))
        self.setStyleSheet(
            f"#retro_state_bar {{ "
            f"background-color: {T.RETRO_ELEVATED}; "
            f"border-bottom: 2px solid {fg}; "
            f"}}"
        )

    def _pulse_tick(self):
        self._pulse_on = not self._pulse_on
        fg = self._state_fg if self._pulse_on else T.RETRO_TEXT_MUTED
        bg = self._state_bg
        self._draw_state_tag(fg, bg, self._state_lbl)

    def _update_clock(self):
        self._clock_lbl.setText(datetime.now().strftime("%H:%M"))
