"""Retro-theme top status bar (40px): state tag, model badge, radio, clock."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme as T

_FONT = "DejaVu Sans Mono"
_PULSE_MS = 530


class RetroStateBar(QWidget):
    """52px top bar. Public slots: on_state_changed, on_model_changed, on_radio_changed."""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setObjectName("retro_state_bar")

        self._state = "IDLE"
        self._model_key = "local"
        self._model_name = ""
        self._pulse_on = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        # State tag (bordered box)
        self._state_tag = QLabel()
        self._state_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_tag.setFixedHeight(36)
        self._state_tag.setContentsMargins(12, 0, 12, 0)

        self._model_tag = QLabel()
        self._model_tag.setFixedHeight(34)
        self._model_tag.setContentsMargins(10, 0, 10, 0)
        self._model_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._radio_tag = QLabel()
        self._radio_tag.setFixedHeight(34)
        self._radio_tag.setContentsMargins(10, 0, 10, 0)
        self._radio_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._radio_tag.setVisible(False)

        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_SECONDARY}; font-family: '{_FONT}'; "
            f"font-size: 18px; letter-spacing: 4px;"
        )
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._spacer_lbl = QLabel("   ")  # gap between state and model
        self._spacer_lbl.setStyleSheet("background: transparent;")

        layout.addWidget(self._state_tag)
        layout.addWidget(self._spacer_lbl)
        layout.addWidget(self._model_tag)
        layout.addStretch()
        layout.addWidget(self._radio_tag)
        layout.addSpacing(12)
        layout.addWidget(self._clock_lbl)

        # Pulse timer for LISTENING / THINKING dot
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(_PULSE_MS)
        self._pulse_timer.timeout.connect(self._pulse_tick)

        self._apply_state("IDLE")
        self._apply_model("local", "")
        self._update_clock()

        clock_timer = QTimer(self)
        clock_timer.setInterval(30_000)
        clock_timer.timeout.connect(self._update_clock)
        clock_timer.start()

        self._connect_signals(bus)

        # Background for whole bar
        self._refresh_bar_style()

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
        self._model_name = model_name
        self._apply_model(backend_key, model_name)

    def on_radio_changed(self, station: str):
        if station:
            self._radio_tag.setText(f"♪  {station[:22]}")
            self._radio_tag.setStyleSheet(
                "color: #4fc3f7; font-family: 'DejaVu Sans Mono'; "
                "font-size: 13px; letter-spacing: 1px; border: 1px solid #4fc3f760;"
            )
            self._radio_tag.setVisible(True)
        else:
            self._radio_tag.setVisible(False)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _apply_state(self, state_name: str):
        self._pulse_timer.stop()
        _, fg, lbl = T.RETRO_STATE.get(state_name, ("", T.RETRO_TEXT_PRIMARY, state_name))
        bg, *_ = T.RETRO_STATE.get(state_name, (T.RETRO_BG,))
        self._state_tag.setText(lbl)
        self._state_tag.setStyleSheet(
            f"color: {fg}; background-color: {bg}; font-family: '{_FONT}'; "
            f"font-size: 15px; letter-spacing: 2px; font-weight: bold; "
            f"border: 2px solid {fg}; padding: 2px 12px;"
        )
        if state_name in ("LISTENING", "THINKING", "SPEAKING"):
            self._pulse_timer.start()

    def _apply_model(self, backend_key: str, model_name: str):
        fg = T.RETRO_MODEL_FG.get(backend_key, T.RETRO_TEXT_SECONDARY)
        label = T.MODEL_LABELS.get(backend_key, backend_key.upper())
        sub = T.RETRO_MODEL_SUB.get(backend_key, "")
        tag = model_name.split("/")[-1][:16] if model_name else sub
        self._model_tag.setText(f"{label} / {tag.upper()}")
        self._model_tag.setStyleSheet(
            f"color: {fg}; font-family: '{_FONT}'; font-size: 13px; "
            f"letter-spacing: 2px; border: 1px solid {fg}60; padding: 2px 10px;"
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
        _, fg, lbl = T.RETRO_STATE.get(self._state, ("", T.RETRO_TEXT_PRIMARY, self._state))
        bg, *_ = T.RETRO_STATE.get(self._state, (T.RETRO_BG,))
        color = fg if self._pulse_on else T.RETRO_TEXT_MUTED
        self._state_tag.setStyleSheet(
            f"color: {color}; background-color: {bg}; font-family: '{_FONT}'; "
            f"font-size: 15px; letter-spacing: 2px; font-weight: bold; "
            f"border: 2px solid {color}; padding: 2px 12px;"
        )

    def _update_clock(self):
        self._clock_lbl.setText(datetime.now().strftime("%H:%M"))
