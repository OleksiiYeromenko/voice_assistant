"""Top state bar: animated state pill, model badge, turn counter, clock."""

from datetime import datetime

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PyQt6.QtGui import QGraphicsOpacityEffect
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme

_THINKING_DOT_FRAMES = ("", ".", "..", "...")
_THINKING_INTERVAL_MS = 400
_LISTENING_PULSE_DURATION_MS = 800
_CLOCK_INTERVAL_MS = 30_000


class StatePill(QLabel):
    """Rounded pill label that changes color per FSM state.

    LISTENING: a separate pulsing dot animates via QPropertyAnimation.
    THINKING:  dots cycle via QTimer to simulate a typing indicator.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(28)
        self.setMinimumWidth(120)

        # Pulsing dot for LISTENING state
        self._dot = QLabel("●", parent)
        self._dot.setFixedSize(16, 16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot_effect = QGraphicsOpacityEffect(self._dot)
        self._dot.setGraphicsEffect(self._dot_effect)
        self._dot_anim = QPropertyAnimation(self._dot_effect, b"opacity", self)
        self._dot_anim.setDuration(_LISTENING_PULSE_DURATION_MS)
        self._dot_anim.setStartValue(1.0)
        self._dot_anim.setEndValue(0.15)
        self._dot_anim.setEasingCurve(QEasingCurve.Type.SineCurve)
        self._dot_anim.setLoopCount(-1)  # infinite
        self._dot.hide()

        # Thinking dot timer
        self._think_timer = QTimer(self)
        self._think_timer.setInterval(_THINKING_INTERVAL_MS)
        self._think_timer.timeout.connect(self._tick_thinking)
        self._think_frame = 0
        self._base_text = ""

        self.set_state("IDLE")

    def set_state(self, state_name: str):
        self._think_timer.stop()
        self._dot_anim.stop()
        self._dot.hide()

        bg, fg = theme.STATE_COLORS.get(state_name, ("#2D3748", "#718096"))
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border-radius: 12px;"
            f"  padding: 2px 14px;"
            f"  font-size: 13px;"
            f"  font-weight: bold;"
            f"  letter-spacing: 1px;"
            f"}}"
        )
        self._dot.setStyleSheet(f"color: {fg}; font-size: 10px;")
        self._base_text = state_name

        if state_name == "LISTENING":
            self.setText(state_name)
            self._dot.show()
            self._dot_anim.setDirection(QPropertyAnimation.Direction.Forward)
            self._dot_anim.start()
        elif state_name == "THINKING":
            self._think_frame = 0
            self.setText(state_name)
            self._think_timer.start()
        else:
            self.setText(state_name)

    def _tick_thinking(self):
        self._think_frame = (self._think_frame + 1) % len(_THINKING_DOT_FRAMES)
        self.setText(self._base_text + _THINKING_DOT_FRAMES[self._think_frame])

    def dot_widget(self) -> QLabel:
        """Return the pulsing dot so the parent layout can position it."""
        return self._dot


class ModelBadge(QLabel):
    """Colored badge showing the active backend and model name."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(24)
        self.setMinimumWidth(80)
        self.set_model("local", "")

    def set_model(self, backend_key: str, model_name: str):
        bg, fg = theme.MODEL_COLORS.get(backend_key, ("#2D3748", "#718096"))
        label = theme.MODEL_LABELS.get(backend_key, backend_key.upper())
        # Show short model tag (last segment after /)
        model_tag = model_name.split("/")[-1].split(":")[0][:14] if model_name else ""
        display = f"{label}  {model_tag}" if model_tag else label
        self.setText(display)
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border-radius: 10px;"
            f"  padding: 2px 10px;"
            f"  font-size: 12px;"
            f"  font-family: 'DejaVu Sans Mono';"
            f"}}"
        )


class StateBar(QWidget):
    """Top bar (44px): [StatePill] [dot] [ModelBadge] [stretch] [Turn #N] [HH:MM]"""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setObjectName("state_bar")
        self.setStyleSheet(
            f"#state_bar {{ background-color: {theme.BG_ELEVATED}; "
            f"border-bottom: 1px solid #2A2A45; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._pill = StatePill(self)
        self._dot = self._pill.dot_widget()
        self._badge = ModelBadge(self)

        self._turn_label = QLabel("", self)
        self._turn_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 13px;")
        self._turn_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._clock = QLabel(self)
        self._clock.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-family: 'DejaVu Sans Mono'; font-size: 13px;"
        )
        self._clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_clock()

        layout.addWidget(self._pill)
        layout.addWidget(self._dot)
        layout.addWidget(self._badge)
        layout.addStretch()
        layout.addWidget(self._turn_label)
        layout.addWidget(self._clock)

        # Clock refresh timer
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(_CLOCK_INTERVAL_MS)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.state_changed.connect(self._on_state_changed)
        bus.model_changed.connect(self._on_model_changed)
        bus.turn_count_updated.connect(self._on_turn_count)

    def _on_state_changed(self, state_name: str):
        self._pill.set_state(state_name)

    def _on_model_changed(self, backend_key: str, model_name: str):
        self._badge.set_model(backend_key, model_name)

    def _on_turn_count(self, count: int):
        self._turn_label.setText(f"Turn #{count}")

    def _update_clock(self):
        self._clock.setText(datetime.now().strftime("%H:%M"))
