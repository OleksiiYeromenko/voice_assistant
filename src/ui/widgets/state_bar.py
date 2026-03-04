"""Top state bar: animated state pill, model badge, turn counter, clock."""

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme

_THINKING_DOT_FRAMES = ("", ".", "..", "...")
_THINKING_INTERVAL_MS = 400
# Listening pulse: alternate between bright and dim fg color via stylesheet
_LISTENING_PULSE_FRAMES = (True, False)   # True = bright, False = dim
_LISTENING_PULSE_MS = 500
_CLOCK_INTERVAL_MS = 30_000


class StatePill(QLabel):
    """Rounded pill label that changes color per FSM state.

    LISTENING: text color pulses bright↔dim via QTimer (no compositing needed).
    THINKING:  dots cycle via QTimer to simulate a typing indicator.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(36)
        self.setMinimumWidth(130)

        # Shared timer for both listening pulse and thinking dots
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_frame = 0
        self._anim_mode = ""   # "listen" | "think" | ""

        self._bg = ""
        self._fg_bright = ""
        self._fg_dim = ""
        self._base_text = ""

        self.set_state("IDLE")

    def _apply_pill_style(self, bg: str, fg: str):
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border-radius: 14px;"
            f"  padding: 4px 16px;"
            f"  font-size: 18px;"
            f"  font-weight: bold;"
            f"  letter-spacing: 1px;"
            f"}}"
        )

    def set_state(self, state_name: str):
        self._anim_timer.stop()
        self._anim_frame = 0
        self._anim_mode = ""

        bg, fg = theme.STATE_COLORS.get(state_name, ("#2D3748", "#718096"))
        self._bg = bg
        self._fg_bright = fg
        # Dim version: same hue at ~40% opacity via a slightly muted color
        self._fg_dim = theme.TEXT_MUTED
        self._base_text = state_name

        self._apply_pill_style(bg, fg)
        self.setText(state_name)

        if state_name == "LISTENING":
            self._anim_mode = "listen"
            self._anim_timer.setInterval(_LISTENING_PULSE_MS)
            self._anim_timer.start()
        elif state_name == "THINKING":
            self._anim_mode = "think"
            self._anim_timer.setInterval(_THINKING_INTERVAL_MS)
            self._anim_timer.start()

    def _tick(self):
        self._anim_frame += 1
        if self._anim_mode == "listen":
            bright = (self._anim_frame % 2 == 0)
            fg = self._fg_bright if bright else self._fg_dim
            self._apply_pill_style(self._bg, fg)
        elif self._anim_mode == "think":
            dots = _THINKING_DOT_FRAMES[self._anim_frame % len(_THINKING_DOT_FRAMES)]
            self.setText(self._base_text + dots)


class ModelBadge(QLabel):
    """Colored badge showing the active backend and model name."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(32)
        self.setMinimumWidth(90)
        self.set_model("local", "")

    def set_model(self, backend_key: str, model_name: str):
        bg, fg = theme.MODEL_COLORS.get(backend_key, ("#2D3748", "#718096"))
        label = theme.MODEL_LABELS.get(backend_key, backend_key.upper())
        icon = theme.MODEL_ICONS.get(backend_key, "")
        model_tag = model_name.split("/")[-1].split(":")[0][:14] if model_name else ""
        prefix = f"{icon} {label}" if icon else label
        display = f"{prefix}  {model_tag}" if model_tag else prefix
        self.setText(display)
        self.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {bg};"
            f"  color: {fg};"
            f"  border-radius: 12px;"
            f"  padding: 4px 12px;"
            f"  font-size: 16px;"
            f"  font-family: 'DejaVu Sans Mono';"
            f"}}"
        )


class StateBar(QWidget):
    """Top bar (58px): [StatePill] [ModelBadge] [stretch] [Turn #N] [HH:MM]"""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setObjectName("state_bar")
        self.setStyleSheet(
            f"#state_bar {{ background-color: {theme.BG_ELEVATED}; "
            f"border-bottom: 1px solid #2A2A45; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._pill = StatePill(self)
        self._badge = ModelBadge(self)

        self._turn_label = QLabel("", self)
        self._turn_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 16px;")
        self._turn_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._clock = QLabel(self)
        self._clock.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-family: 'DejaVu Sans Mono'; font-size: 18px;"
        )
        self._clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_clock()

        layout.addWidget(self._pill)
        layout.addWidget(self._badge)
        layout.addStretch()
        layout.addWidget(self._turn_label)
        layout.addWidget(self._clock)

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
