"""Main chat area: user speech label + inline tool status + scrolling LLM response view."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme

_SPINNER_FRAMES = (".", "..", "...")
_SPINNER_INTERVAL_MS = 450
_TOOL_HIDE_MS = 3000


class UserSpeechLabel(QLabel):
    """Displays the last transcribed user input.

    Fixed height, italic, cleared at the start of each new turn.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.setContentsMargins(12, 4, 12, 4)
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText("")
        self.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-style: italic; font-size: 24px;")

    def set_text(self, text: str):
        # Render "You: <text>" with accent prefix
        self.setText(f"You: {text}")
        self.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-style: italic; font-size: 24px;"
        )

    def clear_speech(self):
        self._set_placeholder()


class ResponseView(QTextEdit):
    """Scrolling LLM response area — tokens appended one by one.

    Cleared at the start of each new LISTENING cycle so only the
    current turn's response is displayed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("DejaVu Sans", 26))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(0)

    def append_token(self, token: str):
        """Append a single LLM token and scroll to bottom."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_response(self):
        self.clear()


class ToolLabel(QLabel):
    """Inline tool status shown between user question and LLM response.

    Animates while EXECUTING; turns green on DONE and auto-hides after 3s.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setContentsMargins(12, 4, 12, 4)
        self.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-family: 'DejaVu Sans Mono'; font-size: 17px;"
            f"background-color: {theme.BG_ELEVATED};"
        )

        self._spinner_frame = 0
        self._tool_name = ""

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_INTERVAL_MS)
        self._spinner_timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_TOOL_HIDE_MS)
        self._hide_timer.timeout.connect(self.hide)

        self.hide()

    def on_tool_started(self, tool_name: str):
        self._hide_timer.stop()
        self._tool_name = tool_name
        self._spinner_frame = 0
        self._spinner_timer.start()
        self.setStyleSheet(
            f"color: #90CAF9; font-family: 'DejaVu Sans Mono'; font-size: 17px;"
            f"background-color: {theme.BG_ELEVATED};"
        )
        self.setText(f"⚙  {tool_name}  ·  EXECUTING.")
        self.show()

    def on_tool_done(self, tool_name: str, result: str):
        self._spinner_timer.stop()
        short = result[:50] + "…" if len(result) > 50 else result
        label = f"⚙  {tool_name}  ·  DONE" + (f"  {short}" if short else "")
        self.setText(label)
        self.setStyleSheet(
            f"color: #69F0AE; font-family: 'DejaVu Sans Mono'; font-size: 17px;"
            f"background-color: {theme.BG_ELEVATED};"
        )
        self._hide_timer.start()

    def _tick(self):
        self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER_FRAMES)
        dots = _SPINNER_FRAMES[self._spinner_frame]
        self.setText(f"⚙  {self._tool_name}  ·  EXECUTING{dots}")


class ChatView(QWidget):
    """Main content area:
        - UserSpeechLabel (fixed 60px)
        - ToolLabel (inline tool status, hidden when idle)
        - Thin horizontal separator
        - ResponseView (stretches to fill)
    """

    def __init__(self, bus, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._user_label = UserSpeechLabel(self)
        self._tool_label = ToolLabel(self)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2A2A45; border: none;")

        self._response = ResponseView(self)

        layout.addWidget(self._user_label)
        layout.addWidget(self._tool_label)
        layout.addWidget(sep)
        layout.addWidget(self._response, stretch=1)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.user_said.connect(self._on_user_said)
        bus.text_chunk.connect(self._response.append_token)
        bus.state_changed.connect(self._on_state_changed)
        bus.tool_started.connect(self._tool_label.on_tool_started)
        bus.tool_done.connect(self._tool_label.on_tool_done)

    def _on_user_said(self, text: str):
        self._user_label.set_text(text)

    def _on_state_changed(self, state_name: str):
        # Clear response area when starting a new listening phase
        if state_name == "LISTENING":
            self._response.clear_response()
            self._user_label.clear_speech()
            self._tool_label.hide()
