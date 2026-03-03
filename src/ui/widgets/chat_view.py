"""Main chat area: user speech label + scrolling LLM response view."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme


class UserSpeechLabel(QLabel):
    """Displays the last transcribed user input.

    Fixed height, italic, cleared at the start of each new turn.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.setContentsMargins(12, 4, 12, 4)
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText("")
        self.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-style: italic; font-size: 15px;")

    def set_text(self, text: str):
        # Render "You: <text>" with accent prefix
        self.setText(f"You: {text}")
        self.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-style: italic; font-size: 15px;"
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
        self.setFont(QFont("DejaVu Sans", 18))
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


class ChatView(QWidget):
    """Main content area:
        - UserSpeechLabel (fixed 44px)
        - Thin horizontal separator
        - ResponseView (stretches to fill)
    """

    def __init__(self, bus, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._user_label = UserSpeechLabel(self)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2A2A45; border: none;")

        self._response = ResponseView(self)

        layout.addWidget(self._user_label)
        layout.addWidget(sep)
        layout.addWidget(self._response, stretch=1)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.user_said.connect(self._on_user_said)
        bus.text_chunk.connect(self._response.append_token)
        bus.state_changed.connect(self._on_state_changed)

    def _on_user_said(self, text: str):
        self._user_label.set_text(text)

    def _on_state_changed(self, state_name: str):
        # Clear response area when starting a new listening phase
        if state_name == "LISTENING":
            self._response.clear_response()
            self._user_label.clear_speech()
