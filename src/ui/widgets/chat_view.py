"""Main chat area: user speech label + unified conversation log (tool events + LLM response)."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QFrame, QLabel, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme

_TOOL_RUNNING_COLOR = "#90CAF9"   # blue — executing
_TOOL_DONE_COLOR    = "#69F0AE"   # green — complete
_RESPONSE_COLOR     = theme.TEXT_PRIMARY
_TOOL_FONT_SIZE     = 16
_RESPONSE_FONT_SIZE = 28


class UserSpeechLabel(QLabel):
    """Displays the last transcribed user input. Fixed height, italic."""

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
        self.setText(f"You: {text}")
        self.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-style: italic; font-size: 24px;"
        )

    def clear_speech(self):
        self._set_placeholder()


class ConversationLog(QTextEdit):
    """Unified scrolling log: persistent tool blocks + streaming LLM response.

    Tool blocks appear above the response text and are updated in-place
    (running → done) without disappearing. Each tool call gets its own line.
    Cleared at the start of each new LISTENING cycle.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(12)
        # Per-tool saved cursors (selecting the tool line text for in-place update)
        self._tool_cursors: dict[str, QTextCursor] = {}
        self._has_tool_content = False
        self._response_started = False

    # ── Tool events ───────────────────────────────────────────────────────────

    def append_tool_start(self, name: str):
        """Insert a blue 'running' line for this tool and save a cursor to it."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_TOOL_RUNNING_COLOR))
        fmt.setFont(QFont("DejaVu Sans Mono", _TOOL_FONT_SIZE))

        cursor.insertText(f"⚙ {name}  ·  running…\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

        # Save a cursor that selects the line we just inserted (excluding trailing \n)
        saved = QTextCursor(self.document())
        saved.movePosition(QTextCursor.MoveOperation.End)
        saved.movePosition(QTextCursor.MoveOperation.PreviousBlock)
        saved.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        saved.movePosition(
            QTextCursor.MoveOperation.EndOfBlock,
            QTextCursor.MoveMode.KeepAnchor,
        )
        self._tool_cursors[name] = saved
        self._has_tool_content = True

    def update_tool_done(self, name: str, result: str):
        """Replace the running line for this tool with the green done line."""
        saved = self._tool_cursors.pop(name, None)
        if saved is None:
            return

        short = result[:70] + "…" if len(result) > 70 else result
        new_text = f"⚙ {name}  ·  {short}" if short else f"⚙ {name}  ·  done"

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_TOOL_DONE_COLOR))
        fmt.setFont(QFont("DejaVu Sans Mono", _TOOL_FONT_SIZE))

        saved.removeSelectedText()
        saved.insertText(new_text, fmt)
        self.ensureCursorVisible()

    # ── LLM response streaming ────────────────────────────────────────────────

    def append_token(self, token: str):
        """Append one LLM token. Inserts a blank separator line after tool blocks."""
        if self._has_tool_content and not self._response_started:
            self._response_started = True
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            sep_fmt = QTextCharFormat()
            sep_fmt.setForeground(QColor(theme.BG_SURFACE))
            cursor.insertText("\n", sep_fmt)
            self.setTextCursor(cursor)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_RESPONSE_COLOR))
        fmt.setFont(QFont("DejaVu Sans", _RESPONSE_FONT_SIZE))

        cursor.insertText(token, fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear_log(self):
        self.clear()
        self._tool_cursors.clear()
        self._has_tool_content = False
        self._response_started = False


class ChatView(QWidget):
    """Main content area:
        - UserSpeechLabel (fixed 68px)
        - Thin horizontal separator
        - ConversationLog (stretches to fill) — shows tool events + LLM response
    """

    def __init__(self, bus, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._user_label = UserSpeechLabel(self)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #2A2A45; border: none;")

        self._log = ConversationLog(self)

        layout.addWidget(self._user_label)
        layout.addWidget(sep)
        layout.addWidget(self._log, stretch=1)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.user_said.connect(self._on_user_said)
        bus.text_chunk.connect(self._log.append_token)
        bus.state_changed.connect(self._on_state_changed)
        bus.tool_started.connect(self._log.append_tool_start)
        bus.tool_done.connect(self._log.update_tool_done)

    def _on_user_said(self, text: str):
        self._user_label.set_text(text)

    def _on_state_changed(self, state_name: str):
        if state_name == "LISTENING":
            self._log.clear_log()
            self._user_label.clear_speech()
