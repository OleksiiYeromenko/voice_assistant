"""Retro-theme chat view: terminal-style user input line + scrolling log."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme as T

_FONT = "DejaVu Sans Mono"
_TOOL_RUNNING_COLOR = "#ffcc00"
_TOOL_DONE_COLOR = "#00ff66"
_RESPONSE_COLOR = T.RETRO_TEXT_PRIMARY
_TOOL_FONT_PX = 18
_RESPONSE_FONT_PX = 26


class RetroConversationLog(QTextEdit):
    """Retro-styled scrolling log: tool events + LLM response.

    Tool calls are boxed with [EXEC]/[DONE] prefixes.
    Response text uses primary green with 'SYS OUTPUT:' header.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(12)
        self.setStyleSheet(
            f"QTextEdit {{ background-color: {T.RETRO_SURFACE}; color: {T.RETRO_TEXT_PRIMARY}; "
            f"border: none; font-family: '{_FONT}'; font-size: {_TOOL_FONT_PX}px; }}"
        )
        self._tool_cursors: dict[str, QTextCursor] = {}
        self._has_tool_content = False
        self._response_started = False
        self._blink_on = True
        self._blink_cursor_pos: int | None = None

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(530)
        self._blink_timer.timeout.connect(self._blink_tick)

    # ── Tool events ───────────────────────────────────────────────────────────

    def append_tool_start(self, name: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_TOOL_RUNNING_COLOR))
        _f = QFont(_FONT)
        _f.setPixelSize(_TOOL_FONT_PX)
        _f.setWeight(QFont.Weight.Bold)
        fmt.setFont(_f)

        cursor.insertText(f"[EXEC] {name.upper()}\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

        saved = QTextCursor(self.document())
        saved.movePosition(QTextCursor.MoveOperation.End)
        saved.movePosition(QTextCursor.MoveOperation.PreviousBlock)
        saved.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        saved.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        self._tool_cursors[name] = saved
        self._has_tool_content = True

    def update_tool_done(self, name: str, result: str):
        saved = self._tool_cursors.pop(name, None)
        if saved is None:
            return

        short = result[:72] + "…" if len(result) > 72 else result
        new_text = f"[DONE] {name.upper()}  {short}" if short else f"[DONE] {name.upper()}"

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_TOOL_DONE_COLOR))
        f = QFont(_FONT)
        f.setPixelSize(_TOOL_FONT_PX)
        fmt.setFont(f)

        saved.removeSelectedText()
        saved.insertText(new_text, fmt)
        self.ensureCursorVisible()

    # ── LLM streaming ─────────────────────────────────────────────────────────

    def append_token(self, token: str):
        if self._has_tool_content and not self._response_started:
            self._response_started = True
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            # Dashed divider
            div_fmt = QTextCharFormat()
            div_fmt.setForeground(QColor(T.RETRO_BORDER))
            f = QFont(_FONT)
            f.setPixelSize(_TOOL_FONT_PX)
            div_fmt.setFont(f)
            cursor.insertText("\n" + "─" * 40 + "\n", div_fmt)
            # "SYS OUTPUT:" header
            hdr_fmt = QTextCharFormat()
            hdr_fmt.setForeground(QColor(T.RETRO_ACCENT))
            cursor.insertText("SYS OUTPUT:\n", hdr_fmt)
            self.setTextCursor(cursor)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_RESPONSE_COLOR))
        f = QFont(_FONT)
        f.setPixelSize(_RESPONSE_FONT_PX)
        fmt.setFont(f)
        cursor.insertText(token, fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def start_stream(self):
        self._blink_timer.start()

    def stop_stream(self):
        self._blink_timer.stop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear_log(self):
        self.clear()
        self._tool_cursors.clear()
        self._has_tool_content = False
        self._response_started = False
        self._blink_timer.stop()

    def _blink_tick(self):
        pass  # blink cursor handled by append_token's trailing █ if needed


class RetroChatView(QWidget):
    """Retro chat area: 40px terminal input line + flex ConversationLog."""

    def __init__(self, bus, parent=None):
        super().__init__(parent)

        self._fsm_state = "IDLE"
        self._blink = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(530)
        self._blink_timer.timeout.connect(self._tick_blink)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── User input line (52px) ───────────────────────────────────────────
        input_row = QWidget()
        input_row.setFixedHeight(52)
        input_row.setStyleSheet(
            f"background-color: #040414; border-bottom: 1px solid {T.RETRO_BORDER};"
        )
        ir_layout = QHBoxLayout(input_row)
        ir_layout.setContentsMargins(18, 0, 18, 0)
        ir_layout.setSpacing(0)

        self._prompt_sym = QLabel(">")
        self._prompt_sym.setFixedWidth(26)
        self._prompt_sym.setStyleSheet(
            f"color: {T.RETRO_ACCENT}; font-family: '{_FONT}'; "
            f"font-size: 16px; letter-spacing: 1px;"
        )

        self._input_lbl = QLabel()
        self._input_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
            f"font-size: 16px; letter-spacing: 1px;"
        )

        self._cursor_lbl = QLabel("█")
        self._cursor_lbl.setStyleSheet(
            f"color: #00ff66; font-family: '{_FONT}'; font-size: 16px;"
        )
        self._cursor_lbl.setVisible(False)

        ir_layout.addWidget(self._prompt_sym)
        ir_layout.addSpacing(6)
        ir_layout.addWidget(self._input_lbl, stretch=1)
        ir_layout.addWidget(self._cursor_lbl)

        # ── Log area ─────────────────────────────────────────────────────────
        self._log = RetroConversationLog(self)

        layout.addWidget(input_row)
        layout.addWidget(self._log, stretch=1)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.user_said.connect(self._on_user_said)
        bus.text_chunk.connect(self._log.append_token)
        bus.state_changed.connect(self._on_state_changed)
        bus.tool_started.connect(self._log.append_tool_start)
        bus.tool_done.connect(self._log.update_tool_done)
        bus.response_complete.connect(self._log.stop_stream)

    def _on_user_said(self, text: str):
        self._input_lbl.setText(text)
        self._input_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_PRIMARY}; font-family: '{_FONT}'; "
            f"font-size: 16px; letter-spacing: 1px;"
        )
        self._cursor_lbl.setVisible(False)
        self._blink_timer.stop()

    def _on_state_changed(self, state_name: str):
        self._fsm_state = state_name
        if state_name == "LISTENING":
            self._log.clear_log()
            self._input_lbl.setText("RECORDING...")
            self._input_lbl.setStyleSheet(
                f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
                f"font-size: 16px; letter-spacing: 2px;"
            )
            self._cursor_lbl.setVisible(True)
            self._blink_timer.start()
        elif state_name == "THINKING":
            self._cursor_lbl.setVisible(False)
            self._blink_timer.stop()
            # Show PROCESSING in log
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#ffcc00"))
            f = QFont(_FONT)
            f.setPixelSize(_TOOL_FONT_PX)
            fmt.setFont(f)
            cursor = self._log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("PROCESSING...\n", fmt)
            self._log.setTextCursor(cursor)

    def _tick_blink(self):
        self._blink = not self._blink
        self._cursor_lbl.setVisible(self._blink)
