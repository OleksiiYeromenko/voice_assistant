"""Retro-theme chat view: terminal input line + scrolling log area.

Spec: docs/design/design_handoff_retro_ui/README.md — Screen 2 / Active
"""
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget

from src.ui import theme as T

_FONT = "Press Start 2P"
_FB = "DejaVu Sans Mono"

# Font pixel sizes matching spec
_FZ_TOOL = 11    # spec 7px — tool status rows
_FZ_RESP = 15    # spec 11px — response text
_FZ_INPUT = 11   # spec 8px — user input line


def _ss(color: str, size: int, spacing: int = 1) -> str:
    return (
        f"color: {color}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {size}px; letter-spacing: {spacing}px; background: transparent;"
    )


class RetroConversationLog(QTextEdit):
    """Retro scrolling log: tool events + LLM response text.

    Tool calls are boxed with [EXEC]/[DONE] prefixes.
    Response text uses 'SYS OUTPUT:' header with cyan text.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(14)
        self.setStyleSheet(
            f"QTextEdit {{ background-color: {T.RETRO_SURFACE}; color: {T.RETRO_TEXT_PRIMARY}; "
            f"border-left: 2px solid {T.RETRO_BORDER}; border-right: 2px solid {T.RETRO_BORDER}; "
            f"border-top: none; border-bottom: none; "
            f"font-family: '{_FONT}', '{_FB}'; font-size: {_FZ_TOOL}px; line-height: 1.8; }}"
        )
        self._tool_cursors: dict[str, QTextCursor] = {}
        self._has_tool_content = False
        self._response_started = False
        self._blink_on = True

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(530)
        self._blink_timer.timeout.connect(self._blink_tick)

    # ── Tool events ───────────────────────────────────────────────────────────

    def append_tool_start(self, name: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ffcc00"))
        f = QFont(_FONT)
        f.setPixelSize(_FZ_TOOL)
        f.setWeight(QFont.Weight.Bold)
        fmt.setFont(f)

        cursor.insertText(f"[EXEC] {name.upper()}\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

        # Save cursor position so we can update this line when done
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

        short = result[:68] + "…" if len(result) > 68 else result
        new_text = f"[DONE] {name.upper()}  {short}" if short else f"[DONE] {name.upper()}"

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#00ff66"))
        f = QFont(_FONT)
        f.setPixelSize(_FZ_TOOL)
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
            f.setPixelSize(_FZ_TOOL)
            div_fmt.setFont(f)
            cursor.insertText("\n" + "─" * 36 + "\n", div_fmt)

            # SYS OUTPUT: header
            hdr_fmt = QTextCharFormat()
            hdr_fmt.setForeground(QColor(T.RETRO_ACCENT))
            hdr_f = QFont(_FONT)
            hdr_f.setPixelSize(_FZ_TOOL)
            hdr_fmt.setFont(hdr_f)
            cursor.insertText("SYS OUTPUT:\n", hdr_fmt)
            self.setTextCursor(cursor)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(T.RETRO_TEXT_PRIMARY))
        f = QFont(_FONT)
        f.setPixelSize(_FZ_RESP)
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
        pass  # streaming cursor placeholder — visual handled by token's trailing █ if needed


class RetroChatView(QWidget):
    """Retro chat area: 40px user input line + flex ConversationLog."""

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

        # ── User input line (h:40) ───────────────────────────────────────────
        input_row = QWidget()
        input_row.setFixedHeight(40)
        input_row.setStyleSheet(
            f"background-color: #040414; border-bottom: 1px solid {T.RETRO_BORDER};"
        )
        ir_lay = QHBoxLayout(input_row)
        ir_lay.setContentsMargins(16, 0, 16, 0)
        ir_lay.setSpacing(0)

        self._prompt_sym = QLabel(">")
        self._prompt_sym.setFixedWidth(20)
        self._prompt_sym.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_INPUT, 0))

        self._input_lbl = QLabel()
        self._input_lbl.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_INPUT, 1))

        self._cursor_lbl = QLabel("█")
        self._cursor_lbl.setStyleSheet(_ss("#00ff66", _FZ_INPUT, 0))
        self._cursor_lbl.setVisible(False)

        ir_lay.addWidget(self._prompt_sym)
        ir_lay.addSpacing(8)
        ir_lay.addWidget(self._input_lbl, stretch=1)
        ir_lay.addWidget(self._cursor_lbl)

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
        self._input_lbl.setStyleSheet(_ss(T.RETRO_TEXT_PRIMARY, _FZ_INPUT, 1))
        self._cursor_lbl.setVisible(False)
        self._blink_timer.stop()

    def _on_state_changed(self, state_name: str):
        self._fsm_state = state_name
        if state_name == "LISTENING":
            self._log.clear_log()
            self._input_lbl.setText("RECORDING...")
            self._input_lbl.setStyleSheet(_ss(T.RETRO_TEXT_MUTED, _FZ_INPUT, 2))
            self._cursor_lbl.setVisible(True)
            self._blink_timer.start()
        elif state_name == "THINKING":
            self._cursor_lbl.setVisible(False)
            self._blink_timer.stop()
            # Show PROCESSING indicator in log
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#ffcc00"))
            f = QFont(_FONT)
            f.setPixelSize(_FZ_TOOL)
            fmt.setFont(f)
            cursor = self._log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("PROCESSING...\n", fmt)
            self._log.setTextCursor(cursor)
        elif state_name == "IDLE":
            # Show placeholder when inactive
            if not self._input_lbl.text() or self._input_lbl.text() == "RECORDING...":
                self._input_lbl.setText("---")
                self._input_lbl.setStyleSheet(_ss(T.RETRO_TEXT_MUTED, _FZ_INPUT, 1))

    def _tick_blink(self):
        self._blink = not self._blink
        self._cursor_lbl.setVisible(self._blink)
