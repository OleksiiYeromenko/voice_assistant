"""Retro-theme chat view: terminal input line + scrolling log area.

Spec: docs/design/design_handoff_retro_ui/README.md — Screen 2 / Active
"""
import html as _html

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme as T

_FONT = "Press Start 2P"
_FB = "DejaVu Sans Mono"

_FZ_TOOL = 14    # spec 7px — tool status rows, SYS OUTPUT header
_FZ_RESP = 18    # spec 11px — response text
_FZ_INPUT = 14   # spec 8px — user input line


def _ss(color: str, size: int, spacing: int = 1) -> str:
    return (
        f"color: {color}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {size}px; letter-spacing: {spacing}px; background: transparent;"
    )


# ── Tool call row ─────────────────────────────────────────────────────────────

class RetroToolRow(QFrame):
    """Bordered container for a single tool call: [EXEC]→[DONE] with result."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        self._apply_running()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(3)

        self._status_lbl = QLabel(f"[EXEC] {name.upper()}")
        self._status_lbl.setStyleSheet(_ss("#ffcc00", _FZ_TOOL, 2))
        lay.addWidget(self._status_lbl)

        self._result_lbl = QLabel()
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setStyleSheet(_ss(T.RETRO_TEXT_SECONDARY, _FZ_TOOL, 1))
        self._result_lbl.setVisible(False)
        lay.addWidget(self._result_lbl)

    def _apply_running(self):
        self.setStyleSheet(
            "RetroToolRow { border: 1px solid rgba(255, 204, 0, 64); background-color: #100c00; }"
        )

    def set_done(self, result: str):
        self.setStyleSheet(
            "RetroToolRow { border: 1px solid rgba(0, 255, 102, 64); background-color: #001808; }"
        )
        self._status_lbl.setText(f"[DONE] {self._name.upper()}")
        self._status_lbl.setStyleSheet(_ss("#00ff66", _FZ_TOOL, 2))
        if result:
            short = result[:80] + "…" if len(result) > 80 else result
            self._result_lbl.setText(short)
            self._result_lbl.setVisible(True)


# ── Scrolling log area ────────────────────────────────────────────────────────

class RetroConversationLog(QScrollArea):
    """Scrolling log: PROCESSING indicator → bordered tool rows → response."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setStyleSheet(
            f"QScrollArea {{ background-color: {T.RETRO_SURFACE}; "
            f"border: 2px solid {T.RETRO_BORDER}; }}"
            f"QWidget {{ background-color: {T.RETRO_SURFACE}; }}"
        )

        self._container = QWidget()
        self._container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._lay = QVBoxLayout(self._container)
        self._lay.setContentsMargins(14, 16, 14, 12)
        self._lay.setSpacing(8)
        self._lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self._container)

        self._thinking_lbl: QLabel | None = None
        self._tool_rows: dict[str, RetroToolRow] = {}
        self._divider: QFrame | None = None
        self._resp_section: QWidget | None = None
        self._resp_lbl: QLabel | None = None
        self._resp_text = ""

        # Streaming cursor blink
        self._blink_on = True
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(530)
        self._blink_timer.timeout.connect(self._blink_tick)

    # ── Public API ────────────────────────────────────────────────────────────

    def show_thinking(self):
        """Show PROCESSING... indicator (THINKING state, no tools yet)."""
        if self._thinking_lbl is not None:
            return
        lbl = QLabel("PROCESSING...")
        lbl.setStyleSheet(_ss("#ffcc00", _FZ_TOOL, 2))
        self._thinking_lbl = lbl
        self._lay.addWidget(lbl)
        self._scroll_to_bottom()

    def append_tool_start(self, name: str):
        self._remove_thinking()
        row = RetroToolRow(name)
        self._tool_rows[name] = row
        self._lay.addWidget(row)
        self._scroll_to_bottom()

    def update_tool_done(self, name: str, result: str):
        row = self._tool_rows.get(name)
        if row:
            row.set_done(result)
        self._scroll_to_bottom()

    def append_token(self, token: str):
        if self._resp_section is None:
            self._remove_thinking()
            # Dashed divider when tools preceded response
            if self._tool_rows:
                div = QFrame()
                div.setFixedHeight(2)
                div.setStyleSheet(
                    f"border: none; border-top: 1px dashed {T.RETRO_BORDER}; "
                    f"background: transparent; margin: 10px 0;"
                )
                self._divider = div
                self._lay.addWidget(div)

            section = QWidget()
            section.setStyleSheet("background: transparent;")
            sec_lay = QVBoxLayout(section)
            sec_lay.setContentsMargins(0, 8, 0, 0)
            sec_lay.setSpacing(12)

            hdr = QLabel("SYS OUTPUT:")
            hdr.setStyleSheet(_ss(T.RETRO_ACCENT, _FZ_TOOL, 2))
            sec_lay.addWidget(hdr)

            resp = QLabel()
            resp.setWordWrap(True)
            resp.setTextFormat(Qt.TextFormat.RichText)
            resp.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            sec_lay.addWidget(resp)

            self._resp_section = section
            self._resp_lbl = resp
            self._lay.addWidget(section)
            self._blink_timer.start()

        self._resp_text += token
        self._resp_lbl.setText(self._resp_html(self._resp_text, True))
        self._scroll_to_bottom()

    def start_stream(self):
        self._blink_timer.start()

    def stop_stream(self):
        self._blink_timer.stop()
        if self._resp_lbl:
            self._resp_lbl.setText(self._resp_html(self._resp_text, False))

    def clear_log(self):
        self._remove_thinking()
        for row in list(self._tool_rows.values()):
            row.setParent(None)
            row.deleteLater()
        self._tool_rows.clear()
        if self._divider:
            self._divider.setParent(None)
            self._divider.deleteLater()
            self._divider = None
        if self._resp_section:
            self._resp_section.setParent(None)
            self._resp_section.deleteLater()
            self._resp_section = None
            self._resp_lbl = None
        self._resp_text = ""
        self._blink_timer.stop()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resp_html(self, text: str, cursor_visible: bool = False) -> str:
        escaped = _html.escape(text)
        cursor_color = T.RETRO_ACCENT if cursor_visible else "transparent"
        cursor_span = f' <span style="color: {cursor_color};">█</span>'
        return (
            f"<p style=\"font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ_RESP}px; color: {T.RETRO_TEXT_PRIMARY}; "
            f'letter-spacing: 1px; line-height: 2.2; margin: 0;">'
            f"{escaped}{cursor_span}</p>"
        )

    def _remove_thinking(self):
        if self._thinking_lbl is not None:
            self._thinking_lbl.setParent(None)
            self._thinking_lbl.deleteLater()
            self._thinking_lbl = None

    def _scroll_to_bottom(self):
        QTimer.singleShot(10, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))

    def _blink_tick(self):
        self._blink_on = not self._blink_on
        if self._resp_lbl and self._resp_text:
            self._resp_lbl.setText(self._resp_html(self._resp_text, self._blink_on))


# ── Chat view (input line + log) ──────────────────────────────────────────────

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
        input_row.setFixedHeight(46)
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
        ir_lay.addWidget(self._input_lbl)
        ir_lay.addWidget(self._cursor_lbl)
        ir_lay.addStretch()

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
            self._log.show_thinking()
        elif state_name == "IDLE":
            self._cursor_lbl.setVisible(False)
            self._blink_timer.stop()
            if not self._input_lbl.text() or self._input_lbl.text() == "RECORDING...":
                self._input_lbl.setText("---")
                self._input_lbl.setStyleSheet(_ss(T.RETRO_TEXT_MUTED, _FZ_INPUT, 1))

    def _tick_blink(self):
        self._blink = not self._blink
        self._cursor_lbl.setVisible(self._blink)
