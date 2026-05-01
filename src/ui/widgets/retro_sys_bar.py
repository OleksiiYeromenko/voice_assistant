"""Retro-theme bottom stats bar (h:36): CPU | TMP | STT | TPS + system tag.

Spec: docs/design/design_handoff_retro_ui/README.md — Screen 2 / Bottom Stats Bar
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme as T

_FONT = "Press Start 2P"
_FB = "DejaVu Sans Mono"
_FZ = 14  # spec 7px → 14px for RPi display


def _ss_label(color: str) -> str:
    return (
        f"color: {color}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {_FZ}px; letter-spacing: 1px; background: transparent;"
    )


def _pipe() -> QLabel:
    lbl = QLabel("│")
    lbl.setStyleSheet(
        f"color: {T.RETRO_BORDER}; font-family: '{_FONT}', '{_FB}'; "
        f"font-size: {_FZ}px; margin: 0 10px; background: transparent;"
    )
    return lbl


class RetroSysBar(QWidget):
    """36px bottom bar: CPU | TMP | STT | TPS  ···  [radio |] POONDYK.SYS READY"""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setObjectName("retro_sys_bar")
        self.setStyleSheet(
            f"#retro_sys_bar {{ background-color: {T.RETRO_ELEVATED}; "
            f"border-top: 2px solid {T.RETRO_BORDER}; }}"
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(2)

        def _pair(key: str) -> tuple[QLabel, QLabel]:
            k = QLabel(f"{key}:")
            k.setStyleSheet(_ss_label(T.RETRO_TEXT_MUTED))
            v = QLabel("---")
            v.setStyleSheet(_ss_label(T.RETRO_TEXT_SECONDARY))
            return k, v

        k_cpu, self._cpu_val = _pair("CPU")
        k_tmp, self._tmp_val = _pair("TMP")
        k_stt, self._stt_val = _pair("STT")
        k_tps, self._tps_val = _pair("TPS")

        self._radio_lbl = QLabel()
        self._radio_lbl.setStyleSheet(
            f"color: #4fc3f7; font-family: '{_FONT}', '{_FB}'; "
            f"font-size: {_FZ}px; letter-spacing: 1px; background: transparent;"
        )
        self._radio_lbl.setVisible(False)

        self._ready_lbl = QLabel("POONDYK.SYS READY")
        self._ready_lbl.setStyleSheet(_ss_label(T.RETRO_TEXT_MUTED))
        self._ready_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for widget in (k_cpu, self._cpu_val, _pipe(),
                       k_tmp, self._tmp_val, _pipe(),
                       k_stt, self._stt_val, _pipe(),
                       k_tps, self._tps_val):
            lay.addWidget(widget)
            lay.addSpacing(2)

        lay.addStretch()
        lay.addWidget(self._radio_lbl)
        lay.addWidget(self._ready_lbl)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.resource_updated.connect(self._on_resource_updated)
        bus.metrics_updated.connect(self._on_metrics_updated)
        bus.radio_changed.connect(self._on_radio_changed)

    def _on_resource_updated(self, cpu: float, _ram_used: float, _ram_total: float, temp):
        cpu_c = T.RETRO_TEXT_PRIMARY if cpu < 65 else ("#ffcc00" if cpu < 85 else "#ff4444")
        self._cpu_val.setText(f"{cpu:.0f}%")
        self._cpu_val.setStyleSheet(_ss_label(cpu_c))
        if temp is not None:
            tmp_c = T.RETRO_TEXT_PRIMARY if temp < 60 else ("#ffcc00" if temp < 72 else "#ff4444")
            self._tmp_val.setText(f"{temp:.0f}C")
            self._tmp_val.setStyleSheet(_ss_label(tmp_c))

    def _on_metrics_updated(
        self, stt_ms: float, _ttft_ms: float, llm_ms: float, token_count: float, _model: str
    ):
        if stt_ms > 0:
            self._stt_val.setText(f"{stt_ms:.0f}MS")
        if llm_ms > 0 and token_count > 0:
            tps = token_count / (llm_ms / 1000)
            self._tps_val.setText(f"{tps:.1f}")

    def _on_radio_changed(self, station: str):
        if station:
            self._radio_lbl.setText(f"| ♪ {station[:18]} |")
            self._radio_lbl.setVisible(True)
        else:
            self._radio_lbl.setVisible(False)
