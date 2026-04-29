"""Retro-theme bottom stats bar (36px): CPU | TMP | STT | TPS + system tag."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui import theme as T

_FONT = "DejaVu Sans Mono"
_S = (
    f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
    "font-size: 15px; letter-spacing: 1px;"
)
_V = (
    f"color: {T.RETRO_TEXT_SECONDARY}; font-family: '{_FONT}'; "
    "font-size: 15px; letter-spacing: 1px;"
)


def _pipe() -> QLabel:
    lbl = QLabel("│")
    lbl.setStyleSheet(
        f"color: {T.RETRO_BORDER}; font-family: '{_FONT}'; font-size: 15px; margin: 0 10px;"
    )
    return lbl


class RetroSysBar(QWidget):
    """44px bottom bar: CPU | TMP | STT | TPS | POONDYK.SYS READY."""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setObjectName("retro_sys_bar")
        self.setStyleSheet(
            f"#retro_sys_bar {{ background-color: {T.RETRO_ELEVATED}; "
            f"border-top: 2px solid {T.RETRO_BORDER}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)

        def _pair(key: str):
            k = QLabel(f"{key}:")
            k.setStyleSheet(_S)
            v = QLabel("---")
            v.setStyleSheet(_V)
            return k, v

        k_cpu, self._cpu_val = _pair("CPU")
        k_tmp, self._tmp_val = _pair("TMP")
        k_stt, self._stt_val = _pair("STT")
        k_tps, self._tps_val = _pair("TPS")

        self._ready_lbl = QLabel("POONDYK.SYS READY")
        self._ready_lbl.setStyleSheet(
            f"color: {T.RETRO_TEXT_MUTED}; font-family: '{_FONT}'; "
            f"font-size: 15px; letter-spacing: 2px;"
        )
        self._ready_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for widget in (k_cpu, self._cpu_val, _pipe(),
                       k_tmp, self._tmp_val, _pipe(),
                       k_stt, self._stt_val, _pipe(),
                       k_tps, self._tps_val):
            layout.addWidget(widget)
            layout.addSpacing(2)

        layout.addStretch()
        layout.addWidget(self._ready_lbl)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.resource_updated.connect(self._on_resource_updated)
        bus.metrics_updated.connect(self._on_metrics_updated)

    def _on_resource_updated(self, cpu: float, ram_used: float, ram_total: float, temp):
        cpu_c = T.RETRO_TEXT_PRIMARY if cpu < 65 else ("#ffcc00" if cpu < 85 else "#ff4444")
        self._cpu_val.setText(f"{cpu:.0f}%")
        self._cpu_val.setStyleSheet(
            f"color: {cpu_c}; font-family: '{_FONT}'; font-size: 15px; letter-spacing: 1px;"
        )
        if temp is not None:
            tmp_c = T.RETRO_TEXT_PRIMARY if temp < 60 else ("#ffcc00" if temp < 72 else "#ff4444")
            self._tmp_val.setText(f"{temp:.0f}C")
            self._tmp_val.setStyleSheet(
                f"color: {tmp_c}; font-family: '{_FONT}'; font-size: 15px; letter-spacing: 1px;"
            )

    def _on_metrics_updated(
        self, stt_ms: float, ttft_ms: float, llm_ms: float, token_count: float, model: str
    ):
        if stt_ms > 0:
            self._stt_val.setText(f"{stt_ms:.0f}MS")
        if llm_ms > 0 and token_count > 0:
            tps = token_count / (llm_ms / 1000)
            self._tps_val.setText(f"{tps:.1f}")
