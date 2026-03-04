"""Bottom system status bar: CPU gauge, RAM gauge, temperature, latency panel."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme


class ResourceGauge(QWidget):
    """Vertical stack: label + thin progress bar for CPU or RAM."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        self._label = QLabel(label, self)
        self._label.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 18px;")

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(12)
        self._bar.setTextVisible(False)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout.addWidget(self._label)
        layout.addWidget(self._bar)

    def update_value(self, value: float, max_value: float = 100.0, color: str | None = None):
        pct = int(min(100, max(0, (value / max_value) * 100))) if max_value > 0 else 0
        self._bar.setValue(pct)
        chunk_color = color or theme.cpu_color(pct)
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {chunk_color}; border-radius: 3px; }}"
        )

    def set_label(self, text: str):
        self._label.setText(text)


class TemperatureLabel(QWidget):
    """Temperature display with color-coded value."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        header = QLabel("TEMP", self)
        header.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 18px;")

        self._value = QLabel("—", self)
        self._value.setStyleSheet(f"color: {theme.COLOR_OK}; font-size: 22px; font-weight: bold;")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(header)
        layout.addWidget(self._value)

    def update_temp(self, celsius: float | None):
        if celsius is None:
            self._value.setText("—")
            self._value.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 22px; font-weight: bold;")
        else:
            color = theme.temp_color(celsius)
            self._value.setText(f"{celsius:.0f}°C")
            self._value.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")


class LatencyPanel(QWidget):
    """Three stacked latency labels: STT / TTFT / Total."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(1)

        style = f"color: {theme.TEXT_SECONDARY}; font-family: 'DejaVu Sans Mono'; font-size: 18px;"

        self._stt = QLabel("STT  —", self)
        self._ttft = QLabel("TTFT —", self)
        self._total = QLabel("TOT  —", self)

        for lbl in (self._stt, self._ttft, self._total):
            lbl.setStyleSheet(style)
            layout.addWidget(lbl)

    @staticmethod
    def _fmt(ms: float) -> str:
        if ms <= 0:
            return "—"
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms / 1000:.1f}s"

    def update_metrics(self, stt_ms: float, ttft_ms: float, total_ms: float, model: str):
        self._stt.setText(f"STT  {self._fmt(stt_ms)}")
        self._ttft.setText(f"TTFT {self._fmt(ttft_ms)}")
        self._total.setText(f"TOT  {self._fmt(total_ms)}")


class SysBar(QWidget):
    """Bottom bar (80px): CPU + RAM gauges, temperature, latency panel."""

    def __init__(self, bus, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setObjectName("sys_bar")
        self.setStyleSheet(f"#sys_bar {{ background-color: {theme.BG_ELEVATED}; border-top: 1px solid #2A2A45; }}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._cpu = ResourceGauge("CPU", self)
        self._ram = ResourceGauge("RAM", self)
        self._temp = TemperatureLabel(self)
        self._latency = LatencyPanel(self)

        # Divider
        divider = QWidget(self)
        divider.setFixedWidth(1)
        divider.setStyleSheet(f"background-color: #2A2A45;")
        divider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout.addWidget(self._cpu, stretch=2)
        layout.addWidget(self._ram, stretch=2)
        layout.addWidget(self._temp)
        layout.addWidget(divider)
        layout.addWidget(self._latency)

        self._connect_signals(bus)

    def _connect_signals(self, bus):
        bus.resource_updated.connect(self._on_resource_updated)
        bus.metrics_updated.connect(self._on_metrics_updated)

    def _on_resource_updated(self, cpu: float, ram_used: float, ram_total: float, temp):
        self._cpu.update_value(cpu, 100.0, theme.cpu_color(cpu))
        self._cpu.set_label(f"CPU {cpu:.0f}%")

        ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0
        self._ram.update_value(ram_used, ram_total, theme.cpu_color(ram_pct))
        self._ram.set_label(f"RAM {ram_used:.0f}/{ram_total:.0f}MB")

        self._temp.update_temp(temp)

    def _on_metrics_updated(self, stt_ms: float, ttft_ms: float, llm_ms: float, total_ms: float, model: str):
        self._latency.update_metrics(stt_ms, ttft_ms, total_ms, model)
