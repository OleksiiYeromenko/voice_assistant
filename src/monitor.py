"""Resource monitor — CPU, RAM, temperature, latency tracking."""

import time
import logging
from dataclasses import dataclass, field

import psutil

log = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Timing for one full pipeline pass."""
    wake_ms: float = 0.0
    stt_ms: float = 0.0
    router_ms: float = 0.0
    llm_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    model_used: str = ""

    @staticmethod
    def _fmt(ms: float) -> str:
        """Format milliseconds as human-readable duration."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        return f"{ms / 1000:.1f}s"

    def summary(self) -> str:
        stages = []
        if self.stt_ms:
            stages.append(f"STT {self._fmt(self.stt_ms)}")
        if self.llm_ms:
            ttft = f" (ttft {self._fmt(self.llm_first_token_ms)})" if self.llm_first_token_ms else ""
            stages.append(f"LLM {self._fmt(self.llm_ms)}{ttft}")
        if self.tts_first_chunk_ms:
            stages.append(f"TTS {self._fmt(self.tts_first_chunk_ms)}")
        pipeline = " → ".join(stages)
        tail = []
        if self.total_ms:
            tail.append(f"total {self._fmt(self.total_ms)}")
        if self.model_used:
            tail.append(self.model_used)
        suffix = " │ ".join(tail)
        return f"{pipeline} │ {suffix}" if suffix else pipeline


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    temp_celsius: float | None

    def summary(self) -> str:
        t = f" temp={self.temp_celsius:.1f}°C" if self.temp_celsius else ""
        return (
            f"cpu={self.cpu_percent:.0f}% "
            f"ram={self.ram_used_mb:.0f}/{self.ram_total_mb:.0f}MB"
            f"{t}"
        )


def snapshot() -> ResourceSnapshot:
    mem = psutil.virtual_memory()
    temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            temp = temps["cpu_thermal"][0].current
    except (AttributeError, KeyError):
        pass

    return ResourceSnapshot(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        ram_used_mb=mem.used / 1024 / 1024,
        ram_total_mb=mem.total / 1024 / 1024,
        temp_celsius=temp,
    )


def check_thresholds(snap: ResourceSnapshot, thresholds: dict) -> list[str]:
    """Check resource snapshot against configured thresholds. Returns warning messages."""
    warnings = []

    ram_pct = (snap.ram_used_mb / snap.ram_total_mb) * 100
    if ram_pct > thresholds.get("ram_percent", 85):
        warnings.append(
            f"RAM usage high: {ram_pct:.0f}% ({snap.ram_used_mb:.0f}/{snap.ram_total_mb:.0f} MB)"
        )

    if snap.cpu_percent > thresholds.get("cpu_percent", 95):
        warnings.append(f"CPU usage high: {snap.cpu_percent:.0f}%")

    if snap.temp_celsius and snap.temp_celsius > thresholds.get("temp_celsius", 80):
        warnings.append(f"CPU temperature high: {snap.temp_celsius:.1f}°C")

    for w in warnings:
        log.warning(f"RESOURCE ALERT: {w}")

    return warnings


class Timer:
    """Simple context-manager timer that records elapsed ms."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def mark(self) -> float:
        """Return ms since __enter__ without stopping."""
        return (time.perf_counter() - self._start) * 1000