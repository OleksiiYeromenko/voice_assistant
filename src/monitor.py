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

    def summary(self) -> str:
        parts = []
        if self.stt_ms:
            parts.append(f"stt={self.stt_ms:.0f}ms")
        if self.llm_ms:
            parts.append(f"llm={self.llm_ms:.0f}ms")
        if self.llm_first_token_ms:
            parts.append(f"llm_ttft={self.llm_first_token_ms:.0f}ms")
        if self.tts_first_chunk_ms:
            parts.append(f"tts_first={self.tts_first_chunk_ms:.0f}ms")
        if self.total_ms:
            parts.append(f"total={self.total_ms:.0f}ms")
        if self.model_used:
            parts.append(f"model={self.model_used}")
        return " | ".join(parts)


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