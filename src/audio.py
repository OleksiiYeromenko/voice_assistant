"""Shared audio utilities — ALSA device detection and streaming via arecord.

Uses arecord (ALSA) directly instead of PyAudio/PortAudio, which fails to
enumerate devices reliably on Raspberry Pi 5.  The ``plughw`` ALSA plugin
handles sample-rate conversion (e.g. 44100 → 16000 Hz) transparently.
"""

import logging
import re
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from typing import IO

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ALSA device detection
# ---------------------------------------------------------------------------
def find_alsa_device() -> str:
    """Find the USB capture device via ``arecord -l``.

    Returns an ALSA device string like ``"plughw:1,0"``.
    Falls back to ``"plughw:1,0"`` if parsing fails.
    """
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if "USB" in line:
            m = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
            if m:
                card, device = m.group(1), m.group(2)
                hw = f"plughw:{card},{device}"
                log.info(f"ALSA capture device: {hw}")
                return hw

    log.warning("No USB mic found in arecord -l, falling back to plughw:1,0")
    return "plughw:1,0"


# ---------------------------------------------------------------------------
# Streaming recording via arecord
# ---------------------------------------------------------------------------
@contextmanager
def record_stream(device: str, rate: int = 16000) -> Generator[IO[bytes], None, None]:
    """Open a streaming ``arecord`` subprocess writing raw S16_LE to stdout.

    Usage::

        with record_stream("plughw:1,0") as stream:
            while True:
                raw = stream.read(2560)  # 1280 int16 samples = 80 ms at 16 kHz
                if not raw:
                    break
                audio = np.frombuffer(raw, dtype=np.int16)
    """
    proc = subprocess.Popen(
        [
            "arecord",
            "-D", device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(rate),
            "-t", "raw",
            "-q",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc.stdout
    finally:
        proc.terminate()
        proc.wait()


# ---------------------------------------------------------------------------
# Resampling (kept as utility — not needed when plughw handles conversion)
# ---------------------------------------------------------------------------
def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample *int16* audio from *from_rate* to *to_rate*.

    Returns *int16* numpy array.  No-op when rates already match.
    """
    if from_rate == to_rate:
        return audio

    from scipy import signal as scipy_signal

    num_samples = int(len(audio) * to_rate / from_rate)
    resampled = scipy_signal.resample(audio.astype(np.float32), num_samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16)
