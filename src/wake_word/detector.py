"""Wake word detection using openWakeWord.

Continuously listens on the microphone via arecord (ALSA) and yields when
the wake word is detected.  Uses ``plughw`` so ALSA handles sample-rate
conversion to 16 kHz transparently — no manual resampling needed.

The arecord subprocess is terminated before each ``yield`` and restarted
afterwards so that the ALSA device is free for the STT engine to record.
"""

import logging
import subprocess
import threading
from collections.abc import Generator

import numpy as np

from src.audio import find_alsa_device

log = logging.getLogger(__name__)

RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord's native frame size
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample


def _kill_proc(proc: subprocess.Popen):
    """Terminate a subprocess and close its stdout to avoid ResourceWarnings."""
    try:
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2)
    except Exception:
        pass


class WakeWordDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.7,
        alsa_device: str | None = None,
    ):
        import openwakeword
        from openwakeword.model import Model

        # Only download pre-trained models when using a built-in model name.
        # Custom .onnx paths don't need the download step.
        if not model.endswith(".onnx"):
            openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self._alsa_device = alsa_device
        self.oww = Model(wakeword_models=[model])
        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")

    def _ensure_alsa_device(self):
        if self._alsa_device is None:
            self._alsa_device = find_alsa_device()

    def detect_once(
        self,
        timeout_s: float = 120.0,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Listen for a single wake word detection. Returns True if detected.

        Unlike :meth:`listen`, this blocks until detection or timeout and does
        NOT yield.  Designed for use in a background thread during TTS playback
        to enable interruption.

        Args:
            timeout_s: Maximum seconds to listen before giving up.
            stop_event: If set by the caller, the method stops and returns False.
                        Checked every 80 ms (one audio chunk).
        """
        import time

        self._ensure_alsa_device()
        start = time.perf_counter()

        proc = subprocess.Popen(
            [
                "arecord",
                "-D", self._alsa_device,
                "-f", "S16_LE",
                "-c", str(CHANNELS),
                "-r", str(RATE),
                "-t", "raw",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        try:
            while time.perf_counter() - start < timeout_s:
                # Check if caller wants us to stop (every 80 ms)
                if stop_event and stop_event.is_set():
                    return False

                data = proc.stdout.read(CHUNK_BYTES)
                if not data or len(data) < CHUNK_BYTES:
                    break

                audio_16k = np.frombuffer(data, dtype=np.int16)
                prediction = self.oww.predict(audio_16k)

                for model_name, score in prediction.items():
                    if score >= self.threshold:
                        log.info(f"Wake word interrupt '{model_name}': {score:.3f}")
                        self.oww.reset()
                        return True
            return False
        finally:
            _kill_proc(proc)

    def listen(self, heartbeat_s: float = 30.0) -> Generator[float | None, None, None]:
        """Block and yield confidence score each time wake word is detected.

        Yields ``None`` every *heartbeat_s* seconds so the caller can perform
        periodic maintenance (e.g. session rotation) without blocking indefinitely.
        The arecord process keeps running across heartbeat yields — the mic device
        is only released when a real wake word fires (non-None yield).

        Usage::

            detector = WakeWordDetector()
            for val in detector.listen():
                if val is None:
                    continue  # heartbeat — do maintenance if needed
                print(f"Wake word detected! ({val:.2f})")
                # ALSA device is free here — safe to record with STT
                audio = stt.record_utterance()
                ...
        """
        import time

        self._ensure_alsa_device()
        log.info(f"Listening for wake word at {RATE} Hz via {self._alsa_device} ...")

        last_heartbeat = time.monotonic()

        try:
            while True:
                # Open a fresh arecord subprocess each cycle
                proc = subprocess.Popen(
                    [
                        "arecord",
                        "-D", self._alsa_device,
                        "-f", "S16_LE",
                        "-c", str(CHANNELS),
                        "-r", str(RATE),
                        "-t", "raw",
                        "-q",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )

                try:
                    detected = False
                    while not detected:
                        data = proc.stdout.read(CHUNK_BYTES)
                        if not data or len(data) < CHUNK_BYTES:
                            log.warning("arecord stream ended unexpectedly")
                            break

                        audio_16k = np.frombuffer(data, dtype=np.int16)
                        prediction = self.oww.predict(audio_16k)

                        for model_name, score in prediction.items():
                            if score >= self.threshold:
                                log.info(f"Wake word '{model_name}': {score:.3f}")
                                # Stop arecord BEFORE yielding so STT can use the device
                                _kill_proc(proc)
                                self.oww.reset()
                                yield score
                                detected = True
                                break

                        if not detected and heartbeat_s > 0:
                            now = time.monotonic()
                            if (now - last_heartbeat) >= heartbeat_s:
                                last_heartbeat = now
                                yield None  # proc still running; resumes here on next()
                finally:
                    _kill_proc(proc)

        except KeyboardInterrupt:
            log.info("Wake word listener stopped.")
