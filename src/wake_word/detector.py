"""Wake word detection using openWakeWord.

Continuously listens on the microphone via arecord (ALSA) and yields when
the wake word is detected.  Uses ``plughw`` so ALSA handles sample-rate
conversion to 16 kHz transparently — no manual resampling needed.

The arecord subprocess is terminated before each ``yield`` and restarted
afterwards so that the ALSA device is free for the STT engine to record.
"""

import logging
import subprocess
from collections.abc import Generator

import numpy as np

from src.audio import find_alsa_device

log = logging.getLogger(__name__)

RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord's native frame size
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample


class WakeWordDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.7,
        alsa_device: str | None = None,
    ):
        import openwakeword
        from openwakeword.model import Model

        # Download pre-trained models on first run
        openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self._alsa_device = alsa_device
        self.oww = Model(wakeword_models=[model])
        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")

    def _ensure_alsa_device(self):
        if self._alsa_device is None:
            self._alsa_device = find_alsa_device()

    def detect_once(self, timeout_s: float = 120.0) -> bool:
        """Listen for a single wake word detection. Returns True if detected.

        Unlike :meth:`listen`, this blocks until detection or timeout and does
        NOT yield.  Designed for use in a background thread during TTS playback
        to enable interruption.
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
            if proc.poll() is None:
                proc.terminate()
                proc.wait()

    def listen(self) -> Generator[float, None, None]:
        """Block and yield confidence score each time wake word is detected.

        Usage::

            detector = WakeWordDetector()
            for confidence in detector.listen():
                print(f"Wake word detected! ({confidence:.2f})")
                # ALSA device is free here — safe to record with STT
                audio = stt.record_utterance()
                ...
        """
        self._ensure_alsa_device()
        log.info(f"Listening for wake word at {RATE} Hz via {self._alsa_device} ...")

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
                    while True:
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
                                proc.terminate()
                                proc.wait()
                                self.oww.reset()
                                yield score
                                # After yield returns, break inner loop
                                # to reopen arecord at the top of outer loop
                                break
                        else:
                            # No wake word this chunk — keep reading
                            continue
                        # Wake word fired (inner for-loop hit break) — exit inner while
                        break
                finally:
                    # Ensure cleanup if inner loop exits unexpectedly
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()

        except KeyboardInterrupt:
            log.info("Wake word listener stopped.")
