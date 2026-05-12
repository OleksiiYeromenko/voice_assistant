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
import wave
from collections import deque
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np

from src.audio import find_alsa_device

log = logging.getLogger(__name__)

RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord's native frame size
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample
PREBUFFER_CHUNKS = 38  # 3 seconds of audio (38 × 80 ms)


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
        record_detections: bool = False,
        verifier_model: str | None = None,
        verifier_threshold: float = 0.1,
    ):
        import pickle

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
        self._model_stem = Path(model).stem
        self._verifier = None
        self._verifier_threshold = verifier_threshold

        if verifier_model:
            with open(verifier_model, "rb") as f:
                self._verifier = pickle.load(f)
            log.info(f"Verifier loaded: {verifier_model} (gate={verifier_threshold})")

        self._record_detections = record_detections
        if record_detections:
            self._capture_dir = Path("data/wake_captures")
            for sub in ("raw", "false_positives", "true_positives"):
                (self._capture_dir / sub).mkdir(parents=True, exist_ok=True)
            log.info(f"Wake capture recording enabled → {self._capture_dir}")

        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")

    def _save_capture(self, ring: deque, raw_score: float, verifier_score: float | None = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_part = f"{raw_score:.3f}_{verifier_score:.3f}" if verifier_score is not None else f"{raw_score:.3f}"
        path = self._capture_dir / "raw" / f"{ts}_{score_part}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(ring))
        log.debug(f"Saved wake capture: {path.name}")

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
                "-D",
                self._alsa_device,
                "-f",
                "S16_LE",
                "-c",
                str(CHANNELS),
                "-r",
                str(RATE),
                "-t",
                "raw",
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

                for model_name, raw_score in prediction.items():
                    verifier_score = None
                    effective_score = raw_score
                    if self._verifier and raw_score >= self._verifier_threshold:
                        features = self.oww.preprocessor.get_features(
                            self.oww.model_inputs[self._model_stem]
                        )
                        verifier_score = float(self._verifier.predict_proba(features)[0][-1])
                        effective_score = verifier_score
                    if effective_score >= self.threshold:
                        log.info(
                            f"Wake word interrupt '{model_name}': "
                            f"raw={raw_score:.3f} verifier={verifier_score}"
                        )
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
        ring: deque[bytes] = deque(maxlen=PREBUFFER_CHUNKS)

        try:
            while True:
                # Open a fresh arecord subprocess each cycle
                proc = subprocess.Popen(
                    [
                        "arecord",
                        "-D",
                        self._alsa_device,
                        "-f",
                        "S16_LE",
                        "-c",
                        str(CHANNELS),
                        "-r",
                        str(RATE),
                        "-t",
                        "raw",
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

                        ring.append(data)
                        audio_16k = np.frombuffer(data, dtype=np.int16)
                        prediction = self.oww.predict(audio_16k)

                        for model_name, raw_score in prediction.items():
                            verifier_score = None
                            effective_score = raw_score
                            if self._verifier and raw_score >= self._verifier_threshold:
                                features = self.oww.preprocessor.get_features(
                                    self.oww.model_inputs[self._model_stem]
                                )
                                verifier_score = float(
                                    self._verifier.predict_proba(features)[0][-1]
                                )
                                effective_score = verifier_score
                            if effective_score >= self.threshold:
                                log.info(
                                    f"Wake word '{model_name}': "
                                    f"raw={raw_score:.3f} verifier={verifier_score}"
                                )
                                # Stop arecord BEFORE yielding so STT can use the device
                                _kill_proc(proc)
                                if self._record_detections:
                                    self._save_capture(ring, raw_score, verifier_score)
                                self.oww.reset()
                                yield effective_score
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
