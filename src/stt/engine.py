"""Speech-to-Text using faster-whisper.

Records audio via arecord (ALSA) after wake word, detects silence to stop,
then transcribes.  The ``plughw`` ALSA plugin resamples from the mic's native
rate (e.g. 44100 Hz) to 16 kHz transparently.
"""

import logging
import time

import numpy as np

from src.audio import find_alsa_device, record_stream

log = logging.getLogger(__name__)

RATE = 16000  # arecord records at 16 kHz via plughw
CHANNELS = 1
CHUNK_SAMPLES = 1024
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes per sample


class STTEngine:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
        vad_filter: bool = True,
        alsa_device: str | None = None,
    ):
        self.model_size = model_size
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter
        self.model = None
        self._alsa_device = alsa_device

        self._device = device
        self._compute_type = compute_type

    def _ensure_alsa_device(self):
        if self._alsa_device is None:
            self._alsa_device = find_alsa_device()

    def load(self):
        """Load the whisper model. Call once at startup."""
        if self.model is not None:
            return
        from faster_whisper import WhisperModel

        log.info(f"Loading STT model: {self.model_size} ...")
        start = time.perf_counter()
        self.model = WhisperModel(
            self.model_size,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=4,
            local_files_only=True,
        )
        elapsed = time.perf_counter() - start
        log.info(f"STT model loaded in {elapsed:.1f}s")

    def record_utterance(
        self,
        max_duration_s: float = 10.0,
        silence_timeout_s: float = 1.5,
        silence_threshold: int = 500,
    ) -> np.ndarray:
        """Record from mic until silence is detected. Returns int16 numpy array at 16 kHz."""
        self._ensure_alsa_device()

        frames: list[bytes] = []
        silent_chunks = 0
        max_silent_chunks = int(RATE / CHUNK_SAMPLES * silence_timeout_s)
        max_chunks = int(RATE / CHUNK_SAMPLES * max_duration_s)

        log.info(f"Recording at {RATE} Hz via {self._alsa_device} ...")

        with record_stream(self._alsa_device, RATE) as stream:
            for _ in range(max_chunks):
                data = stream.read(CHUNK_BYTES)
                if not data or len(data) < CHUNK_BYTES:
                    log.warning("arecord stream ended unexpectedly")
                    break
                frames.append(data)

                # Simple energy-based silence detection
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_chunk).mean()

                if energy < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks >= max_silent_chunks:
                        log.info("Silence detected, stopping recording.")
                        break
                else:
                    silent_chunks = 0

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        duration = len(audio) / RATE
        log.info(f"Recorded {duration:.1f}s of audio")
        return audio

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe int16 numpy audio. Returns (text, elapsed_seconds)."""
        if self.model is None:
            self.load()

        # faster-whisper expects float32 in [-1, 1]
        audio_f32 = audio.astype(np.float32) / 32768.0

        start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_f32,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - start

        log.info(f"STT: '{text}' ({elapsed:.2f}s)")
        return text.strip(), elapsed

    def record_and_transcribe(
        self,
        max_duration_s: float = 10.0,
        silence_timeout_s: float = 1.5,
    ) -> tuple[str, float, float]:
        """Convenience: record then transcribe. Returns (text, record_time, transcribe_time)."""
        t0 = time.perf_counter()
        audio = self.record_utterance(
            max_duration_s=max_duration_s,
            silence_timeout_s=silence_timeout_s,
        )
        record_time = time.perf_counter() - t0

        text, transcribe_time = self.transcribe(audio)
        return text, record_time, transcribe_time
