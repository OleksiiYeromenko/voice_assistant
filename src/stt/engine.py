"""Speech-to-Text using faster-whisper.

Records audio after wake word, detects silence to stop, then transcribes.
"""

import ctypes
import logging
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio

log = logging.getLogger(__name__)

# Silence ALSA warnings
_ERROR_HANDLER = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)
def _py_error_handler(*_): pass
_c_error_handler = _ERROR_HANDLER(_py_error_handler)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so")
    _asound.snd_lib_error_set_handler(_c_error_handler)
except OSError:
    pass

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1024


class STTEngine:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str = "en",
        vad_filter: bool = True,
        mic_device_index: int | None = None,
    ):
        self.model_size = model_size
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter
        self.model = None
        self.mic_device_index = mic_device_index

        self._device = device
        self._compute_type = compute_type

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
        )
        elapsed = time.perf_counter() - start
        log.info(f"STT model loaded in {elapsed:.1f}s")

    def record_utterance(
        self,
        max_duration_s: float = 10.0,
        silence_timeout_s: float = 1.5,
        silence_threshold: int = 500,
    ) -> np.ndarray:
        """Record from mic until silence is detected. Returns int16 numpy array."""
        pa = pyaudio.PyAudio()
        stream_kwargs = {
            "format": pyaudio.paInt16,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "input": True,
            "frames_per_buffer": CHUNK_SAMPLES,
        }
        if self.mic_device_index is not None:
            stream_kwargs["input_device_index"] = self.mic_device_index
        stream = pa.open(**stream_kwargs)

        frames: list[bytes] = []
        silent_chunks = 0
        max_silent_chunks = int(SAMPLE_RATE / CHUNK_SAMPLES * silence_timeout_s)
        max_chunks = int(SAMPLE_RATE / CHUNK_SAMPLES * max_duration_s)

        log.info("Recording utterance...")
        try:
            for _ in range(max_chunks):
                data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
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
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        duration = len(audio) / SAMPLE_RATE
        log.info(f"Recorded {duration:.1f}s of audio")
        return audio

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe int16 numpy audio. Returns (text, elapsed_seconds)."""
        if self.model is None:
            self.load()

        # faster-whisper accepts file path or numpy array
        # Convert int16 to float32 in [-1, 1]
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