"""Wake word detection using openWakeWord.

Continuously listens on the microphone and yields when the wake word is detected.
Handles USB microphones that only support 44100 Hz by resampling to 16 kHz.
"""

import logging
from collections.abc import Generator

import numpy as np
import pyaudio

from src.audio import find_mic, resample

log = logging.getLogger(__name__)

TARGET_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord's native frame size


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.7, mic_device_index: int | None = None):
        import openwakeword
        from openwakeword.model import Model

        # Download pre-trained models on first run
        openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self._preferred_mic_index = mic_device_index
        self.oww = Model(wakeword_models=[model])
        log.info(f"Wake word detector ready: '{model}' (threshold={threshold})")

    def listen(self) -> Generator[float, None, None]:
        """Block and yield confidence score each time wake word is detected.

        Usage:
            detector = WakeWordDetector()
            for confidence in detector.listen():
                print(f"Wake word detected! ({confidence:.2f})")
                audio = record_utterance()
                ...
        """
        pa = pyaudio.PyAudio()
        mic_index, native_rate = find_mic(pa, self._preferred_mic_index)

        # Calculate chunk size at native rate for ~80 ms
        native_chunk = int(native_rate * 80 / 1000)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=native_rate,
            input=True,
            frames_per_buffer=native_chunk,
            input_device_index=mic_index,
        )
        log.info(f"Listening for wake word at {native_rate} Hz ...")

        try:
            while True:
                raw = stream.read(native_chunk, exception_on_overflow=False)
                audio_native = np.frombuffer(raw, dtype=np.int16)

                # Resample to 16 kHz for openWakeWord
                audio_16k = resample(audio_native, native_rate, TARGET_RATE)

                # Ensure exactly CHUNK_SAMPLES (1280) for openWakeWord
                if len(audio_16k) < CHUNK_SAMPLES:
                    audio_16k = np.pad(audio_16k, (0, CHUNK_SAMPLES - len(audio_16k)))
                elif len(audio_16k) > CHUNK_SAMPLES:
                    audio_16k = audio_16k[:CHUNK_SAMPLES]

                prediction = self.oww.predict(audio_16k)

                for model_name, score in prediction.items():
                    if score >= self.threshold:
                        log.info(f"Wake word '{model_name}': {score:.3f}")
                        self.oww.reset()  # Reset state for next detection
                        yield score
        except KeyboardInterrupt:
            log.info("Wake word listener stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
