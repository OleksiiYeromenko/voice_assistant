"""Wake word detection using openWakeWord.

Continuously listens on the microphone and yields when the wake word is detected.
"""

import logging
import struct
import time
from collections.abc import Generator

import numpy as np
import pyaudio

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openWakeWord's native frame size


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.7, mic_device_index: int | None = None):
        import openwakeword
        from openwakeword.model import Model

        # Download pre-trained models on first run
        openwakeword.utils.download_models()

        self.model_name = model
        self.threshold = threshold
        self.mic_device_index = mic_device_index
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
        log.info("Listening for wake word...")

        try:
            while True:
                raw = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                # Convert to int16 numpy array
                audio = np.frombuffer(raw, dtype=np.int16)
                # openWakeWord expects float32 in [-1, 1] range
                # But the predict method handles int16 directly
                prediction = self.oww.predict(audio)

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