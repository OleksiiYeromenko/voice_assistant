"""Shared audio utilities — microphone detection and resampling.

Handles USB microphones that may only support 44100 Hz natively by
auto-detecting the device and resampling to 16 kHz for Whisper / openWakeWord.
"""

import ctypes
import logging

import numpy as np
import pyaudio
from scipy import signal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silence ALSA warnings (reusable — previously duplicated in stt/engine.py
# and scripts/list_audio_devices.py).
# ---------------------------------------------------------------------------
_ERROR_HANDLER = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)


def _py_error_handler(*_):
    pass


_c_error_handler = _ERROR_HANDLER(_py_error_handler)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so")
    _asound.snd_lib_error_set_handler(_c_error_handler)
except OSError:
    pass  # Not on Linux / ALSA not available


# ---------------------------------------------------------------------------
# Microphone auto-detection
# ---------------------------------------------------------------------------
def find_mic(
    pa: pyaudio.PyAudio,
    preferred_index: int | None = None,
) -> tuple[int, int]:
    """Find a working input device and return ``(device_index, native_sample_rate)``.

    Resolution order:
      1. *preferred_index* — if valid input device, use it.
      2. First device whose name contains ``"usb"`` (case-insensitive).
      3. System default input device.

    Raises ``RuntimeError`` if no microphone is found.
    """
    # 1. Try preferred index
    if preferred_index is not None:
        try:
            info = pa.get_device_info_by_index(preferred_index)
            if info["maxInputChannels"] > 0:
                rate = int(info["defaultSampleRate"])
                log.info(
                    f"Using configured mic: index={preferred_index}, "
                    f"rate={rate} Hz, name='{info['name']}'"
                )
                return preferred_index, rate
        except Exception:
            log.warning(f"Configured mic index {preferred_index} is invalid, auto-detecting...")

    # 2. Scan for USB microphone
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and "usb" in info["name"].lower():
                rate = int(info["defaultSampleRate"])
                log.info(
                    f"Auto-detected USB mic: index={i}, "
                    f"rate={rate} Hz, name='{info['name']}'"
                )
                return i, rate
        except Exception:
            continue

    # 3. Fallback to system default
    try:
        info = pa.get_default_input_device_info()
        rate = int(info["defaultSampleRate"])
        log.warning(
            f"No USB mic found, using default: index={info['index']}, "
            f"rate={rate} Hz, name='{info['name']}'"
        )
        return int(info["index"]), rate
    except Exception as exc:
        raise RuntimeError("No microphone found") from exc


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample *int16* audio from *from_rate* to *to_rate*.

    Returns *int16* numpy array.  No-op when rates already match.
    """
    if from_rate == to_rate:
        return audio

    num_samples = int(len(audio) * to_rate / from_rate)
    resampled = signal.resample(audio.astype(np.float32), num_samples)
    return np.clip(resampled, -32768, 32767).astype(np.int16)
