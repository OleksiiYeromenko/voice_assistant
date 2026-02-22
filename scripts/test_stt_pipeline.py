#!/usr/bin/env python3
"""STT pipeline diagnostic — record WAVs at different sample rates and transcribe.

Saves WAV files to stt_output/ so you can listen and compare quality.
Runs three tests:
  1. Record at mic's native rate (e.g. 44100 Hz), save WAV
  2. Resample that recording to 16 kHz, save WAV, transcribe (VAD on + off)
  3. Record at 16 kHz directly from mic (ALSA plughw), save WAV, transcribe

Usage:
  uv run scripts/test_stt_pipeline.py
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio

sys.path.insert(0, ".")
from src.audio import find_mic, resample

OUTPUT_DIR = Path("stt_output")
OUTPUT_DIR.mkdir(exist_ok=True)

CHANNELS = 1
CHUNK = 1024
DURATION_S = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_wav(path: Path, audio: np.ndarray, rate: int):
    """Write int16 numpy array to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    print(f"  Saved: {path}  ({len(audio) / rate:.1f}s, {rate} Hz)")


def print_energy(audio: np.ndarray):
    """Print amplitude statistics."""
    abs_audio = np.abs(audio.astype(np.int32))
    print(f"  Energy — mean: {abs_audio.mean():.0f}  max: {abs_audio.max()}  "
          f"samples: {len(audio)}")


def record(pa: pyaudio.PyAudio, device_index: int, rate: int, duration_s: float) -> np.ndarray:
    """Record fixed-duration audio with progress. Returns int16 numpy array."""
    print(f"  Opening stream: device={device_index}, rate={rate}, chunk={CHUNK}")
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )
    print("  Stream opened. Reading ...")

    total_samples = int(rate * duration_s)
    frames = []
    collected = 0
    last_sec = 0
    t_start = time.monotonic()

    while collected < total_samples:
        # Timeout safety: abort after 2x expected duration
        if time.monotonic() - t_start > duration_s * 2:
            print(f"\n  TIMEOUT after {time.monotonic() - t_start:.1f}s "
                  f"(collected {collected}/{total_samples} samples)")
            break

        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        collected += CHUNK

        # Print countdown
        elapsed = time.monotonic() - t_start
        sec = int(elapsed)
        if sec > last_sec:
            last_sec = sec
            remaining = max(0, duration_s - elapsed)
            print(f"  ... {remaining:.0f}s remaining  ({collected} samples)", flush=True)

    stream.stop_stream()
    stream.close()
    elapsed = time.monotonic() - t_start
    print(f"  Done recording: {elapsed:.1f}s, {collected} samples")
    return np.frombuffer(b"".join(frames), dtype=np.int16)


def transcribe(audio_16k: np.ndarray, model, label: str):
    """Transcribe 16 kHz int16 audio with VAD on and off."""
    audio_f32 = audio_16k.astype(np.float32) / 32768.0

    for vad in (True, False):
        tag = "VAD on " if vad else "VAD off"
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            audio_f32, beam_size=1, language="en", vad_filter=vad,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.perf_counter() - t0
        print(f"  [{label}, {tag}] ({elapsed:.2f}s) → '{text}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STT Pipeline Diagnostic")
    print("=" * 60)

    # Detect mic
    pa = pyaudio.PyAudio()
    mic_idx, native_rate = find_mic(pa)
    print(f"\nMic: index={mic_idx}, native rate={native_rate} Hz\n")

    # Load Whisper once
    print("Loading Whisper model (base.en) ...")
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    print("Model ready.\n")

    # ------------------------------------------------------------------
    # Test 1: Record at native rate
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"TEST 1: Record {DURATION_S}s at native rate ({native_rate} Hz)")
    print("  >>> Speak now! <<<")
    audio_native = record(pa, mic_idx, native_rate, DURATION_S)
    print_energy(audio_native)
    wav1 = OUTPUT_DIR / f"01_native_{native_rate}hz.wav"
    save_wav(wav1, audio_native, native_rate)

    # ------------------------------------------------------------------
    # Test 2: Resample to 16 kHz, transcribe
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print(f"TEST 2: Resample {native_rate} → 16000 Hz (scipy), then transcribe")
    audio_resampled = resample(audio_native, native_rate, 16000)
    print_energy(audio_resampled)
    wav2 = OUTPUT_DIR / "02_resampled_16khz.wav"
    save_wav(wav2, audio_resampled, 16000)
    transcribe(audio_resampled, model, "resampled")

    # ------------------------------------------------------------------
    # Test 3: Record at 16 kHz directly
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print(f"TEST 3: Record {DURATION_S}s at 16000 Hz directly (ALSA plughw)")
    print("  >>> Speak now! <<<")
    try:
        audio_direct = record(pa, mic_idx, 16000, DURATION_S)
        print_energy(audio_direct)
        wav3 = OUTPUT_DIR / "03_direct_16khz.wav"
        save_wav(wav3, audio_direct, 16000)
        transcribe(audio_direct, model, "direct 16k")
    except OSError as e:
        print(f"  FAILED to open mic at 16 kHz: {e}")
        print("  This mic does not support 16 kHz directly.")

    pa.terminate()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Done. WAV files saved to stt_output/")
    print("Transfer them to another machine to listen and compare quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
