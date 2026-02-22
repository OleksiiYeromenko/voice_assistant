#!/usr/bin/env python3
"""STT pipeline diagnostic — record WAVs at different sample rates and transcribe.

Uses arecord directly (bypasses PyAudio/PortAudio) since PortAudio often fails
to enumerate ALSA devices on Raspberry Pi after process cleanup.

Saves WAV files to stt_output/ so you can listen and compare quality.
Runs three tests:
  1. Record at native rate (44100 Hz) via arecord, save WAV
  2. Resample that recording to 16 kHz (scipy), save WAV, transcribe
  3. Record at 16 kHz directly via arecord (ALSA plughw resampling), transcribe

Usage:
  uv run scripts/test_stt_pipeline.py
"""

import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.audio import resample

OUTPUT_DIR = Path("stt_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DURATION_S = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_alsa_capture_device() -> str:
    """Find the USB capture device using arecord -l. Returns ALSA hw string."""
    import re

    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    print("  arecord -l output:")
    for line in result.stdout.strip().splitlines():
        print(f"    {line}")

    # Parse "card N: ... device M: ..." using regex
    for line in result.stdout.splitlines():
        if "USB" in line:
            m = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
            if m:
                card, device = m.group(1), m.group(2)
                hw = f"plughw:{card},{device}"
                print(f"\n  Using ALSA device: {hw}")
                return hw

    # Fallback: try card 1
    print("\n  No USB device parsed, falling back to plughw:1,0")
    return "plughw:1,0"


def record_arecord(device: str, rate: int, duration_s: int, output_path: Path) -> np.ndarray:
    """Record using arecord. Returns int16 numpy array."""
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-c", "1",
        "-r", str(rate),
        "-d", str(duration_s),
        "-t", "wav",
        str(output_path),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 5)

    if result.returncode != 0:
        print(f"  arecord stderr: {result.stderr.strip()}")
        raise RuntimeError(f"arecord failed: {result.stderr.strip()}")

    if result.stderr.strip():
        # arecord prints info to stderr even on success
        for line in result.stderr.strip().splitlines():
            print(f"    {line}")

    # Read back the WAV
    with wave.open(str(output_path), "rb") as wf:
        n_frames = wf.getnframes()
        actual_rate = wf.getframerate()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    print(f"  Recorded: {len(audio)} samples, {len(audio)/actual_rate:.1f}s at {actual_rate} Hz")
    return audio


def print_energy(audio: np.ndarray):
    """Print amplitude statistics."""
    abs_audio = np.abs(audio.astype(np.int32))
    print(f"  Energy — mean: {abs_audio.mean():.0f}  max: {abs_audio.max()}  "
          f"samples: {len(audio)}")


def save_wav(path: Path, audio: np.ndarray, rate: int):
    """Write int16 numpy array to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    print(f"  Saved: {path}  ({len(audio) / rate:.1f}s, {rate} Hz)")


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
    print("STT Pipeline Diagnostic (using arecord)")
    print("=" * 60)

    # Find ALSA device first (read-only, doesn't open it)
    print()
    device = find_alsa_capture_device()

    # Kill ALL processes holding the sound card
    print("\nReleasing audio device ...")
    card_num = device.split(":")[1].split(",")[0]
    dev_path = f"/dev/snd/pcmC{card_num}D0c"
    print(f"  Checking {dev_path} ...")

    subprocess.run(["killall", "-q", "arecord", "aplay"], capture_output=True)

    result = subprocess.run(["fuser", "-v", dev_path], capture_output=True, text=True)
    fuser_out = (result.stdout + result.stderr).strip()
    if fuser_out:
        print(f"  {fuser_out}")
        print(f"  Running: fuser -k {dev_path}")
        subprocess.run(["fuser", "-k", dev_path], capture_output=True)
        time.sleep(2)
        print("  Done.")
    else:
        print("  Device is free.")

    # Load Whisper once
    print("\nLoading Whisper model (base.en) ...")
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    print("Model ready.\n")

    # ------------------------------------------------------------------
    # Test 1: Record at 44100 Hz (native)
    # ------------------------------------------------------------------
    print("-" * 60)
    native_rate = 44100
    print(f"TEST 1: Record {DURATION_S}s at {native_rate} Hz (native rate)")
    print("  >>> Speak now! <<<")
    wav1 = OUTPUT_DIR / f"01_native_{native_rate}hz.wav"
    audio_native = record_arecord(device, native_rate, DURATION_S, wav1)
    print_energy(audio_native)

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
    # Test 3: Record at 16 kHz directly (ALSA plughw does the resampling)
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print(f"TEST 3: Record {DURATION_S}s at 16000 Hz directly (ALSA plughw resamples)")
    print("  >>> Speak now! <<<")
    try:
        wav3 = OUTPUT_DIR / "03_direct_16khz.wav"
        audio_direct = record_arecord(device, 16000, DURATION_S, wav3)
        print_energy(audio_direct)
        transcribe(audio_direct, model, "direct 16k")
    except Exception as e:
        print(f"  FAILED: {e}")

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
