#!/usr/bin/env python3
"""Test local Gemma model for audio transcription via Ollama.

Goal: determine whether gemma4:e2b (or another local multimodal model) can
replace faster-whisper for STT, eliminating those heavy dependencies.

Records audio via arecord and sends raw WAV bytes as base64 to the Ollama
multimodal API.

Usage:
  uv run scripts/test_gemma_stt.py
  uv run scripts/test_gemma_stt.py --model gemma4:e2b --compare --duration 8
"""

import argparse
import base64
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, ".")


def find_alsa_capture_device() -> str:
    """Find the USB capture device using arecord -l. Returns ALSA hw string."""
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    print("  arecord -l output:")
    for line in result.stdout.strip().splitlines():
        print(f"    {line}")

    for line in result.stdout.splitlines():
        if "USB" in line:
            m = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
            if m:
                hw = f"plughw:{m.group(1)},{m.group(2)}"
                print(f"\n  Using ALSA device: {hw}")
                return hw

    print("\n  No USB device parsed, falling back to plughw:1,0")
    return "plughw:1,0"


def release_device(device: str):
    """Kill any processes holding the capture device, including their parents."""
    import os
    import signal

    card_num = device.split(":")[1].split(",")[0]
    dev_path = f"/dev/snd/pcmC{card_num}D0c"

    result = subprocess.run(["fuser", dev_path], capture_output=True, text=True)
    pids = [int(p) for p in (result.stdout + result.stderr).split() if p.strip().isdigit()]

    if pids:
        print(f"  Processes holding {dev_path}: {pids}")
        # Kill parent processes too (e.g. the voice assistant that spawned arecord)
        parent_pids = set()
        for pid in pids:
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                            if ppid > 1:
                                parent_pids.add(ppid)
            except OSError:
                pass
        for ppid in parent_pids:
            print(f"  Killing parent PID {ppid}")
            try:
                os.kill(ppid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        subprocess.run(["fuser", "-k", dev_path], capture_output=True)
        time.sleep(2)
    else:
        print(f"  {dev_path} is free")


def record_arecord(device: str, duration_s: float) -> tuple[bytes, str]:
    """Record via arecord at 16 kHz. Returns (WAV bytes, temp file path)."""
    tmp = tempfile.mktemp(suffix=".wav", prefix="gemma_stt_")
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-c", "1",
        "-r", "16000",
        "-d", str(int(duration_s)),
        "-t", "wav",
        tmp,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(duration_s) + 5)

    if result.returncode != 0:
        raise RuntimeError(f"arecord failed: {result.stderr.strip()}")

    wav_bytes = Path(tmp).read_bytes()
    with wave.open(tmp) as wf:
        n_frames = wf.getnframes()
    print(f"  Recorded {n_frames / 16000:.1f}s ({len(wav_bytes)} bytes WAV)")
    return wav_bytes, tmp


def try_direct_audio(model: str, wav_bytes: bytes) -> tuple[str | None, float]:
    """Send WAV as base64 to Ollama. Returns (text, latency_s)."""
    import ollama

    b64 = base64.b64encode(wav_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": "Transcribe this audio exactly. Output only the spoken words, nothing else.",
            "images": [b64],
        }
    ]
    t0 = time.perf_counter()
    try:
        resp = ollama.chat(model=model, messages=messages)
        elapsed = time.perf_counter() - t0
        return resp["message"]["content"].strip(), elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  Failed: {e}")
        return None, elapsed


def run_whisper_comparison(wav_path: str) -> tuple[str, float]:
    """Run faster-whisper on the saved WAV for comparison."""
    import numpy as np
    from faster_whisper import WhisperModel

    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    with wave.open(wav_path) as wf:
        raw = wf.readframes(wf.getnframes())
    audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_f32, beam_size=1, language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Test local Gemma model for STT")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model name")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording seconds")
    parser.add_argument("--compare", action="store_true", help="Also run Whisper for comparison")
    parser.add_argument("--keep-wav", action="store_true", help="Don't delete temp WAV file")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Gemma STT Test — model: {args.model}")
    print("=" * 60)

    device = find_alsa_capture_device()
    release_device(device)

    print(f"\n>>> Speak now! Recording {args.duration:.0f}s... <<<")
    wav_bytes, wav_path = record_arecord(device, args.duration)

    print(f"\nSending WAV to {args.model}...")
    text, latency = try_direct_audio(args.model, wav_bytes)
    if text is not None:
        print(f"  Result:  {text!r}")
        print(f"  Latency: {latency:.2f}s")
    else:
        print("  No output — model may not support audio input.")

    if args.compare:
        print("\n[Whisper comparison]")
        whisper_text, whisper_lat = run_whisper_comparison(wav_path)
        print(f"  Result:  {whisper_text!r}")
        print(f"  Latency: {whisper_lat:.2f}s")

        print("\n" + "=" * 60)
        if text:
            if text.lower().strip() == whisper_text.lower().strip():
                print("=> MATCH — Gemma can likely replace Whisper")
            else:
                print("=> DIFFERENT — manual comparison needed")
                print(f"   Gemma:   {text!r}")
                print(f"   Whisper: {whisper_text!r}")
        else:
            print("=> Gemma produced no output — keep Whisper")

    if args.keep_wav:
        print(f"\nWAV saved: {wav_path}")
    else:
        Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
