#!/usr/bin/env python3
"""Test local Gemma model for audio transcription via Ollama.

Goal: determine whether gemma4:e2b (or another local multimodal model) can
replace faster-whisper for STT, eliminating those heavy dependencies.

Tries two strategies:
  A) Direct audio: send raw WAV bytes as base64 via Ollama multimodal API
  B) Spectrogram: convert WAV to a mel spectrogram PNG and send as image

Usage:
  uv run scripts/test_gemma_stt.py
  uv run scripts/test_gemma_stt.py --model gemma4:e2b --compare --duration 8
"""

import argparse
import base64
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, ".")


def record_audio(duration: float = 10.0) -> tuple[bytes, str]:
    """Record audio from mic, return (raw WAV bytes, temp file path)."""
    from src.config import load_config
    from src.stt.engine import STTEngine

    cfg = load_config()
    alsa_dev = cfg["stt"].get("alsa_device")
    stt = STTEngine(alsa_device=alsa_dev)

    print(f"  Recording (up to {duration:.0f}s, stops on silence)...")
    audio_np = stt.record_utterance(max_duration_s=duration, silence_timeout_s=2.0)

    # Save to temp WAV file (16-bit PCM, 16 kHz, mono)
    tmp = tempfile.mktemp(suffix=".wav", prefix="gemma_stt_")
    with wave.open(tmp, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(16000)
        wf.writeframes(audio_np.tobytes())

    wav_bytes = Path(tmp).read_bytes()
    print(f"  Recorded {len(audio_np) / 16000:.1f}s ({len(wav_bytes)} bytes WAV)")
    return wav_bytes, tmp


def try_direct_audio(model: str, wav_bytes: bytes) -> tuple[str | None, float]:
    """Strategy A: send WAV as base64 audio to Ollama. Returns (text, latency_s)."""
    import ollama

    b64 = base64.b64encode(wav_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": "Transcribe this audio exactly. Output only the spoken words, nothing else.",
            "images": [b64],  # Ollama uses 'images' field for all binary blobs
        }
    ]
    t0 = time.perf_counter()
    try:
        resp = ollama.chat(model=model, messages=messages)
        elapsed = time.perf_counter() - t0
        return resp["message"]["content"].strip(), elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  Strategy A failed: {e}")
        return None, elapsed


def make_spectrogram_png(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to a mel spectrogram PNG in memory."""
    import io
    import struct

    import numpy as np
    from scipy.signal import spectrogram as scipy_spectrogram

    # Parse WAV
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sample_rate = wf.getframerate()

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Compute spectrogram
    freqs, times, Sxx = scipy_spectrogram(audio, fs=sample_rate, nperseg=512, noverlap=384)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Normalize to 0–255 grayscale
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)
    img_array = (Sxx_norm * 255).astype(np.uint8)
    # Flip freq axis so low freqs at bottom
    img_array = np.flipud(img_array)

    # Write as PNG using only stdlib
    height, width = img_array.shape
    buf = io.BytesIO()

    def write_chunk(chunk_type: bytes, data: bytes):
        import zlib
        length = len(data)
        buf.write(struct.pack(">I", length))
        buf.write(chunk_type)
        buf.write(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        buf.write(struct.pack(">I", crc))

    import zlib

    buf.write(b"\x89PNG\r\n\x1a\n")  # PNG signature
    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    write_chunk(b"IHDR", ihdr)
    # IDAT — raw image data with filter byte 0 per row
    raw_rows = b"".join(b"\x00" + bytes(img_array[y]) for y in range(height))
    write_chunk(b"IDAT", zlib.compress(raw_rows))
    write_chunk(b"IEND", b"")

    return buf.getvalue()


def try_spectrogram(model: str, wav_bytes: bytes) -> tuple[str | None, float]:
    """Strategy B: send mel spectrogram as image. Returns (text, latency_s)."""
    import ollama

    print("  Generating spectrogram image...")
    png_bytes = make_spectrogram_png(wav_bytes)
    b64 = base64.b64encode(png_bytes).decode()
    print(f"  Spectrogram PNG: {len(png_bytes)} bytes")

    messages = [
        {
            "role": "user",
            "content": (
                "This is a mel spectrogram of someone speaking. "
                "Low frequencies are at the bottom, time flows left to right. "
                "What words are being spoken? Output only the spoken words."
            ),
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
        print(f"  Strategy B failed: {e}")
        return None, elapsed


def run_whisper_comparison(wav_path: str) -> tuple[str, float]:
    """Run faster-whisper on the saved WAV for comparison."""
    from src.stt.engine import STTEngine

    stt = STTEngine(model_size="base.en")
    stt.load()
    import numpy as np

    with wave.open(wav_path) as wf:
        raw = wf.readframes(wf.getnframes())
    audio_np = np.frombuffer(raw, dtype=np.int16)
    return stt.transcribe(audio_np)


def main():
    parser = argparse.ArgumentParser(description="Test local Gemma model for STT")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model name")
    parser.add_argument("--duration", type=float, default=10.0, help="Max recording seconds")
    parser.add_argument("--compare", action="store_true", help="Also run Whisper for comparison")
    parser.add_argument("--keep-wav", action="store_true", help="Don't delete temp WAV file")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Gemma STT Test — model: {args.model}")
    print("=" * 60)

    # Step 1: Record
    wav_bytes, wav_path = record_audio(duration=args.duration)

    # Step 2: Strategy A — direct audio
    print(f"\n[Strategy A] Sending WAV directly to {args.model}...")
    text_a, lat_a = try_direct_audio(args.model, wav_bytes)
    if text_a is not None:
        print(f"  Result:   {text_a!r}")
        print(f"  Latency:  {lat_a:.2f}s")
    else:
        # Step 3: Strategy B — spectrogram image
        print(f"\n[Strategy B] Sending spectrogram image to {args.model}...")
        text_b, lat_b = try_spectrogram(args.model, wav_bytes)
        if text_b is not None:
            print(f"  Result:   {text_b!r}")
            print(f"  Latency:  {lat_b:.2f}s")
        else:
            print("  Both strategies failed — model may not support multimodal input.")

    # Step 4: Whisper comparison
    if args.compare:
        print("\n[Whisper comparison]")
        whisper_text, whisper_lat = run_whisper_comparison(wav_path)
        print(f"  Result:   {whisper_text!r}")
        print(f"  Latency:  {whisper_lat:.2f}s")

        # Verdict
        gemma_result = text_a if text_a else (text_b if text_b else None)
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        if gemma_result:
            print(f"  Gemma:   {gemma_result!r}")
            print(f"  Whisper: {whisper_text!r}")
            print()
            if gemma_result.lower().strip() == whisper_text.lower().strip():
                print("  => MATCH — Gemma can likely replace Whisper")
            else:
                print("  => DIFFERENT — manual comparison needed")
        else:
            print("  => Gemma produced no output — keep Whisper")

    if not args.keep_wav:
        Path(wav_path).unlink(missing_ok=True)
    else:
        print(f"\n  WAV saved: {wav_path}")


if __name__ == "__main__":
    main()
