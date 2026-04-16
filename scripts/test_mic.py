#!/usr/bin/env python3
"""Record short voice clips for wake word model retraining.

Usage:
    uv run scripts/test_mic.py                     # 30 clips, 2s each
    uv run scripts/test_mic.py --count 50 --duration 1.5
    uv run scripts/test_mic.py --out my_clips/     # custom output dir
    uv run scripts/test_mic.py --no-zip            # skip zip

Output: wake_word_clips/clip_001.wav ... + wake_word_clips.zip
"""

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def find_capture_device() -> str:
    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "card" in line.lower():
            import re
            m = re.search(r"card (\d+):.*device (\d+):", line)
            if m:
                return f"plughw:{m.group(1)},{m.group(2)}"
    return "plughw:3,0"


def record_clip(device: str, path: Path, duration: float):
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", str(int(duration)),
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"arecord failed: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Record wake word training clips")
    parser.add_argument("--count", type=int, default=30, help="Number of clips (default: 30)")
    parser.add_argument("--duration", type=float, default=2.0, help="Clip length in seconds (default: 2.0)")
    parser.add_argument("--out", type=Path, default=Path("sounds/wake_word_clips"), help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="ALSA device (auto-detect if omitted)")
    parser.add_argument("--no-zip", action="store_true", help="Skip creating zip archive")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause between clips in seconds (default: 0.5)")
    args = parser.parse_args()

    device = args.device or find_capture_device()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Clips  : {args.count} × {args.duration:.1f}s  →  {out_dir}/")
    print()
    print("Say the wake word once per recording when you hear 'Recording...'")
    print("Press Ctrl+C to stop early.\n")

    recorded: list[Path] = []
    try:
        for i in range(1, args.count + 1):
            input(f"[{i:3}/{args.count}] Press Enter when ready...")
            clip_path = out_dir / f"clip_{i:03d}.wav"
            print(f"  Recording {args.duration:.1f}s ... ", end="", flush=True)
            record_clip(device, clip_path, args.duration)
            print("done")
            recorded.append(clip_path)
            if i < args.count:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nStopped early.")

    if not recorded:
        print("No clips recorded.")
        sys.exit(0)

    print(f"\nRecorded {len(recorded)} clip(s) in {out_dir}/")

    if not args.no_zip:
        zip_path = Path(f"{out_dir}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for clip in recorded:
                zf.write(clip, clip.name)
        print(f"Zipped  → {zip_path}  ({zip_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
