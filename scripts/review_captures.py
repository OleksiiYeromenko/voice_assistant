#!/usr/bin/env python3
"""Review wake word captures and label them as true/false positives.

Usage:
    uv run scripts/review_captures.py

Keys during review:
    y — true positive  → move to true_positives/
    n — false positive → move to false_positives/
    r — replay the audio
    s — skip (leave in raw/ for later)

After collecting false positives, ZIP them for Colab upload:
    cd data/wake_captures && zip -r ~/fp_batch.zip false_positives/
"""

import subprocess
import sys
from pathlib import Path

CAPTURES_DIR = Path("data/wake_captures")
RAW_DIR = CAPTURES_DIR / "raw"
FP_DIR = CAPTURES_DIR / "false_positives"
TP_DIR = CAPTURES_DIR / "true_positives"


def play(path: Path) -> None:
    subprocess.run(["aplay", "-q", str(path)], check=False)


def main() -> None:
    for d in (FP_DIR, TP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.wav"))
    if not files:
        print("No unreviewed captures in data/wake_captures/raw/")
        return

    tp = fp = skipped = 0
    print(f"Reviewing {len(files)} capture(s). Keys: [y]es / [n]o / [r]eplay / [s]kip\n")

    for i, wav in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {wav.name}")
        play(wav)

        while True:
            try:
                key = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\nInterrupted. True positives: {tp}  False positives: {fp}  Skipped: {skipped}")
                sys.exit(0)

            if key == "y":
                wav.rename(TP_DIR / wav.name)
                tp += 1
                break
            elif key == "n":
                wav.rename(FP_DIR / wav.name)
                fp += 1
                break
            elif key == "r":
                play(wav)
            elif key == "s":
                skipped += 1
                break
            else:
                print("  Invalid key. Use y / n / r / s.")

    print(f"\nDone.  True positives: {tp}  False positives: {fp}  Skipped: {skipped}")
    if fp:
        print(f"\nZIP false positives for Colab:")
        print(f"  cd data/wake_captures && zip -r ~/fp_batch.zip false_positives/")


if __name__ == "__main__":
    main()
