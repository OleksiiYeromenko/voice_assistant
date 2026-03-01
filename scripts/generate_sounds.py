#!/usr/bin/env python3
"""Generate pre-recorded WAV sound files using Piper TTS.

Usage:
    # Generate a single file
    python scripts/generate_sounds.py --text "Poondik is here" --out sounds/startup/startup_01.wav

    # Generate all default sounds at once
    python scripts/generate_sounds.py --all

    # Override the Piper voice model
    python scripts/generate_sounds.py --all --voice ./voices/en_US-lessac-medium.onnx
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Default Piper voice (matches config.yaml)
DEFAULT_VOICE = "./voices/en_US-hfc_male-medium.onnx"

# All default sounds: category → list of (filename, text) tuples
DEFAULT_SOUNDS: dict[str, list[tuple[str, str]]] = {
    "startup": [
        ("startup_01.wav", "Oh great, I'm awake again."),
    ],
    "greeting": [
        ("greeting_01.wav", "Yeah?"),
        ("greeting_02.wav", "What now?"),
        ("greeting_03.wav", "Go ahead."),
        ("greeting_04.wav", "I'm listening."),
        ("greeting_05.wav", "Sure, speak."),
        ("greeting_06.wav", "Uh huh?"),
        ("greeting_07.wav", "Hit me."),
    ],
    "thinking": [
        ("thinking_01.wav", "Mmm, let me think."),
        ("thinking_02.wav", "Hang on."),
        ("thinking_03.wav", "Give me a second."),
        ("thinking_04.wav", "Oh, that's a good one."),
        ("thinking_05.wav", "Working on it."),
        ("thinking_06.wav", "Right, right, right."),
        ("thinking_07.wav", "Interesting."),
    ],
}


def check_piper() -> bool:
    """Return True if piper is available on PATH."""
    return shutil.which("piper") is not None


def generate_wav(text: str, out_path: Path, voice: str) -> bool:
    """Generate a WAV file from text using Piper TTS.

    Returns True on success, False on failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["piper", "--model", voice, "--output_file", str(out_path)],
        input=text.encode(),
        capture_output=True,
        timeout=60,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        print(f"  ERROR: Piper failed for '{text}': {stderr}", file=sys.stderr)
        return False

    size_kb = out_path.stat().st_size // 1024
    print(f"  OK  {out_path}  ({size_kb} KB)  — \"{text}\"")
    return True


def generate_all(sounds_dir: Path, voice: str) -> int:
    """Generate all default sounds. Returns number of failures."""
    failures = 0
    total = sum(len(files) for files in DEFAULT_SOUNDS.values())
    done = 0

    for category, files in DEFAULT_SOUNDS.items():
        print(f"\n[{category}]")
        for filename, text in files:
            out_path = sounds_dir / category / filename
            if out_path.exists():
                print(f"  SKIP {out_path}  (already exists, use --force to overwrite)")
                done += 1
                continue
            success = generate_wav(text, out_path, voice)
            if not success:
                failures += 1
            done += 1

    print(f"\nDone: {done - failures}/{total} files generated.")
    return failures


def generate_all_force(sounds_dir: Path, voice: str) -> int:
    """Generate all default sounds, overwriting existing files."""
    failures = 0
    total = sum(len(files) for files in DEFAULT_SOUNDS.values())
    done = 0

    for category, files in DEFAULT_SOUNDS.items():
        print(f"\n[{category}]")
        for filename, text in files:
            out_path = sounds_dir / category / filename
            success = generate_wav(text, out_path, voice)
            if not success:
                failures += 1
            done += 1

    print(f"\nDone: {done - failures}/{total} files generated.")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-recorded WAV sounds for the voice assistant using Piper TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", help="Text to synthesize (requires --out)")
    parser.add_argument("--out", help="Output WAV file path (requires --text)")
    parser.add_argument("--all", action="store_true", help="Generate all default sounds")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files (used with --all)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Piper voice model path (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--sounds-dir",
        default="./sounds",
        help="Root directory for sounds (default: ./sounds)",
    )
    args = parser.parse_args()

    if not check_piper():
        print("ERROR: 'piper' not found on PATH. Install piper-tts first.", file=sys.stderr)
        sys.exit(1)

    voice_path = Path(args.voice)
    if not voice_path.exists():
        print(f"ERROR: Voice model not found: {args.voice}", file=sys.stderr)
        sys.exit(1)

    sounds_dir = Path(args.sounds_dir)

    if args.all:
        fn = generate_all_force if args.force else generate_all
        failures = fn(sounds_dir, args.voice)
        sys.exit(1 if failures else 0)

    elif args.text and args.out:
        out_path = Path(args.out)
        success = generate_wav(args.text, out_path, args.voice)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        print(
            "\nError: provide either --all or both --text and --out.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
