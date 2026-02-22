#!/usr/bin/env python3
"""List audio devices — uses ALSA tools (arecord/aplay) instead of PyAudio.

Usage: uv run scripts/list_audio_devices.py
"""

import re
import subprocess
import sys


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def list_devices(direction: str):
    """List capture or playback devices. direction = 'capture' or 'playback'."""
    cmd = ["arecord", "-l"] if direction == "capture" else ["aplay", "-l"]
    label = "INPUT (microphone)" if direction == "capture" else "OUTPUT (speaker)"
    output = run(cmd)

    print(f"\n🎤 {label} devices:" if direction == "capture" else f"\n🔊 {label} devices:")

    if not output.strip():
        print("  (none found)")
        return []

    devices = []
    for line in output.splitlines():
        m = re.search(r"card\s+(\d+):\s+(\w+)\s+\[(.+?)\],\s+device\s+(\d+):\s+(.+?)\s+\[(.+?)\]", line)
        if m:
            card, card_id, card_name, dev, dev_id, dev_name = m.groups()
            hw = f"plughw:{card},{dev}"
            devices.append((hw, card_name, dev_name))
            print(f"  {hw}  {card_name} / {dev_name}")

    return devices


def main():
    print("=" * 60)
    print("AUDIO DEVICES (ALSA)")
    print("=" * 60)

    capture = list_devices("capture")
    playback = list_devices("playback")

    print("\n" + "=" * 60)

    # Suggest USB mic
    usb = [d for d in capture if "usb" in d[1].lower() or "usb" in d[2].lower()]
    if usb:
        hw, card, dev = usb[0]
        print(f"✅ USB mic found: {hw}  ({card} / {dev})")
        print()
        print("   The assistant auto-detects USB mics via 'arecord -l'.")
        print(f"   To force this device: set alsa_device: \"{hw}\" in config/config.yaml")
    else:
        print("⚠  No USB mic found. Check connections and try 'arecord -l'.")
        if capture:
            print(f"   Available: {capture[0][0]}  ({capture[0][1]})")

    print()


if __name__ == "__main__":
    main()
