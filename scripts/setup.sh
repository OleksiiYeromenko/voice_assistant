#!/usr/bin/env bash
# Voice Assistant — RPi 5 one-time setup.
#
# This script handles system-level setup that pyproject.toml / uv cannot:
#   1. System packages via apt (espeak-ng, alsa-utils, mpv)
#   2. Python deps via uv sync
#   3. openWakeWord base models (LFS files not fetched by uv — downloaded from GitHub)
#   4. (reserved)
#   5. Ollama model download (qwen3:4b, ~2.5 GB)
#   6. Piper voice model download from HuggingFace (~65 MB)
#   7. Runtime directory creation (data/, tts_output/, stt_output/, memory/)
#   8. Audio device detection and mic index suggestion
#
# Safe to run multiple times (idempotent).
# Usage: bash scripts/setup.sh

set -euo pipefail

echo "=== Voice Assistant Setup ==="
echo ""

# 1. System dependencies
echo "→ Checking system packages..."
NEEDED=""
dpkg -s espeak-ng &>/dev/null || NEEDED="$NEEDED espeak-ng"
dpkg -s alsa-utils &>/dev/null || NEEDED="$NEEDED alsa-utils"
dpkg -s mpv &>/dev/null || NEEDED="$NEEDED mpv"
# portaudio19-dev no longer needed — we use arecord (ALSA) directly

if [ -n "$NEEDED" ]; then
    echo "  Installing:$NEEDED"
    sudo apt update -qq
    sudo apt install -y -qq $NEEDED
else
    echo "  ✓ All system packages present"
fi

# 2. Python dependencies
echo "→ Installing Python dependencies..."
uv sync --extra cloud --extra tools --extra ui
echo "  ✓ Python deps installed"

# 3. openWakeWord base models
# The git-based install skips LFS files so melspectrogram.onnx / embedding_model.onnx
# are missing after `uv sync`. Fix: trigger openwakeword's own auto-download by
# instantiating Model(), which writes the binaries into the package resources dir.
echo "→ Checking openWakeWord base models..."
uv run python - << 'PYEOF'
import sys
from pathlib import Path

try:
    import openwakeword
    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Remove any LFS pointer files (valid ONNX files are > 1 MB)
    for p in models_dir.glob("*.onnx"):
        if p.stat().st_size < 10_240:
            p.unlink()

    missing = [f for f in ("melspectrogram.onnx", "embedding_model.onnx")
               if not (models_dir / f).exists()]

    if missing:
        print(f"  Downloading base models via openwakeword.Model()...")
        openwakeword.Model(wakeword_models=[])  # triggers auto-download
        still_missing = [f for f in missing if not (models_dir / f).exists()]
        if still_missing:
            print(f"  ✗ Still missing: {still_missing}")
            print("    Copy manually: cp /path/to/backup/.venv/.../openwakeword/resources/models/*.onnx \\")
            print(f"      {models_dir}/")
            sys.exit(1)
    print("  ✓ openWakeWord base models present")
except Exception as e:
    print(f"  ✗ {e}")
    sys.exit(1)
PYEOF

# 5. Ollama model
echo "→ Checking Ollama model..."
if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | grep -q "qwen3"; then
        echo "  ✓ Qwen3 model already pulled"
    else
        echo "  Pulling qwen3:4b (this takes a few minutes)..."
        ollama pull qwen3:4b
        echo "  ✓ qwen3:4b ready"
    fi
else
    echo "  ⚠ Ollama not installed — install from https://ollama.com"
fi

# 6. Piper voice
echo "→ Checking Piper voice..."
VOICE_DIR="./voices"
VOICE_FILE="${VOICE_DIR}/en_US-hfc_male-medium.onnx"
if [ -f "$VOICE_FILE" ]; then
    echo "  ✓ Voice already exists"
else
    mkdir -p "$VOICE_DIR"
    echo "  Downloading voice model..."
    wget -q --show-progress -P "$VOICE_DIR" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx"
    wget -q --show-progress -P "$VOICE_DIR" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json"
    echo "  ✓ Voice downloaded"
fi

# 7. Directories
mkdir -p data tts_output stt_output memory

# 8. Detect audio devices
echo ""
echo "→ Detecting audio devices..."
echo "  Playback:"
aplay -l 2>/dev/null | grep "^card" | while read line; do echo "    $line"; done
echo "  Capture:"
arecord -l 2>/dev/null | grep "^card" | while read line; do echo "    $line"; done

# 9. List audio devices (ALSA)
echo ""
echo "→ Listing audio devices..."
uv run scripts/list_audio_devices.py 2>/dev/null || echo "  (run 'uv run scripts/list_audio_devices.py' manually)"

# 10. Summary
echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Restore gitignored files (copy from backup if reinstalling):"
echo "     cp /path/to/backup/.env .env                   # API keys"
echo "     cp /path/to/backup/models/*.onnx models/       # wake word model"
echo "     cp /path/to/backup/memory/USER.md memory/      # user memory (optional)"
echo ""
echo "  2. Set ALSA device (if needed):"
echo "     Edit config/config.yaml → stt.alsa_device"
echo "     (null = auto-detect USB mic via arecord -l)"
echo ""
echo "  3. Install systemd service:"
echo "     bash scripts/va install --ui    # UI mode (PyQt6 display)"
echo "     bash scripts/va install         # headless / wake word only"
echo "     sudo systemctl start voice-assistant-ui"
echo ""
echo "  4. Set cloud API keys (if not using .env from backup):"
echo "     echo ANTHROPIC_API_KEY=sk-ant-... >> .env"
echo "     echo GOOGLE_API_KEY=...           >> .env"
echo ""
echo "  5. Test components:"
echo "     uv run scripts/test_components.py stt"
echo "     uv run scripts/test_components.py tts"
echo "     uv run scripts/test_components.py llm"
echo ""
echo "  6. Run the assistant:"
echo "     uv run python -m src.main --no-wake   # keyboard mode"
echo "     uv run python -m src.main --ui        # with display"
echo ""