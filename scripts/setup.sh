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
uv sync
uv sync --extra cloud --extra tools
echo "  ✓ Python deps installed"

# 3. openWakeWord base models
# These are Git LFS files in the upstream repo and are not downloaded by `uv sync`.
# Without them the wake word detector falls back to keyboard mode.
echo "→ Checking openWakeWord base models..."
OWW_MODELS=$(uv run python -c \
  "import openwakeword, os; print(os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models'))" \
  2>/dev/null)
if [ -n "$OWW_MODELS" ]; then
    mkdir -p "$OWW_MODELS"
    BASE_URL="https://media.githubusercontent.com/media/dscripka/openWakeWord/e49345a/openwakeword/resources/models"
    ALL_PRESENT=true
    for MODEL in melspectrogram.onnx embedding_model.onnx; do
        DEST="$OWW_MODELS/$MODEL"
        # LFS pointer files are tiny (~130 bytes) — re-download if size < 10 KB
        if [ ! -f "$DEST" ] || [ "$(wc -c < "$DEST")" -lt 10240 ]; then
            echo "  Downloading $MODEL ..."
            wget -q -O "$DEST" "$BASE_URL/$MODEL" || { echo "  ✗ Failed — copy manually from a working venv"; ALL_PRESENT=false; }
        fi
    done
    $ALL_PRESENT && echo "  ✓ openWakeWord base models present"
else
    echo "  ⚠ Could not locate openWakeWord package"
fi

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