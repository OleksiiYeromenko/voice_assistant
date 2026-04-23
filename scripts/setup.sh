#!/usr/bin/env bash
# Voice Assistant — RPi 5 one-time setup.
#
# This script handles system-level setup that pyproject.toml / uv cannot:
#   1. System packages via apt (espeak-ng, alsa-utils)
#   2. Python deps via uv sync (convenience wrapper)
#   3. Ollama model download (qwen3:4b, ~2.5 GB)
#   4. Piper voice model download from HuggingFace (~65 MB)
#   5. Runtime directory creation (data/, tts_output/, stt_output/)
#   6. Audio device detection and mic index suggestion
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

# 3. Ollama model
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

# 4. Piper voice
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

# 5. Directories
mkdir -p data tts_output stt_output

# 6. Detect audio devices
echo ""
echo "→ Detecting audio devices..."
echo "  Playback:"
aplay -l 2>/dev/null | grep "^card" | while read line; do echo "    $line"; done
echo "  Capture:"
arecord -l 2>/dev/null | grep "^card" | while read line; do echo "    $line"; done

# 7. List audio devices (ALSA)
echo ""
echo "→ Listing audio devices..."
uv run scripts/list_audio_devices.py 2>/dev/null || echo "  (run 'uv run scripts/list_audio_devices.py' manually)"

# 8. Summary
echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Set ALSA device (if needed):"
echo "     Edit config/config.yaml → stt.alsa_device"
echo "     (null = auto-detect USB mic via arecord -l)"
echo ""
echo "  2. Set cloud API keys (optional):"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
echo "     export GOOGLE_API_KEY='...'"
echo ""
echo "  3. Test components:"
echo "     uv run scripts/test_components.py router"
echo "     uv run scripts/test_components.py weather"
echo "     uv run scripts/test_components.py tts"
echo "     uv run scripts/test_components.py stt"
echo "     uv run scripts/test_components.py llm"
echo ""
echo "  4. Run the assistant:"
echo "     uv run python -m src.main --no-wake"
echo ""