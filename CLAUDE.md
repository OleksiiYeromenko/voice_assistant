# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local-first voice assistant for Raspberry Pi 5 (16GB). Processes audio through a pipeline: wake word → STT → LLM → tool calling → streaming TTS. Supports local inference (Ollama) with optional cloud fallback (Claude, Gemini).

**Stack:** Python 3.13, `uv` package manager, `hatchling` build backend.

## Commands

```bash
# Run (wake word enabled)
uv run python -m src.main

# Run (keyboard trigger, no wake word)
uv run python -m src.main --no-wake

# Run (text input only)
uv run python -m src.main --text

# Component tests (each is independent)
uv run scripts/test_components.py pipeline     # Full text→LLM→tools, no audio
uv run scripts/test_components.py stt          # Mic + transcription
uv run scripts/test_components.py tts          # Text-to-speech
uv run scripts/test_components.py llm          # Ollama streaming
uv run scripts/test_components.py llm-tools    # Ollama with tool calling
uv run scripts/test_components.py gemini       # Gemini streaming
uv run scripts/test_components.py weather      # Weather tool
uv run scripts/test_components.py search       # Web search tool
uv run scripts/test_components.py router       # Model routing logic
uv run scripts/test_components.py all          # Run all tests

# Other test scripts
uv run scripts/wake_word_test.py               # Wake word detector standalone
uv run scripts/test_stt_pipeline.py            # Full STT recording + transcription
uv run scripts/list_audio_devices.py           # Enumerate ALSA audio devices

# Lint / format
uv run ruff check src/
uv run ruff format src/

# One-time setup
bash scripts/setup.sh
```

## Architecture

### State Machine (`src/state_machine.py`)

The assistant is orchestrated by a FSM with 6 states:

```
IDLE → SESSION_CHECK → LISTENING → THINKING → (INTERRUPTED) → IDLE
```

- **IDLE**: Wait for trigger (wake word, keypress, or text)
- **SESSION_CHECK**: Check inactivity timeout; rotate sessions with background summarization
- **LISTENING**: Record mic via `arecord`, transcribe with faster-whisper
- **THINKING**: Route to LLM, tool calling loop (max 3 rounds), stream TTS response
- **INTERRUPTED**: Wake word detected during TTS playback → cancel and re-listen
- **SHUTDOWN**: Clean exit

`src/main.py` contains the LLM streaming loop and TTS orchestration called from THINKING state.

### LLM Backends (`src/llm/backends.py`)

Three backends selected at runtime by `src/router/router.py`:

- **OllamaBackend** (default): Tries remote GPU at `192.168.1.74:11434`, falls back to `localhost:11434`
- **ClaudeBackend**: Requires `ANTHROPIC_API_KEY`
- **GeminiBackend**: Requires `GOOGLE_API_KEY`

Router priority: explicit user trigger ("use claude") → session preference → automatic (remote if reachable) → fallback chain.

### Streaming TTS (`src/tts/engine.py`)

Buffers LLM tokens until a sentence boundary, then synthesizes via Piper and pipes directly to `aplay` (no disk writes). This allows audio playback to start before the LLM finishes generating.

### Tool Calling (`src/tools/executor.py`)

Available tools: `get_weather`, `web_search`, `get_time`, `add_to_shopping_list`, `remember`, `recall`. The loop in `src/main.py` retries up to 3 rounds before returning the final response.

### Memory (`src/memory.py`)

- **Markdown files** (git-tracked, human-editable): `memory/PERSONA.md` (system prompt), `memory/PROFILE.md` (user prefs), `memory/FACTS.md` (appended facts)
- **SQLite** (`data/memory.db`): Session summaries only; auto-rotated on inactivity timeout with background LLM summarization

## Configuration

Primary config: `config/config.yaml`. All keys can be overridden with `VA_`-prefixed environment variables (e.g., `VA_STT_MODEL=tiny.en`, `VA_LLM_LOCAL_MODEL=qwen3:1.7b`, `VA_WAKE_WORD_THRESHOLD=0.6`).

API keys go in `.env` (gitignored): `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

Key config values:
- STT model: `base.en` (options: `tiny.en`, `small.en`)
- LLM model: `qwen3:4b-instruct`, `num_predict: 384` (keep low for RPi latency)
- TTS voice: `./voices/en_US-hfc_male-medium.onnx`
- Wake word: `./models/hey_Poon-dyk.onnx`, threshold `0.5`
- Audio output: `aplay_device: "plughw:2,0"` (USB audio)

## Hardware Notes

Target: RPi 5 (16GB). The `num_thread: 4` Ollama setting is tuned for RPi 5's CPU. The remote Ollama server (`192.168.1.74:11434`) is a GPU PC on the LAN — if unreachable, the assistant falls back to the local Ollama instance automatically. `src/monitor.py` tracks CPU/RAM/temperature.
