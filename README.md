# Voice Assistant — RPi 5 Home Lab

Local-first voice assistant with smart cloud fallback, running on Raspberry Pi 5 (16GB).

## Architecture

```
Mic → Wake Word (openWakeWord) → STT (faster-whisper)
   → Router → LLM (Ollama local / Claude / Gemini)
   → Tool Executor → Streaming TTS (Piper) → Speaker
```

### State Machine

The assistant runs as a finite state machine with explicit states and transitions:

```
                    ┌──────────┐
          Ctrl+C    │ SHUTDOWN │
         ┌─────────→│          │
         │          └──────────┘
         │
    ┌────┴───┐   wake / key / text   ┌───────────────┐
    │  IDLE  │──────────────────────→│ SESSION_CHECK  │
    └────────┘                       └───┬───────┬────┘
         ↑                               │       │
         │                          text input  otherwise
         │                               │       │
         │                               ↓       ↓
         │                       ┌──────────┐  ┌───────────┐
         │       no speech       │ THINKING │←─│ LISTENING  │
         │       ┌───────────────│          │  └───────────┘
         │       │               └──┬────┬──┘      ↑
         │       │          done    │    │ wake     │
         └───────┘       ┌─────────┘    │ word     │
                         │              ↓          │
                         │       ┌─────────────┐   │
                         │       │ INTERRUPTED  │──┘
                         │       └─────────────┘
                         └→ IDLE
```

| State | What happens | Mic owner |
|---|---|---|
| IDLE | Wait for trigger (wake word / Enter / text) | Wake detector (wake mode) |
| SESSION_CHECK | Check inactivity timeout, rotate session | None |
| LISTENING | STT: record + transcribe | STT (arecord) |
| THINKING | Route → LLM + tool loop + streaming TTS | Interrupt listener (wake mode) |
| INTERRUPTED | TTS stopped, clean up | Releasing → free |
| SHUTDOWN | Close session, exit | None |

## Prerequisites

- **Python 3.13+** — Required by the project
- **`uv` package manager** — Fast, zero-config Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **System packages** (Ubuntu/Debian):
  ```bash
  sudo apt-get install alsa-utils espeak-ng mpv
  ```
- **Ollama** (local inference) or API keys for cloud models:
  - **Local**: [Ollama](https://ollama.ai) running on localhost:11434 or remote GPU PC
  - **Claude**: Set `ANTHROPIC_API_KEY` environment variable
  - **Gemini**: Set `GOOGLE_API_KEY` environment variable
  - **Todoist** (optional): Set `TODOIST_API_TOKEN` for shopping list tool

## Installation

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd voice_assistant
   bash scripts/setup.sh
   ```

2. **Configure:**
   - Edit `config/config.yaml` for your audio devices and model preferences
   - Set API keys in `.env` (git-ignored):
     ```bash
     echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
     echo "GOOGLE_API_KEY=..." >> .env
     ```

3. **Test components:**
   ```bash
   uv run scripts/test_components.py all
   ```

## Quick Start

```bash
# Full mode with wake word
uv run python -m src.main

# Keyboard mode (press Enter to speak)
uv run python -m src.main --no-wake

# Text-only mode (for testing)
uv run python -m src.main --text

# With UI (PyQt6)
uv run python -m src.main --ui
```

## Development Setup

To contribute or modify the code:

1. **Install development dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```

2. **Run linter:**
   ```bash
   uv run ruff check src/
   uv run ruff format src/
   ```

3. **Run tests:**
   ```bash
   # Component tests
   uv run scripts/test_components.py all
   
   # Specific component
   uv run scripts/test_components.py weather
   uv run scripts/test_components.py llm
   ```

4. **Debug with environment variables:**
   ```bash
   VA_LLM_LOCAL_MODEL=qwen3:1.7b uv run python -m src.main --no-wake
   VA_STT_MODEL=tiny.en uv run python -m src.main --text
   VA_WAKE_WORD_THRESHOLD=0.6 uv run python -m src.main
   ```

5. **View logs:**
   ```bash
   tail -f data/logs/assistant.log
   ```

## Model Routing

The assistant routes to the right model automatically:

| Trigger | Model | Example |
|---|---|---|
| (default) | Local (qwen3:4b) | "What time is it?" |
| "Use Claude..." | Claude Sonnet | "Use Claude to analyze this code" |
| "Ask Gemini..." | Gemini Flash | "Ask Gemini about quantum physics" |
| "Switch to Claude" | Claude (sticky) | Sets session preference |
| "Go back to automatic" | Local | Resets to auto-routing |

Cloud → local fallback happens automatically on network failure.

## Tools

| Tool | Status | Provider |
|---|---|---|
| Weather | ✅ Working | Open-Meteo (free) |
| Web Search | ✅ Working | DuckDuckGo |
| Shopping List | ✅ Working | Todoist API |
| Google Keep | 🔲 Planned | gkeepapi |
| Smart Home | 🔲 Planned | n8n webhooks |

## Project Structure

```
voice-assistant/
├── config/config.yaml       # All configuration
├── src/
│   ├── main.py              # Entry point + LLM streaming functions
│   ├── state_machine.py     # FSM: states, transitions, orchestration
│   ├── config.py            # Config loader
│   ├── monitor.py           # CPU/RAM/temp tracking
│   ├── wake_word/detector.py
│   ├── stt/engine.py        # faster-whisper
│   ├── llm/backends.py      # Ollama, Claude, Gemini
│   ├── router/router.py     # Model selection
│   ├── tools/executor.py    # Tool definitions + execution
│   └── tts/engine.py        # Piper streaming TTS
├── scripts/
│   ├── setup.sh             # One-time setup
│   └── test_components.py   # Component tests
├── voices/                  # Piper voice models
├── data/                    # Shopping list, etc.
└── pyproject.toml
```

## Phases

### Phase 1 ✅ — Core Loop
Wake word → STT → Local LLM → Streaming TTS. Basic tool calling (weather, search, shopping list).

### Phase 2 — Cloud + Polish
- Claude/Gemini backends with fallback
- Google Keep integration
- Conversation persistence (SQLite)
- Resource usage dashboard

### Phase 3 — Display + Advanced
- PyQt6 screen UI with model badges
- Streaming text display
- n8n webhook integration
- Custom wake word training
- Benchmark test suite for model comparison

## Configuration

Edit `config/config.yaml` or override with environment variables:

```bash
VA_LLM_LOCAL_MODEL=qwen3:1.7b uv run python -m src.main  # Use smaller model
VA_STT_MODEL=tiny.en uv run python -m src.main            # Faster STT
```

## Resource Budget (RPi 5, 16GB)

| Component | RAM | CPU | Notes |
|---|---|---|---|
| OS + services | ~1 GB | — | |
| Ollama (qwen3:4b) | ~3-4 GB | 100% during inference | |
| faster-whisper (base.en) | ~0.5 GB | Burst during transcription | |
| openWakeWord | ~50 MB | ~5% continuous | |
| Piper TTS | ~100 MB | Burst during synthesis | |
| n8n | ~0.5 GB | Idle mostly | |
| **Headroom** | **~10 GB** | | Room for upgrades |