# Voice Assistant — RPi 5 Home Lab

Local-first voice assistant with smart cloud fallback, running on Raspberry Pi 5.

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
         └───────┘               └──────────┘
                                      │ done
                                      └→ IDLE
```

| State | What happens | Mic owner |
|---|---|---|
| IDLE | Wait for trigger (wake word / Enter / text) | Wake detector (wake mode) |
| SESSION_CHECK | Check inactivity timeout, rotate session | None |
| LISTENING | STT: record + transcribe | STT (arecord) |
| THINKING | Route → LLM + tool loop (max 2 rounds) + streaming TTS | None |
| SHUTDOWN | Close session, exit | None |

## Prerequisites

- **Python 3.13+** — Required by the project
- **`uv` package manager** — Fast, zero-config Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **System packages** (Ubuntu/Debian):
  ```bash
  sudo apt-get install alsa-utils espeak-ng mpv
  ```
- **LLM backends** (at least one required):
  - **Remote**: [Ollama](https://ollama.ai) on a GPU PC at `192.168.1.74:11434`
  - **Local**: [llama.cpp](https://github.com/ggerganov/llama.cpp) server on the RPi at `localhost:8080`
  - **Claude**: Set `ANTHROPIC_API_KEY` in `.env`
  - **Gemini**: Set `GOOGLE_API_KEY` in `.env`
- **Optional**: `TODOIST_API_TOKEN` + `TODOIST_PROJECT_ID` for shopping list; `NOTION_API_KEY` + `NOTION_RECIPES_DB_ID` for recipe lookup

## Installation

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd voice_assistant
   bash scripts/setup.sh
   ```

2. **Restore gitignored files** (not in the repo — copy from a backup or create fresh):

   | File | What it is | How to restore |
   |---|---|---|
   | `.env` | API keys | Copy from backup or create (see below) |
   | `models/*.onnx` | Custom wake word model | Copy from backup |
   | `memory/USER.md` | User facts/prefs | Copy from backup (or let assistant recreate) |

   ```bash
   # Minimum .env to get started
   echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
   echo "GOOGLE_API_KEY=..." >> .env
   ```

3. **Install the systemd service:**
   ```bash
   bash scripts/va install --ui    # UI mode (Raspberry Pi display)
   bash scripts/va install         # headless / wake word only
   sudo systemctl start voice-assistant-ui
   ```

4. **Test components:**
   ```bash
   uv run scripts/test_components.py all
   ```

### Reinstall / Recovery

If the git repo is corrupted (e.g. after a power cut) or you need a clean slate:

```bash
cd ~/Projects

# Back up gitignored files first
cp voice_assistant/.env /tmp/va.env 2>/dev/null || true
cp -r voice_assistant/models /tmp/va-models 2>/dev/null || true
cp voice_assistant/memory/USER.md /tmp/va-user.md 2>/dev/null || true

# Fresh clone
mv voice_assistant voice_assistant.bak  # keep as safety net
git clone <repo-url> voice_assistant
cd voice_assistant

# Restore gitignored files
cp /tmp/va.env .env
cp /tmp/va-models/*.onnx models/ 2>/dev/null || true
cp /tmp/va-user.md memory/ 2>/dev/null || true

# Setup (downloads deps + openwakeword models + Piper voice)
bash scripts/setup.sh

# Reinstall service
bash scripts/va install --ui
sudo systemctl start voice-assistant-ui
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

| Trigger | Backend | Notes |
|---|---|---|
| (default, `preferred_backend` healthy) | Configured preferred (default: remote) | Fastest |
| (default, preferred unreachable) | Local llama.cpp (RPi) | Always available |
| "Use Claude / ask Claude" | Claude | Sticky for session |
| "Use Gemini / ask Gemini" | Gemini | Sticky for session |
| "Use GPU / use remote" | Remote Ollama | Sticky for session |
| "Use local / go offline" | Local llama.cpp | Sticky for session |
| "Go back to automatic" | Auto | Clears session preference |

`BackendHealthMonitor` (`src/health.py`) polls all backends continuously (remote every 60 s, cloud every 5 min) and is the single source of truth for both the router and the UI. All backends are registered at startup regardless of connectivity.

**Failure handling:** If you explicitly request a backend ("use Claude") and it fails, the assistant voices the error and resets to automatic routing. Silent fallback to `local` applies only to automatic/session-preference routes.

## Tools

| Tool | Provider |
|---|---|
| Weather (current + forecast) | Open-Meteo (free, no key) |
| Web search | DuckDuckGo (free, no key) |
| Time / date (any timezone) | System + geocoding |
| Shopping list (add + view) | Todoist API |
| Timers (set + cancel) | Background thread |
| Internet radio (play + stop + volume) | Radio Browser API + mpv |
| Recipe lookup | Notion family recipe DB |
| Remember / recall | Markdown files + SQLite |

## Project Structure

```
voice-assistant/
├── config/config.yaml         # All configuration
├── memory/
│   ├── PERSONA.md             # Assistant identity + tool instructions (edit freely)
│   └── USER.md                # User preferences + facts (written by remember tool)
├── src/
│   ├── main.py                # Entry point + run_llm_with_tools
│   ├── state_machine.py       # FSM: states, transitions, orchestration
│   ├── config.py              # Config loader (YAML + VA_ env overrides)
│   ├── health.py              # BackendHealthMonitor — unified polling for router + UI
│   ├── audio.py               # ALSA device detection + arecord streaming
│   ├── monitor.py             # CPU/RAM/temp tracking + latency records
│   ├── wake_word/detector.py  # openWakeWord integration
│   ├── stt/engine.py          # faster-whisper STT
│   ├── llm/backends.py        # OllamaBackend, LlamaCppBackend, ClaudeBackend, GeminiBackend
│   ├── router/router.py       # Model selection logic
│   ├── tools/
│   │   ├── executor.py        # Tool schemas + dispatch
│   │   ├── player.py          # Internet radio via mpv
│   │   ├── recipes.py         # Notion recipe library
│   │   └── timers.py          # Countdown timers
│   ├── tts/engine.py          # Piper streaming TTS
│   └── ui/                    # PyQt6 display (--ui flag)
├── scripts/
│   ├── setup.sh               # One-time setup
│   └── test_components.py     # Component tests
├── voices/                    # Piper voice models (.onnx)
├── models/                    # Wake word models (.onnx)
└── pyproject.toml
```


## Configuration

Edit `config/config.yaml` or override with environment variables:

```bash
VA_LLM_LOCAL_MODEL=qwen3:1.7b uv run python -m src.main  # Use smaller model
VA_STT_MODEL=tiny.en uv run python -m src.main            # Faster STT
VA_LLM_PREFERRED_BACKEND=local uv run python -m src.main  # Default to local backend
```

Key config values:
- `llm.preferred_backend`: Default backend when no session preference is set (`"remote"` or `"local"`, default `"remote"`). The health monitor automatically falls back to `"local"` if the preferred backend is unreachable.

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