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

## Quick Start

```bash
# One-time setup
bash scripts/setup.sh

# Set cloud API keys (optional)
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Test components individually
uv run scripts/test_components.py weather
uv run scripts/test_components.py llm
uv run scripts/test_components.py all

# Run the assistant
uv run python -m src.main --no-wake    # Keyboard mode (no wake word)
uv run python -m src.main              # Full mode with wake word
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