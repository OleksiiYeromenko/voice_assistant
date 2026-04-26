# Ubiquitous Language

## Interaction lifecycle

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Turn** | One complete user request → assistant spoken response cycle (LISTENING → THINKING → IDLE) | Interaction, exchange, request |
| **Session** | A continuous period of turns bounded by an inactivity timeout; stored as a database record with a summary | Conversation (ambiguous — see below) |
| **Conversation** | The in-memory list of role/content message dicts for the current session, passed to the LLM on each turn | History, context, chat |
| **Utterance** | The raw transcribed text of a single user speech input | Input, query, prompt, message |
| **Wake Word** | A specific spoken phrase (e.g. "Hey Poon") that activates the assistant from IDLE | Hotword, activation phrase |
| **Trigger** | A text keyword or phrase in an utterance that routes the turn to a specific backend (e.g. "use claude") | Wake word (see flagged ambiguities) |

## Pipeline stages

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Transcription** | The process of converting recorded audio into text; output is an utterance | Speech recognition, STT (as a verb) |
| **Route Decision** | The router's output: which backend to use, why, and the utterance with any trigger phrase removed | Routing result |
| **Tool Round** | One iteration of: LLM generates tool calls → tools execute → results returned as messages | Tool loop iteration, function call round |
| **Synthesis** | Converting a text sentence into WAV audio via Piper | Text-to-speech (as a process), rendering |
| **Warm-start** | Pre-loading a model into RAM before the first query so the first turn isn't slow | Preload, model loading |

## Components

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **STT Engine** | Records mic audio via `arecord` and transcribes it with faster-whisper | Speech recognizer, ASR |
| **TTS Engine** | Buffers LLM tokens into sentences, synthesizes each via Piper, and plays via `aplay` | Speech synthesizer, voice engine |
| **Backend** | An LLM inference provider (OllamaBackend, LlamaCppBackend, ClaudeBackend, GeminiBackend) | Model (ambiguous — see below), provider |
| **Router** | Selects the backend for each turn based on triggers, session preference, and remote availability | Dispatcher, model selector |
| **Wake Word Detector** | Listens continuously for the wake word on the mic; yields when detected, then releases the mic | Hotword detector |
| **Memory Store** | Persists facts, preferences, and session summaries across restarts (markdown files + SQLite) | Memory, storage, database |
| **Remote Availability Monitor** | Background thread that polls the remote Ollama server every 60 s and updates the router's default backend key | Health checker, monitor |

## Memory concepts

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Persona** | The assistant's identity, behavioral rules, and tool instructions; loaded from `memory/PERSONA.md` and never changed at runtime | System prompt (which is assembled from multiple sources) |
| **Profile** | User preferences stored as key-value pairs in `memory/PROFILE.md`; written by the `remember` tool | Settings, user config |
| **Fact** | A discrete piece of information about the user, appended with a date to `memory/FACTS.md` | Note, memory item |
| **Preference** | A structured behavioral rule extracted from a fact and stored in the Profile (e.g. "use 24-hour time") | Setting, option |
| **System Prompt** | The full instruction block assembled before each LLM call from persona + profile + known facts | Base prompt, context prompt |
| **Session Summary** | An LLM-generated one-sentence description of a closed session's conversation, written to SQLite | Summary, recap |

## Tool calling

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Tool** | A named capability the LLM can invoke: a JSON schema (for the LLM) plus a Python function (the implementation) | Function, skill |
| **Tool Schema** | The JSON description of a tool's name, purpose, and parameters, sent to the LLM | Tool definition, function spec |
| **Tool Call** | The LLM's parsed request to invoke a specific tool with given arguments | Function call, action |
| **Volatile Tool** | A tool whose results become stale immediately; conversation entries using it are dropped so the LLM never echoes old values | Stateful tool |

## Audio playback

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Speech** | Audio output from the TTS Engine — the assistant's spoken voice | TTS playback, voice output |
| **Stream** | A background audio source (radio station or music) played by the media player, independent of speech | Playback (ambiguous), radio |
| **Radio Station** | An internet radio station found by name, genre, or country and played as a stream | Channel, station |
| **Thinking Sound** | A short audio cue played after transcription while the LLM is generating, to signal the assistant is working | Processing sound, wait sound |
| **Greeting** | A short synthesized phrase played after the wake word fires, to confirm activation | Acknowledgement sound |

## Routing and backends

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Backend Key** | A string identifier for a backend ("remote", "local", "claude", "gemini") | Backend name (ambiguous with `.name` property) |
| **Remote Backend** | The GPU PC's Ollama server on the LAN; preferred default when reachable | GPU backend, network backend |
| **Local Backend** | The LlamaCpp server running on the RPi; always-available fallback | On-device backend, RPi backend |
| **Session Preference** | A sticky backend key set when the user uses an explicit trigger; persists for all subsequent turns until reset | Preferred backend, locked backend |
| **Fallback Chain** | The ordered sequence of backends tried when the primary backend fails (remote → local → none; cloud → default) | Failover list, backup chain |

## FSM states

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **IDLE** | The assistant is waiting for a trigger (wake word, keypress, or text input) | Standby, listening (overloaded) |
| **SESSION_CHECK** | Checks whether the inactivity timeout has elapsed and rotates the session if so | Timeout check |
| **LISTENING** | The mic is open; the STT Engine is recording and will transcribe when silence is detected | Recording, capturing |
| **THINKING** | The router, LLM, tool loop, and TTS Engine are all active for one turn | Processing, generating |
| **SHUTDOWN** | Terminal state; triggers cleanup and process exit | Stopping, closing |

## Input modes

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Wake Word Mode** | Turns are triggered acoustically by detecting the wake word | Voice mode |
| **Keyboard Mode** | Turns are triggered by pressing Enter; audio is still recorded for STT | Interactive mode |
| **Text Mode** | Turns are triggered by typed utterances; no audio involved | CLI mode |

---

## Relationships

- A **Session** contains zero or more **Turns**; each turn appends messages to the **Conversation**.
- A **Session** ends when the inactivity timeout elapses; a **Session Summary** is then generated asynchronously.
- A **Turn** flows through: **LISTENING** (produces an **Utterance**) → **Route Decision** (selects a **Backend**) → up to 3 **Tool Rounds** → **Speech** (synthesized sentence by sentence).
- A **Tool** has exactly one **Tool Schema** (sent to the LLM) and one Python implementation (executed by the executor).
- A **Volatile Tool**'s **Turn** is excluded from the **Conversation** so stale values are never replayed.
- A **Stream** is paused when the **Wake Word** fires and resumed after **Speech** ends.
- The **Memory Store** assembles the **System Prompt** from **Persona** + **Profile** + recent **Facts** before each LLM call.

---

## Example dialogue

> **Dev:** "When the user says 'play jazz', does that start a new **Turn**?"
>
> **Domain expert:** "Yes — the **Wake Word** fires in IDLE, which triggers a **Turn**. The **Utterance** 'play jazz' is transcribed. The **Router** produces a **Route Decision** (no **Trigger** phrase, so it picks the default **Backend**). The LLM responds with a **Tool Call** to `play_radio`, which starts a **Stream**."
>
> **Dev:** "If the user then says 'stop', is that a second **Turn** in the same **Session**?"
>
> **Domain expert:** "Exactly. Both turns belong to the same **Session** as long as the inactivity timeout hasn't elapsed. The second **Turn** calls `stop_radio`, which halts the **Stream**."
>
> **Dev:** "What if the user says 'use Claude to search for that'? Is 'use Claude' a **Trigger** or a **Wake Word**?"
>
> **Domain expert:** "'Use Claude' is a **Trigger** — it's a text keyword in the **Utterance** that routes the **Turn** to the Claude **Backend** and sets a **Session Preference**. A **Wake Word** is only ever an acoustic activation phrase heard by the **Wake Word Detector** before any **Turn** begins. The **Router** strips the trigger phrase from the **Utterance** before passing it to the LLM."
>
> **Dev:** "And if Claude goes down mid-session?"
>
> **Domain expert:** "The **Fallback Chain** kicks in: claude → default (**Remote Backend** or **Local Backend**). The **Session Preference** for Claude is kept; the fallback is silent and transparent to the user."

---

## Flagged ambiguities

- **"wake word" vs "trigger"** — used interchangeably in early code comments but they are distinct concepts. A **Wake Word** is acoustic and activates the assistant from IDLE. A **Trigger** is a text keyword in an utterance that selects a backend. They must not be conflated.

- **"backend" vs "model"** — the code uses both loosely. A **Backend** is an inference provider instance (OllamaBackend, ClaudeBackend, etc.). A **model** is a specific weights file loaded on that backend (e.g. `qwen3:4b`). A single **Backend** could theoretically serve different models; they are not synonyms.

- **"session" vs "conversation"** — both appear in the codebase. A **Session** is a database record with an ID, timestamps, and a **Session Summary**. A **Conversation** is the in-memory list of messages. Sessions live in SQLite; conversations live in RAM and are cleared when a session rotates.

- **"playback"** — used in the codebase to mean both the TTS Engine speaking (**Speech**) and the media player running a **Stream** (radio). These are independent subsystems and should be referred to by their distinct terms to avoid confusion in PRs or config keys.

- **"tool" (schema) vs "tool" (function)** — `WEATHER_TOOL` is a schema dict; `get_weather` is the implementation function. Both are informally called "tool". Prefer **Tool Schema** when referring to the JSON descriptor and **Tool** when referring to the full capability (schema + implementation together).
