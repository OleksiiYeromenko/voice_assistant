"""Voice Assistant — entry point.

Initializes components and hands control to the FSM (src/state_machine.py).
Pure functions (build_backends, run_llm_with_tools, etc.) stay here as they're
called by the FSM state handlers.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src import runtime_state
from src.config import load_config
from src.health import BackendHealthMonitor
from src.llm.backends import (
    ClaudeBackend,
    GeminiBackend,
    LlamaCppBackend,
    OllamaBackend,
    ToolCall,
    check_ollama_connectivity,
    get_running_remote_model,
)
from src.memory import MarkdownMemoryStore
from src.monitor import LatencyRecord, snapshot
from src.router.router import ModelRouter
from src.stt.engine import STTEngine
from src.tools.executor import (
    PERSISTENT_WRITE_TOOLS,
    READ_ONLY_TOOLS,
    UNTRUSTED_TOOLS,
    execute_tool,
    register_tool,
)
from src.tools.player import init_player
from src.tools.timers import register_alert_callback
from src.tools.vacuum import init_vacuum
from src.tts.engine import TTSEngine

log = logging.getLogger(__name__)

# Max tool-call rounds to prevent infinite loops
MAX_TOOL_ROUNDS = 2

# Lookup tools whose result is meant to feed another tool call (e.g. recall the
# favourite station, then play_radio it). Tools stay offered on the next round
# after these; after any other tool they are withheld (see run_llm_with_tools).
CHAINING_TOOLS: frozenset[str] = frozenset({"recall"})

# Tools that don't need a second LLM call — the tool result is spoken directly.
# Query tools (get_weather, web_search, recall, etc.) still go through LLM synthesis.
ACTION_TOOLS: frozenset[str] = frozenset({
    "set_volume",
    "stop_radio",
    "set_timer",
    "cancel_timer",
    "add_to_shopping_list",
    "remember",
    "start_vacuum",
})


@dataclass
class TurnProgress:
    """What a turn has already done, so a failed turn is only retried when that is safe.

    Re-running a turn on a fallback backend repeats everything: the user would hear the
    reply twice and a tool like add_to_shopping_list could run twice.
    """

    spoke: bool = False  # some of the reply was already spoken
    tools: set[str] = field(default_factory=set)  # tools started (even if they then failed)

    def safe_to_retry(self) -> bool:
        return not self.spoke and self.tools <= READ_ONLY_TOOLS


_LOCAL_SUMMARY_MAX_WAIT_S = 600  # a local summary waits this long for idle (runtime_state)


def strip_untrusted_tool_messages(delta: list[dict]) -> list[dict]:
    """Remove raw output of UNTRUSTED_TOOLS (and the calls that made it) from a turn's
    messages before they are kept as conversation history.

    The model's own final answer stays, so follow-up questions still have context, but
    web text can no longer sit in history and influence later turns (e.g. get turned
    into a saved "preference" by a later remember call).
    """
    out: list[dict] = []
    for msg in delta:
        if msg.get("role") == "tool" and msg.get("tool_name") in UNTRUSTED_TOOLS:
            continue
        calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and calls:
            kept = [c for c in calls if c.get("function", {}).get("name") not in UNTRUSTED_TOOLS]
            if len(kept) != len(calls):
                if not kept and not msg.get("content"):
                    continue
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                if kept:
                    msg["tool_calls"] = kept
        out.append(msg)
    return out


def model_info_line(backend, backend_key: str) -> str:
    """The <current_model> text put in the system prompt. Single source of truth: the
    startup cache priming must build the exact same prompt the state machine sends."""
    return f"You are running as: {backend.name} (backend: {backend_key})"


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------
def build_backends(cfg: dict) -> dict:
    """Create all LLM backend objects from config.

    Remote backend is always registered regardless of current connectivity —
    health monitoring handles availability tracking at runtime.

    Remote model selection: use whatever is running in Ollama RAM if reachable
    at startup, otherwise fall back to the configured remote_model.
    """
    backends: dict = {}
    llm_cfg = cfg["llm"]
    remote_url = llm_cfg.get("remote_base_url", "http://192.168.1.74:11434")
    local_url = llm_cfg.get("local_base_url", "http://localhost:8080")
    model_cfg = llm_cfg["local"]
    preferred_remote_model = llm_cfg.get("remote_model", model_cfg["model"])

    # Local backend — always available, always registered
    backends["local"] = LlamaCppBackend(
        base_url=local_url,
        model=model_cfg.get("model", ""),
        temperature=model_cfg["temperature"],
        num_predict=model_cfg.get("num_predict"),
        label="local",
    )

    # Remote backend — always registered; use running model if reachable, else configured model
    if check_ollama_connectivity(remote_url):
        running = get_running_remote_model(remote_url)
        remote_model = running or preferred_remote_model
        log.info(
            f"GPU PC Ollama reachable — using model '{remote_model}'"
            + (" (already in RAM)" if running else " (will activate)")
        )
    else:
        remote_model = preferred_remote_model
        log.warning(
            f"GPU PC Ollama not reachable at startup — registered with model '{remote_model}'"
        )

    backends["remote"] = OllamaBackend(
        model=remote_model,
        base_url=remote_url,
        temperature=model_cfg["temperature"],
        num_ctx=model_cfg["num_ctx"],
        num_predict=model_cfg.get("num_predict"),
        num_thread=model_cfg.get("num_thread"),
        think=model_cfg.get("think", False),
        read_timeout_s=float(llm_cfg.get("remote_read_timeout_s", 60)),
        label="remote",
    )

    # Cloud backends — optional, fail gracefully if API keys missing
    try:
        cloud_cfg = llm_cfg["cloud"]["claude"]
        backends["claude"] = ClaudeBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Claude backend registered")
    except Exception as e:
        log.warning(f"Claude backend not available: {e}")

    try:
        cloud_cfg = llm_cfg["cloud"]["gemini"]
        backends["gemini"] = GeminiBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Gemini backend registered")
    except Exception as e:
        log.warning(f"Gemini backend not available: {e}")

    return backends


# ---------------------------------------------------------------------------
# LLM streaming with tool calling
# ---------------------------------------------------------------------------
def _create_token_gen(
    backend,
    messages: list[dict],
    tools: list[dict] | None,
    system: str,
    latency: LatencyRecord,
    tool_calls_list: list | None = None,
    ui_bus=None,
    is_first_round: bool = True,
):
    """Factory for token generators with shared latency/UI tracking.

    Returns a generator that yields text tokens and collects tool_calls
    (if tool_calls_list is provided). Updates latency record with timing info.
    """

    def token_gen():
        first_token_seen = False
        _generation_ms = 0.0
        stream = iter(backend.stream(messages, tools=tools, system=system))
        while True:
            _t0 = time.perf_counter()
            try:
                chunk = next(stream)
            except StopIteration:
                break
            _generation_ms += (time.perf_counter() - _t0) * 1000

            if chunk.text:
                if not first_token_seen:
                    first_token_seen = True
                    if is_first_round:
                        latency.llm_first_token_ms = _generation_ms
                latency.token_count += 1
                if ui_bus is not None:
                    ui_bus.text_chunk.emit(chunk.text)
                yield chunk.text
            if chunk.thinking:
                log.debug(f"[think] {chunk.thinking}")
            if tool_calls_list is not None:
                tool_calls_list.extend(chunk.tool_calls)

        latency.llm_ms += _generation_ms
        latency.model_used = backend.name

    return token_gen


def run_llm_with_tools(
    backend,
    messages: list[dict],
    tools: list[dict],
    system: str,
    tts: TTSEngine,
    latency: LatencyRecord,
    thinking_proc=None,
    ui_bus=None,
    progress: TurnProgress | None = None,
) -> tuple[str, set[str], list[dict]]:
    """Run LLM with tool calling and per-sentence streaming TTS.

    Returns (response_text, tools_used, delta_messages) where delta_messages are
    the messages appended during this call (tool calls, results, final response).
    Callers extend conversation history with delta_messages so the model sees the
    full tool-call pattern on the next turn — preventing the 2B model from copying
    a canned action-tool response instead of calling the tool.
    ui_bus is optional; when provided, streaming tokens and tool events are emitted.
    """
    progress = progress if progress is not None else TurnProgress()
    full_response = ""
    tools_used: set[str] = set()
    last_round_tools: set[str] = set()
    initial_len = len(messages)

    for round_num in range(MAX_TOOL_ROUNDS):
        tool_calls: list[ToolCall] = []
        text_buffer = ""
        printed_prefix = False

        # After tool calls have been made, don't send tools again — the model
        # only needs to produce a spoken confirmation, and omitting tools prevents
        # it returning a degenerate empty response instead of text. Exception: after
        # a lookup tool (CHAINING_TOOLS) the model may need one more call to act on
        # the result. If that uses up the last round, the forced text-only call
        # below still produces the spoken reply.
        stream_tools = tools if last_round_tools <= CHAINING_TOOLS else None

        token_gen = _create_token_gen(
            backend,
            messages,
            stream_tools,
            system,
            latency,
            tool_calls_list=tool_calls,
            ui_bus=ui_bus,
            is_first_round=(round_num == 0),
        )

        # Stream tokens through TTS — speaks complete sentences as they arrive.
        # pre_proc (thinking sound) is passed only on the first round; cleared inside stream_speak.
        speaking_emitted = False
        pre_proc = thinking_proc if round_num == 0 else None
        for text in tts.stream_speak(token_gen(), latency=latency, pre_proc=pre_proc):
            if not speaking_emitted and text.strip() and ui_bus is not None:
                ui_bus.state_changed.emit("SPEAKING")
                speaking_emitted = True
            if text.strip():
                progress.spoke = True
            if not printed_prefix:
                print("\n🤖 ", end="", flush=True)
                printed_prefix = True
            text_buffer += text
            print(text, end="", flush=True)

        if text_buffer.strip():
            print()  # newline after streaming
            if ui_bus is not None:
                ui_bus.response_complete.emit()

        # If tool calls were issued, execute them and loop
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": text_buffer or "",
                    "tool_calls": [
                        {
                            "function": {"name": tc.name, "arguments": tc.arguments},
                            **(
                                {"thought_signature": tc.thought_signature}
                                if tc.thought_signature
                                else {}
                            ),
                        }
                        for tc in tool_calls
                    ],
                }
            )

            last_round_tools = {tc.name for tc in tool_calls}
            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                tools_used.add(tc.name)
                progress.tools.add(tc.name)
                if ui_bus is not None:
                    ui_bus.tool_started.emit(tc.name)
                if tc.name in PERSISTENT_WRITE_TOOLS and (
                    progress.tools & UNTRUSTED_TOOLS
                    or any(c.name in UNTRUSTED_TOOLS for c in tool_calls)
                ):
                    # Text from a web search must not be able to write to long-term memory.
                    log.warning(f"Refused {tc.name}: web content is in play this turn")
                    result = (
                        "ERROR: Cannot save to memory in the same request as a web search. "
                        "Tell the user to ask you to remember it separately."
                    )
                else:
                    result = execute_tool(tc.name, tc.arguments, backend=backend)
                if ui_bus is not None:
                    ui_bus.tool_done.emit(tc.name, str(result)[:80])
                if tc.name == "get_recipe" and ui_bus is not None:
                    from src.tools.recipes import parse_recipe_result
                    recipe_data = parse_recipe_result(result)
                    if recipe_data:
                        ui_bus.recipe_ready.emit(recipe_data)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tc.name,
                        "content": result,
                    }
                )

            # Action-only tools: speak the tool result directly, skip LLM round 2.
            # Falls through to LLM if any result is an error string.
            if all(tc.name in ACTION_TOOLS for tc in tool_calls):
                results = [m["content"] for m in messages if m["role"] == "tool"][-len(tool_calls):]
                if not any(r.startswith("ERROR:") for r in results):
                    canned = " ".join(results)

                    def _canned_gen():
                        yield canned

                    printed_prefix = False
                    for text in tts.stream_speak(_canned_gen(), latency=latency):
                        if not printed_prefix:
                            print("\n🤖 ", end="", flush=True)
                            printed_prefix = True
                        print(text, end="", flush=True)
                    if printed_prefix:
                        print()
                    if ui_bus is not None:
                        ui_bus.state_changed.emit("SPEAKING")
                        ui_bus.response_complete.emit()
                    messages.append({"role": "assistant", "content": canned})
                    return canned, tools_used, messages[initial_len:]

            continue

        # No tool calls — this is the final text response
        full_response = text_buffer.strip()
        break

    # Model exhausted tool rounds without producing a final answer.
    # Force one text-only call so the user always hears a response.
    if not full_response and tools_used:
        log.warning(
            f"LLM used {MAX_TOOL_ROUNDS} tool rounds without producing text; "
            "forcing final response without tools."
        )
        text_buffer = ""
        printed_prefix = False

        final_token_gen = _create_token_gen(
            backend,
            messages,
            None,  # no tools on fallback
            system,
            latency,
            tool_calls_list=None,
            ui_bus=ui_bus,
            is_first_round=False,  # not first round anymore
        )

        speaking_emitted = False
        for text in tts.stream_speak(final_token_gen(), latency=latency):
            if not speaking_emitted and text.strip() and ui_bus is not None:
                ui_bus.state_changed.emit("SPEAKING")
                speaking_emitted = True
            if not printed_prefix:
                print("\n🤖 ", end="", flush=True)
                printed_prefix = True
            text_buffer += text
            print(text, end="", flush=True)

        if text_buffer.strip():
            print()
            if ui_bus is not None:
                ui_bus.response_complete.emit()
        full_response = text_buffer.strip()

    if full_response:
        log.info(f"Response: {full_response[:500]}")
        messages.append({"role": "assistant", "content": full_response})
    return full_response, tools_used, messages[initial_len:]



# ---------------------------------------------------------------------------
# Local llama.cpp prompt-cache priming
# ---------------------------------------------------------------------------
class _LocalCachePrimer:
    """Re-fills the local llama.cpp KV cache with the real system prompt + tools.

    llama-server runs one slot (-np 1), so any other request (e.g. a session summary)
    evicts the cached prefix, and a cold prefill takes about a minute on the Pi. Priming
    runs on a background thread; overlapping requests are skipped.
    """

    def __init__(self, backend, memory):
        self._backend = backend
        self._memory = memory
        self._busy = threading.Lock()

    def prime_async(self, reason: str) -> None:
        threading.Thread(
            target=self._run, args=(reason,), daemon=True, name="prime-local-cache"
        ).start()

    def _run(self, reason: str) -> None:
        if not self._busy.acquire(blocking=False):
            return  # already priming
        try:
            from src.tools.executor import ALL_TOOLS

            log.info(f"Priming local prompt cache ({reason})")
            system = self._memory.build_system_prompt(
                model_info=model_info_line(self._backend, "local")
            )
            self._backend.prime_cache(system, ALL_TOOLS)
        finally:
            self._busy.release()


def _setup_local_priming(backends: dict, memory, health, preferred_key: str) -> None:
    """Prime the local llama.cpp KV cache (a cold prefill takes ~a minute on the Pi) so the
    first local question does not pay for it.

    Priming keeps all CPU cores busy, so do it at startup only when local is the default
    backend. Otherwise do it when the GPU PC drops offline, which is the moment local
    becomes the default.
    """
    local_backend = backends.get("local")
    if not isinstance(local_backend, LlamaCppBackend):
        return
    primer = runtime_state.local_primer = _LocalCachePrimer(local_backend, memory)
    if preferred_key == "local" or not health.is_up(preferred_key):
        primer.prime_async("local is the default backend")
    else:
        log.info(f"Skipping startup cache priming ({preferred_key} is the default and up)")

    remote_was_up = [health.is_up("remote")]

    def _prime_when_remote_drops(statuses) -> None:
        remote = next((st for st in statuses if st.key == "remote"), None)
        if remote is None:
            return
        if remote_was_up[0] and not remote.online:
            primer.prime_async("remote went offline")
        remote_was_up[0] = remote.online

    health.add_listener(_prime_when_remote_drops)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def _summarize_in_background(backend, messages, memory, session_id):
    """Run session summarization on a background thread.

    Writes summary + topics back to the (already-closed) session record.
    """
    raw_text = ""
    if isinstance(backend, LlamaCppBackend):
        # Local summaries use the same single llama.cpp slot as the user's query; wait for
        # the assistant to go idle instead of slowing the turn that triggered the rotation.
        runtime_state.assistant_idle.wait(timeout=_LOCAL_SUMMARY_MAX_WAIT_S)
    t0 = time.perf_counter()
    try:
        for chunk in backend.stream(
            messages,
            system="You are a summarizer. Respond in the exact format requested.",
        ):
            if chunk.text:
                raw_text += chunk.text

        if raw_text.strip():
            summary_line = None
            topics_line = None
            for line in raw_text.strip().splitlines():
                line = line.strip()
                if line.lower().startswith("summary:"):
                    summary_line = line[len("summary:") :].strip()
                elif line.lower().startswith("topics:"):
                    topics_line = line[len("topics:") :].strip()

            if not summary_line:
                summary_line = raw_text.strip().splitlines()[0].strip()

            memory.update_session_summary(session_id, summary_line, topics_line)
            elapsed = time.perf_counter() - t0
            log.info(
                f"Session {session_id} summary ({elapsed:.1f}s) [{backend.name}]: {summary_line[:120]}"
            )
            if topics_line:
                log.info(f"Session topics: {topics_line}")
    except Exception as e:
        log.warning(f"Background session summary failed: {e}")
    finally:
        # A local summary evicts the primed prompt from the single llama.cpp slot.
        if isinstance(backend, LlamaCppBackend) and runtime_state.local_primer is not None:
            runtime_state.local_primer.prime_async("after local session summary")


def _maybe_end_session(
    conversation, memory, backends, cfg, last_interaction_time, session_id, health=None
):
    """Check if session has expired. If so, close and open a new one.

    Summarization runs in a background thread so SESSION_CHECK is non-blocking.
    Returns new session_id if session was ended, None otherwise.
    """
    import threading

    timeout = cfg.get("session", {}).get("inactivity_timeout_s", 300)

    if not conversation or last_interaction_time is None:
        return None

    elapsed = time.time() - last_interaction_time
    if elapsed < timeout:
        return None

    log.info(f"Session expired ({elapsed:.0f}s idle). Rotating session...")

    # Capture conversation snapshot BEFORE clearing
    should_summarize = (
        cfg.get("session", {}).get("auto_summarize", True) and memory and len(conversation) >= 4
    )
    summary_messages = None
    if should_summarize:
        # Prefer the GPU PC when it is up: summarizing locally would evict the primed
        # prompt from the Pi's single llama.cpp slot and compete with the next query.
        if health is not None and "remote" in backends and health.is_up("remote"):
            backend = backends["remote"]
        else:
            backend = backends.get("local") or backends.get("remote")
        if backend:
            summary_messages = list(conversation[-10:])
            summary_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Summarize this conversation in one sentence. "
                        "Include specific names (radio stations, recipes, people) that were mentioned or played. "
                        "Then on a second line, list 2-5 topic keywords separated by commas.\n"
                        "Format:\n"
                        "Summary: <one sentence>\n"
                        "Topics: <keyword1, keyword2, ...>"
                    ),
                }
            )

    # Close current session immediately (no summary yet) and open a fresh one
    old_session_id = session_id
    memory.close_session(session_id, summary=None, topics=None)
    conversation.clear()
    new_session_id = memory.open_session()
    log.info("Conversation history cleared for new session.")

    # Fire-and-forget: summarize in background, update the closed session record
    if summary_messages and backend:
        t = threading.Thread(
            target=_summarize_in_background,
            args=(backend, summary_messages, memory, old_session_id),
            daemon=True,
            name="session-summarizer",
        )
        t.start()
        log.info(f"Background summarization started for session {old_session_id}")

    return new_session_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def assistant_loop(cfg: dict):
    """Initialize components and run the FSM."""
    from src.state_machine import AssistantFSM

    # Initialize components
    stt_cfg = cfg["stt"]
    stt = STTEngine(
        model_size=stt_cfg["model"],
        device=stt_cfg["device"],
        compute_type=stt_cfg["compute_type"],
        beam_size=stt_cfg["beam_size"],
        language=stt_cfg["language"],
        vad_filter=stt_cfg["vad_filter"],
        alsa_device=stt_cfg.get("alsa_device"),
    )
    stt.load()

    tts = TTSEngine(
        voice=cfg["tts"]["voice"],
        aplay_device=cfg["tts"]["aplay_device"],
        sounds_dir=cfg["tts"].get("sounds_dir", "./sounds"),
        volume=int(cfg["tts"].get("volume", 80)),
    )
    tts.load_voice()  # keep Piper loaded in-process: no per-sentence voice reload

    register_alert_callback(tts.speak_alert)
    init_player(cfg)
    init_vacuum(cfg)

    backends = build_backends(cfg)

    # Health monitor — single source of truth for all backend availability.
    # Polls remote every 60s, cloud every 5min. Router and UI both read from it.
    health = BackendHealthMonitor(cfg)
    health.start()

    preferred_key = cfg["llm"].get("preferred_backend", "remote")

    # Warm-start: preload model(s) into RAM (only if reachable at startup).
    # Retry up to 3 times for Ollama in case it wasn't fully ready at boot.
    # If the remote model is not installed (404), skip warm-up for remote.
    warm_keys = ["remote", "local"] if health.is_up("remote") else ["local"]
    for key in warm_keys:
        b = backends.get(key)
        if not b or not hasattr(b, "warm"):
            continue
        for attempt in range(3):
            try:
                b.warm()
            except Exception as e:
                if "404" in str(e).lower() or "not found" in str(e).lower():
                    log.warning(f"{key}: model not installed on server — skipping warm-up")
                    break
                # transient error — retry
            if hasattr(b, "is_loaded") and b.is_loaded():
                log.info(f"{key} model confirmed loaded in Ollama RAM")
                break
            if attempt < 2:
                log.warning(
                    f"{key} model not in Ollama RAM after attempt {attempt + 1}/3"
                    " — retrying in 5s..."
                )
                time.sleep(5)
        else:
            log.warning(
                f"{key} model not confirmed in Ollama RAM after 3 attempts"
                " — first query may be slow"
            )

    router = ModelRouter(
        backends=backends,
        triggers=cfg["llm"]["router"]["triggers"],
        preferred_backend_key=preferred_key,
        health=health,
    )

    # Persistent memory (.md files for profile/facts, SQLite for sessions)
    memory = MarkdownMemoryStore()
    memory.close_stale_sessions()
    register_tool("remember", lambda fact: memory.remember(fact))
    register_tool("recall", lambda query="": memory.recall(query))

    _setup_local_priming(backends, memory, health, preferred_key)

    # Wake word detector (optional)
    wake_detector = None
    use_wake_word = "--no-wake" not in sys.argv and "--text" not in sys.argv

    if use_wake_word:
        try:
            from src.wake_word.detector import WakeWordDetector

            ww_cfg = cfg["wake_word"]
            wake_detector = WakeWordDetector(
                model=ww_cfg["model"],
                verifier_threshold=ww_cfg["verifier_threshold"],
                alsa_device=stt_cfg.get("alsa_device"),
                record_detections=ww_cfg.get("record_detections", False),
                verifier_model=ww_cfg.get("verifier_model"),
                oww_threshold=ww_cfg.get("oww_threshold", 0.1),
            )
        except Exception as e:
            log.warning(f"Wake word not available: {e}. Using keyboard mode.")

    log.info("Voice assistant ready!")
    log.info(f"Resources: {snapshot().summary()}")
    tts.play_startup()

    # Hand control to the state machine
    fsm = AssistantFSM(
        cfg=cfg,
        stt=stt,
        tts=tts,
        backends=backends,
        router=router,
        memory=memory,
        wake_detector=wake_detector,
        health=health,
    )

    if "--ui" in sys.argv:
        from src.ui.app import run_ui

        sys.exit(run_ui(fsm, health=health))
    else:
        fsm.run()


class _DedupeFilter(logging.Filter):
    """Let the first occurrence of each unique message through; drop repeats."""

    def __init__(self):
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        key = f"{record.name}:{record.getMessage()}"
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def _setup_logging():
    log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — short timestamps for journald / interactive use
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # Rotating file handler — 5 MB × 5 files = ~25 MB cap, survives reboots
    file_handler = RotatingFileHandler(
        log_dir / "assistant.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Suppress repeated identical warnings from third-party libraries
    # (e.g. openwakeword's "tflite not found, switching to onnxruntime" on every load)
    logging.getLogger("openwakeword").addFilter(_DedupeFilter())


def main():
    _setup_logging()

    cfg = load_config()
    assistant_loop(cfg)


if __name__ == "__main__":
    main()
