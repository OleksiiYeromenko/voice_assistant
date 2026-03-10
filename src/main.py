"""Voice Assistant — entry point.

Initializes components and hands control to the FSM (src/state_machine.py).
Pure functions (build_backends, run_llm_with_tools, etc.) stay here as they're
called by the FSM state handlers.
"""

import logging
import sys
import time

from src.config import load_config
from src.memory import MarkdownMemoryStore
from src.monitor import LatencyRecord, Timer, snapshot
from src.stt.engine import STTEngine
from src.tts.engine import TTSEngine
from src.llm.backends import OllamaBackend, ClaudeBackend, GeminiBackend, ToolCall, check_ollama_connectivity
from src.router.router import ModelRouter
from src.tools.executor import ALL_TOOLS, execute_tool, register_tool

log = logging.getLogger(__name__)

# Max tool-call rounds to prevent infinite loops
MAX_TOOL_ROUNDS = 3


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------
def build_backends(cfg: dict) -> tuple[dict, str]:
    """Create LLM backends from config.

    Returns (backends, default_key) where default_key is "remote" if the GPU
    PC Ollama is reachable at startup, otherwise "local" (localhost fallback).
    """
    backends = {}
    llm_cfg = cfg["llm"]
    remote_url = llm_cfg.get("remote_base_url", "http://192.168.1.74:11434")
    local_url = llm_cfg.get("local_base_url", "http://localhost:11434")
    model_cfg = llm_cfg["local"]

    # Remote backend (GPU PC)
    backends["remote"] = OllamaBackend(
        model=model_cfg["model"],
        base_url=remote_url,
        temperature=model_cfg["temperature"],
        num_ctx=model_cfg["num_ctx"],
        num_thread=model_cfg.get("num_thread"),
        think=model_cfg.get("think", False),
        label="remote",
    )

    # Local backend (localhost RPi Ollama — always-available fallback)
    backends["local"] = OllamaBackend(
        model=model_cfg["model"],
        base_url=local_url,
        temperature=model_cfg["temperature"],
        num_ctx=model_cfg["num_ctx"],
        num_thread=model_cfg.get("num_thread"),
        think=model_cfg.get("think", False),
        label="local",
    )

    # Connectivity check: prefer remote (GPU PC) if reachable
    if check_ollama_connectivity(remote_url):
        log.info(f"GPU PC Ollama reachable at {remote_url} — using as default")
        default_key = "remote"
    else:
        log.warning(
            f"GPU PC Ollama not reachable at {remote_url} — falling back to localhost"
        )
        default_key = "local"

    # Cloud backends — optional, fail gracefully
    try:
        cloud_cfg = llm_cfg["cloud"]["claude"]
        backends["claude"] = ClaudeBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Claude backend available")
    except Exception as e:
        log.warning(f"Claude backend not available: {e}")

    try:
        cloud_cfg = llm_cfg["cloud"]["gemini"]
        backends["gemini"] = GeminiBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Gemini backend available")
    except Exception as e:
        log.warning(f"Gemini backend not available: {e}")

    return backends, default_key


# ---------------------------------------------------------------------------
# LLM streaming with tool calling
# ---------------------------------------------------------------------------
def run_llm_with_tools(
    backend,
    messages: list[dict],
    tools: list[dict],
    system: str,
    tts: TTSEngine,
    latency: LatencyRecord,
    thinking_proc=None,
    ui_bus=None,
) -> tuple[str, set[str]]:
    """Run LLM with tool calling and per-sentence streaming TTS.

    Returns (response_text, tools_used) so callers can detect volatile tool usage.
    ui_bus is optional; when provided, streaming tokens and tool events are emitted.
    """
    full_response = ""
    tools_used: set[str] = set()

    for round_num in range(MAX_TOOL_ROUNDS):
        tool_calls: list[ToolCall] = []
        text_buffer = ""
        printed_prefix = False

        def token_gen():
            """Yield visible text tokens; silently collect tool_calls + thinking."""
            first_token_seen = False
            with Timer() as t:
                for chunk in backend.stream(messages, tools=tools, system=system):
                    if chunk.text:
                        if not first_token_seen:
                            first_token_seen = True
                            if round_num == 0:
                                latency.llm_first_token_ms = t.mark()
                        if ui_bus is not None:
                            ui_bus.text_chunk.emit(chunk.text)
                        yield chunk.text
                    if chunk.thinking:
                        log.debug(f"[think] {chunk.thinking}")
                    tool_calls.extend(chunk.tool_calls)
            latency.llm_ms += t.elapsed_ms
            latency.model_used = backend.name

        # Stream tokens through TTS — speaks complete sentences as they arrive.
        # pre_proc (thinking sound) is passed only on the first round; cleared inside stream_speak.
        for text in tts.stream_speak(token_gen(), latency=latency, pre_proc=thinking_proc if round_num == 0 else None):
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
            messages.append({
                "role": "assistant",
                "content": text_buffer or "",
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                tools_used.add(tc.name)
                if ui_bus is not None:
                    ui_bus.tool_started.emit(tc.name)
                result = execute_tool(tc.name, tc.arguments)
                if ui_bus is not None:
                    ui_bus.tool_done.emit(tc.name, str(result)[:80])
                messages.append({
                    "role": "tool",
                    "tool_name": tc.name,
                    "content": result,
                })
            continue

        # No tool calls — this is the final text response
        full_response = text_buffer.strip()
        break

    return full_response, tools_used


def run_streaming_llm(
    backend,
    messages: list[dict],
    tools: list[dict],
    system: str,
    tts: TTSEngine,
    latency: LatencyRecord,
    thinking_proc=None,
    ui_bus=None,
) -> tuple[str, set[str]]:
    """Stream LLM → TTS sentence-by-sentence with tool support for cloud models.

    Returns (response_text, tools_used) for consistency with run_llm_with_tools.
    Cloud path currently doesn't loop tool calls, so tools_used is always empty.
    ui_bus is optional; when provided, streaming tokens are emitted to the UI.
    """
    full_text = ""
    first_token = True

    def token_gen():
        nonlocal first_token, latency
        with Timer() as t:
            for chunk in backend.stream(messages, tools=tools, system=system):
                if first_token and chunk.text:
                    latency.llm_first_token_ms = t.mark()
                    first_token = False
                if chunk.text:
                    if ui_bus is not None:
                        ui_bus.text_chunk.emit(chunk.text)
                    yield chunk.text
        latency.llm_ms = t.elapsed_ms
        latency.model_used = backend.name

    for text in tts.stream_speak(token_gen(), latency=latency, pre_proc=thinking_proc):
        full_text += text
        print(text, end="", flush=True)

    print()  # Newline after streaming
    if ui_bus is not None:
        ui_bus.response_complete.emit()
    return full_text.strip(), set()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def _summarize_in_background(backend, messages, memory, session_id):
    """Run session summarization on a background thread.

    Writes summary + topics back to the (already-closed) session record.
    """
    raw_text = ""
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
                    summary_line = line[len("summary:"):].strip()
                elif line.lower().startswith("topics:"):
                    topics_line = line[len("topics:"):].strip()

            if not summary_line:
                summary_line = raw_text.strip().splitlines()[0].strip()

            memory.update_session_summary(session_id, summary_line, topics_line)
            if topics_line:
                log.info(f"Session topics: {topics_line}")
    except Exception as e:
        log.warning(f"Background session summary failed: {e}")


def _maybe_end_session(conversation, memory, backends, cfg, last_interaction_time, session_id):
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
        cfg.get("session", {}).get("auto_summarize", True)
        and memory
        and len(conversation) >= 4
    )
    summary_messages = None
    if should_summarize:
        backend = backends.get("local") or backends.get("remote")
        if backend:
            summary_messages = list(conversation[-10:])
            summary_messages.append({
                "role": "user",
                "content": (
                    "Summarize this conversation in one sentence. "
                    "Then on a second line, list 2-5 topic keywords separated by commas.\n"
                    "Format:\n"
                    "Summary: <one sentence>\n"
                    "Topics: <keyword1, keyword2, ...>"
                ),
            })

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
    )

    backends, default_key = build_backends(cfg)

    # Warm-start: preload model(s) into RAM.
    # Retry up to 3 times per backend in case Ollama wasn't fully ready at boot.
    warm_keys = [default_key]
    if default_key != "local":
        warm_keys.append("local")
    for key in warm_keys:
        b = backends.get(key)
        if not b or not hasattr(b, "warm"):
            continue
        for attempt in range(3):
            try:
                b.warm()
            except Exception as e:
                if "404" in str(e).lower() or "not found" in str(e).lower():
                    log.warning(f"{key}: model not installed on server — skipping retries")
                    break  # permanent failure; retrying won't help
                # transient error (connection refused, timeout) — fall through to retry
            if hasattr(b, "is_loaded") and b.is_loaded():
                log.info(f"{key} model confirmed loaded in Ollama RAM")
                break
            if attempt < 2:
                log.warning(
                    f"{key} model NOT in Ollama RAM after warm-up attempt "
                    f"{attempt + 1}/3 — retrying in 5s..."
                )
                time.sleep(5)
        else:
            log.warning(
                f"{key} model NOT confirmed in Ollama RAM after 3 attempts — "
                f"first query may be slow"
            )

    router = ModelRouter(
        backends=backends,
        triggers=cfg["llm"]["router"]["triggers"],
        default_backend_key=default_key,
    )

    # Persistent memory (.md files for profile/facts, SQLite for sessions)
    memory = MarkdownMemoryStore()
    memory.close_stale_sessions()
    register_tool("remember", lambda fact: memory.remember(fact))
    register_tool("recall", lambda query="": memory.recall(query))

    # Wake word detector (optional)
    wake_detector = None
    use_wake_word = "--no-wake" not in sys.argv and "--text" not in sys.argv

    if use_wake_word:
        try:
            from src.wake_word.detector import WakeWordDetector
            ww_cfg = cfg["wake_word"]
            wake_detector = WakeWordDetector(
                model=ww_cfg["model"],
                threshold=ww_cfg["threshold"],
                alsa_device=stt_cfg.get("alsa_device"),
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
    )

    if "--ui" in sys.argv:
        from src.ui.app import run_ui
        sys.exit(run_ui(fsm))
    else:
        fsm.run()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    assistant_loop(cfg)


if __name__ == "__main__":
    main()
