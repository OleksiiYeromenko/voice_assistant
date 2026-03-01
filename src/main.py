"""Voice Assistant — main loop.

Wires together: wake word → STT → router → LLM (+ tools) → streaming TTS
"""

import logging
import sys
import threading
import time
from pathlib import Path

from src.config import load_config
from src.memory import MarkdownMemoryStore
from src.monitor import LatencyRecord, Timer, check_thresholds, snapshot
from src.stt.engine import STTEngine
from src.tts.engine import TTSEngine
from src.llm.backends import OllamaBackend, ClaudeBackend, GeminiBackend, LLMChunk, ToolCall, check_ollama_connectivity
from src.router.router import ModelRouter
from src.tools.executor import ALL_TOOLS, VOLATILE_TOOLS, execute_tool, register_tool

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
    # System prompt is now built dynamically per-turn by MarkdownMemoryStore
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
) -> tuple[str, set[str]]:
    """Run LLM with tool calling and per-sentence streaming TTS.

    Returns (response_text, tools_used) so callers can detect volatile tool usage.
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
                        yield chunk.text
                    if chunk.thinking:
                        log.debug(f"[think] {chunk.thinking}")
                    tool_calls.extend(chunk.tool_calls)
            latency.llm_ms += t.elapsed_ms
            latency.model_used = backend.name

        # Stream tokens through TTS — speaks complete sentences as they arrive
        for text in tts.stream_speak(token_gen(), latency=latency):
            if not printed_prefix:
                print("\n🤖 ", end="", flush=True)
                printed_prefix = True
            text_buffer += text
            print(text, end="", flush=True)

        if text_buffer.strip():
            print()  # newline after streaming

        # If tool calls were issued, execute them and loop
        if tool_calls:
            messages.append({"role": "assistant", "content": text_buffer or ""})

            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                tools_used.add(tc.name)
                result = execute_tool(tc.name, tc.arguments)
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
) -> tuple[str, set[str]]:
    """Stream LLM → TTS sentence-by-sentence with tool support for cloud models.

    Returns (response_text, tools_used) for consistency with run_llm_with_tools.
    Cloud path currently doesn't loop tool calls, so tools_used is always empty.
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
                    yield chunk.text
        latency.llm_ms = t.elapsed_ms
        latency.model_used = backend.name

    for text in tts.stream_speak(token_gen(), latency=latency):
        full_text += text
        print(text, end="", flush=True)

    print()  # Newline after streaming
    return full_text.strip(), set()


# ---------------------------------------------------------------------------
# TTS interruption via background wake word detection
# ---------------------------------------------------------------------------
def _start_interrupt_listener(wake_detector, tts):
    """Spawn a background thread that listens for wake word and interrupts TTS.

    Returns (thread, stop_event) so the caller can stop the listener
    before the main wake-word loop resumes (avoiding ALSA device conflicts).
    """
    if wake_detector is None:
        return None, None

    stop_event = threading.Event()

    def _listen():
        try:
            if wake_detector.detect_once(timeout_s=120, stop_event=stop_event):
                log.info("Wake word detected during TTS — interrupting")
                tts.interrupt()
        except Exception as e:
            log.debug(f"Interrupt listener error: {e}")

    t = threading.Thread(target=_listen, daemon=True, name="interrupt-listener")
    t.start()
    return t, stop_event


def _stop_interrupt_listener(thread, stop_event):
    """Signal the interrupt listener to stop and wait for it to release the mic."""
    if thread is None:
        return
    stop_event.set()
    thread.join(timeout=2)  # 2s max — detect_once checks stop_event every 80ms


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def _maybe_end_session(conversation, memory, backends, cfg, last_interaction_time, session_id):
    """Check if session has expired. If so, summarize, close, and open a new one.

    Returns new session_id if session was ended, None otherwise.
    """
    timeout = cfg.get("session", {}).get("inactivity_timeout_s", 300)

    if not conversation or last_interaction_time is None:
        return None

    elapsed = time.time() - last_interaction_time
    if elapsed < timeout:
        return None

    log.info(f"Session expired ({elapsed:.0f}s idle). Summarizing...")

    summary_line = None
    topics_line = None

    if cfg.get("session", {}).get("auto_summarize", True) and memory and len(conversation) >= 4:
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
            raw_text = ""
            try:
                for chunk in backend.stream(
                    summary_messages,
                    system="You are a summarizer. Respond in the exact format requested.",
                ):
                    if chunk.text:
                        raw_text += chunk.text

                if raw_text.strip():
                    # Parse structured response
                    for line in raw_text.strip().splitlines():
                        line = line.strip()
                        if line.lower().startswith("summary:"):
                            summary_line = line[len("summary:"):].strip()
                        elif line.lower().startswith("topics:"):
                            topics_line = line[len("topics:"):].strip()

                    # Fallback: if model didn't follow format, use entire text
                    if not summary_line:
                        summary_line = raw_text.strip().splitlines()[0].strip()

                    log.info(f"Session summary: {summary_line}")
                    if topics_line:
                        log.info(f"Session topics: {topics_line}")
            except Exception as e:
                log.warning(f"Session summary failed: {e}")

    # Close current session and open a fresh one
    memory.close_session(session_id, summary=summary_line, topics=topics_line)
    conversation.clear()
    new_session_id = memory.open_session()
    log.info("Conversation history cleared for new session.")
    return new_session_id


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def assistant_loop(cfg: dict):
    """Main loop: wake word → STT → LLM → TTS, repeat."""
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
    )

    backends, default_key = build_backends(cfg)

    # Warm-start: preload model(s) into RAM.
    # Always warm the default backend first; also warm localhost if it's not the default
    # so the fallback is ready without a cold-start penalty.
    warm_keys = [default_key]
    if default_key != "local":
        warm_keys.append("local")
    for key in warm_keys:
        b = backends.get(key)
        if b and hasattr(b, "warm"):
            b.warm()

    router = ModelRouter(
        backends=backends,
        triggers=cfg["llm"]["router"]["triggers"],
        default_backend_key=default_key,
    )

    # Persistent memory (.md files for profile/facts, SQLite for sessions)
    memory = MarkdownMemoryStore()
    memory.close_stale_sessions()  # close any sessions left open from crash/restart
    current_session_id = memory.open_session()

    register_tool("remember", lambda fact: memory.remember(fact))
    register_tool("recall", lambda query="": memory.recall(query))

    # Conversation history
    conversation: list[dict] = []
    last_interaction_time: float | None = None

    # Input mode: --text (type messages) | --no-wake (press Enter → STT) | default (wake word)
    use_text_input = "--text" in sys.argv
    use_wake_word = "--no-wake" not in sys.argv and not use_text_input
    wake_detector = None

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
            use_wake_word = False

    log.info("Voice assistant ready!")
    log.info(f"Resources: {snapshot().summary()}")

    try:
        if use_text_input:
            log.info("Text input mode — type your message, Ctrl+C to quit.")
            while True:
                try:
                    user_text = input("\n[Type message] ").strip()
                    if not user_text:
                        continue
                    new_sid = _maybe_end_session(conversation, memory, backends, cfg, last_interaction_time, current_session_id)
                    if new_sid is not None:
                        current_session_id = new_sid
                    _handle_interaction(
                        stt, tts, router, backends,
                        conversation, cfg, memory=memory, text=user_text,
                    )
                    last_interaction_time = time.time()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
        elif use_wake_word:
            log.info("Say the wake word to start...")
            for confidence in wake_detector.listen():
                tts.play_beep()  # Acknowledge wake word (non-blocking, plays on speaker)
                new_sid = _maybe_end_session(conversation, memory, backends, cfg, last_interaction_time, current_session_id)
                if new_sid is not None:
                    current_session_id = new_sid
                _handle_interaction(
                    stt, tts, router, backends,
                    conversation, cfg, memory=memory, wake_detector=wake_detector,
                )
                last_interaction_time = time.time()
        else:
            log.info("Keyboard mode — press Enter to speak, Ctrl+C to quit.")
            while True:
                try:
                    input("\n[Press Enter to speak] ")
                    new_sid = _maybe_end_session(conversation, memory, backends, cfg, last_interaction_time, current_session_id)
                    if new_sid is not None:
                        current_session_id = new_sid
                    _handle_interaction(
                        stt, tts, router, backends,
                        conversation, cfg, memory=memory,
                    )
                    last_interaction_time = time.time()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
    finally:
        # Graceful shutdown: close the current session
        memory.close_session(current_session_id)
        log.info("Session closed on shutdown.")


def _handle_interaction(
    stt, tts, router, backends, conversation, cfg,
    *, memory=None, wake_detector=None, text=None,
):
    """Handle one full interaction cycle."""
    latency = LatencyRecord()
    start = time.perf_counter()

    # 1. STT (skip if text provided directly, e.g. --text mode)
    if text is None:
        with Timer() as stt_timer:
            text, rec_time, trans_time = stt.record_and_transcribe(
                silence_timeout_s=cfg["stt"]["silence_timeout_s"],
            )
        latency.stt_ms = stt_timer.elapsed_ms

        if not text:
            log.info("No speech detected.")
            return

    print(f"\n🎤 You: {text}")

    # 2. Route
    with Timer() as route_timer:
        decision = router.route(text)
    latency.router_ms = route_timer.elapsed_ms

    backend = router.get_backend(decision.backend_key)
    log.info(f"Router: {decision.reason} → {backend.name}")

    if not isinstance(backend, OllamaBackend):
        print(f"  [Using {decision.backend_key}]")
    elif decision.backend_key != router.default_backend_key:
        print(f"  [Using {decision.backend_key} ollama]")

    # 3. Build messages with memory context
    conversation.append({"role": "user", "content": decision.cleaned_text})
    recent = conversation[-10:]  # Last 5 turns

    system_prompt = memory.build_system_prompt() if memory else ""

    # 4. Start interrupt listener (background wake word detection during TTS)
    #    Started AFTER STT so mic is free.
    interrupt_thread, interrupt_stop = _start_interrupt_listener(wake_detector, tts)

    # 5. LLM + Tools + TTS
    # OllamaBackend (both "remote" and "local") supports full tool calling.
    # Cloud backends (Claude, Gemini) use streaming-only path.
    tools_used: set[str] = set()
    try:
        if isinstance(backend, OllamaBackend):
            response, tools_used = run_llm_with_tools(
                backend, list(recent), ALL_TOOLS, system_prompt, tts, latency,
            )
        else:
            response, tools_used = run_streaming_llm(
                backend, list(recent), ALL_TOOLS, system_prompt, tts, latency,
            )
    except Exception as e:
        log.error(f"LLM failed: {e}")
        fallback = router.get_fallback(decision.backend_key)
        if fallback:
            print(f"  [Falling back to {fallback.name}]")
            try:
                if isinstance(fallback, OllamaBackend):
                    response, tools_used = run_llm_with_tools(
                        fallback, list(recent), ALL_TOOLS, system_prompt, tts, latency,
                    )
                else:
                    response, tools_used = run_streaming_llm(
                        fallback, list(recent), ALL_TOOLS, system_prompt, tts, latency,
                    )
            except Exception as e2:
                response = "Sorry, I'm having trouble right now."
                tts.speak(response)
        else:
            response = "Sorry, I'm having trouble right now."
            tts.speak(response)
    finally:
        # CRITICAL: Stop interrupt listener BEFORE returning so its arecord
        # releases the mic. Otherwise listen() can't reopen the device.
        _stop_interrupt_listener(interrupt_thread, interrupt_stop)

    # Store response in conversation history. For volatile tools (e.g., get_time),
    # drop the entire exchange (user + assistant) so no stale time value or confusing
    # placeholder ends up in context. The system prompt enforces fresh get_time calls.
    if tools_used & VOLATILE_TOOLS:
        conversation.pop()  # Remove the user message appended before the LLM call
        # No assistant message added — history stays clean for the next turn
    else:
        conversation.append({"role": "assistant", "content": response})

    # 6. Log metrics + resource alerts
    latency.total_ms = (time.perf_counter() - start) * 1000
    print(f"  📊 {latency.summary()}")
    snap = snapshot()
    log.info(f"Resources: {snap.summary()}")
    thresholds = cfg.get("monitor", {}).get("thresholds", {})
    if thresholds:
        alerts = check_thresholds(snap, thresholds)
        for alert in alerts:
            print(f"  ⚠️  {alert}")


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