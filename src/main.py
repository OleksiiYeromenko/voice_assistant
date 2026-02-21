"""Voice Assistant — main loop.

Wires together: wake word → STT → router → LLM (+ tools) → streaming TTS
"""

import logging
import sys
import time
from pathlib import Path

from src.config import load_config
from src.monitor import LatencyRecord, Timer, snapshot
from src.stt.engine import STTEngine
from src.tts.engine import TTSEngine
from src.llm.backends import OllamaBackend, ClaudeBackend, GeminiBackend, LLMChunk, ToolCall
from src.router.router import ModelRouter
from src.tools.executor import ALL_TOOLS, execute_tool

log = logging.getLogger(__name__)

# Max tool-call rounds to prevent infinite loops
MAX_TOOL_ROUNDS = 3


def build_backends(cfg: dict) -> dict:
    """Create LLM backends from config."""
    backends = {}

    # Local is always available
    local_cfg = cfg["llm"]["local"]
    backends["local"] = OllamaBackend(
        model=local_cfg["model"],
        base_url=local_cfg["base_url"],
        temperature=local_cfg["temperature"],
        num_ctx=local_cfg["num_ctx"],
        system_prompt=local_cfg["system_prompt"],
    )

    # Cloud backends — optional, fail gracefully
    try:
        cloud_cfg = cfg["llm"]["cloud"]["claude"]
        backends["claude"] = ClaudeBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Claude backend available")
    except Exception as e:
        log.warning(f"Claude backend not available: {e}")

    try:
        cloud_cfg = cfg["llm"]["cloud"]["gemini"]
        backends["gemini"] = GeminiBackend(
            model=cloud_cfg["model"],
            max_tokens=cloud_cfg["max_tokens"],
        )
        log.info("Gemini backend available")
    except Exception as e:
        log.warning(f"Gemini backend not available: {e}")

    return backends


def run_llm_with_tools(
    backend,
    messages: list[dict],
    tools: list[dict],
    system: str,
    tts: TTSEngine,
    latency: LatencyRecord,
) -> str:
    """Run LLM, handle tool calls, stream TTS. Returns full response text."""
    full_response = ""

    for round_num in range(MAX_TOOL_ROUNDS):
        # Collect streaming response
        text_buffer = ""
        tool_calls: list[ToolCall] = []
        first_token = True

        with Timer() as llm_timer:
            for chunk in backend.stream(messages, tools=tools, system=system):
                if first_token and chunk.text:
                    latency.llm_first_token_ms = llm_timer.mark()
                    first_token = False

                text_buffer += chunk.text
                tool_calls.extend(chunk.tool_calls)

        latency.llm_ms += llm_timer.elapsed_ms
        latency.model_used = backend.name

        # If there are tool calls, execute them and continue
        if tool_calls:
            # Add assistant's tool call message
            messages.append({"role": "assistant", "content": text_buffer or ""})

            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                result = execute_tool(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_name": tc.name,
                    "content": result,
                })

            # Loop back for the LLM to generate a final response with tool results
            continue

        # No tool calls — this is the final text response
        full_response = text_buffer.strip()
        break

    # Speak the final response with streaming TTS
    if full_response:
        tts.speak(full_response)

    return full_response


def run_streaming_llm(
    backend,
    messages: list[dict],
    tools: list[dict],
    system: str,
    tts: TTSEngine,
    latency: LatencyRecord,
) -> str:
    """Stream LLM → TTS sentence-by-sentence. No tool calling in this path."""
    full_text = ""
    first_token = True

    def token_gen():
        nonlocal first_token, latency
        with Timer() as t:
            for chunk in backend.stream(messages, system=system):
                if first_token and chunk.text:
                    latency.llm_first_token_ms = t.mark()
                    first_token = False
                if chunk.text:
                    yield chunk.text
        latency.llm_ms = t.elapsed_ms
        latency.model_used = backend.name

    for text in tts.stream_speak(token_gen()):
        full_text += text
        print(text, end="", flush=True)

    print()  # Newline after streaming
    return full_text.strip()


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
        mic_device_index=stt_cfg.get("mic_device_index"),
    )
    stt.load()

    tts = TTSEngine(
        voice=cfg["tts"]["voice"],
        aplay_device=cfg["tts"]["aplay_device"],
    )

    backends = build_backends(cfg)
    router = ModelRouter(
        backends=backends,
        triggers=cfg["llm"]["router"]["triggers"],
    )

    system_prompt = cfg["llm"]["local"]["system_prompt"]

    # Conversation history (simple list, could add persistence later)
    conversation: list[dict] = []

    # Wake word (optional — can also run in keyboard mode)
    use_wake_word = "--no-wake" not in sys.argv
    wake_detector = None

    if use_wake_word:
        try:
            from src.wake_word.detector import WakeWordDetector
            ww_cfg = cfg["wake_word"]
            wake_detector = WakeWordDetector(
                model=ww_cfg["model"],
                threshold=ww_cfg["threshold"],
                mic_device_index=stt_cfg.get("mic_device_index"),
            )
        except Exception as e:
            log.warning(f"Wake word not available: {e}. Using keyboard mode.")
            use_wake_word = False

    log.info("Voice assistant ready!")
    log.info(f"Resources: {snapshot().summary()}")

    if use_wake_word:
        log.info("Say the wake word to start...")
        for confidence in wake_detector.listen():
            _handle_interaction(stt, tts, router, backends, system_prompt, conversation, cfg)
    else:
        log.info("Keyboard mode — press Enter to speak, Ctrl+C to quit.")
        while True:
            try:
                input("\n[Press Enter to speak] ")
                _handle_interaction(stt, tts, router, backends, system_prompt, conversation, cfg)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break


def _handle_interaction(stt, tts, router, backends, system_prompt, conversation, cfg):
    """Handle one full interaction cycle."""
    latency = LatencyRecord()
    start = time.perf_counter()

    # 1. STT
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

    if decision.backend_key != "local":
        print(f"  [Using {decision.backend_key}]")

    # 3. Build messages
    conversation.append({"role": "user", "content": decision.cleaned_text})

    # Keep conversation short to fit context window
    recent = conversation[-10:]  # Last 5 turns

    # 4. LLM + Tools + TTS
    try:
        # For local model: use tool calling path
        if decision.backend_key == "local":
            response = run_llm_with_tools(
                backend, list(recent), ALL_TOOLS, system_prompt, tts, latency,
            )
        else:
            # Cloud models: stream directly (simpler, tools can be added later)
            response = run_streaming_llm(
                backend, list(recent), ALL_TOOLS, system_prompt, tts, latency,
            )
    except Exception as e:
        log.error(f"LLM failed: {e}")
        # Fallback
        fallback = router.get_fallback(decision.backend_key)
        if fallback:
            print(f"  [Falling back to {fallback.name}]")
            try:
                response = run_llm_with_tools(
                    fallback, list(recent), ALL_TOOLS, system_prompt, tts, latency,
                )
            except Exception as e2:
                response = "Sorry, I'm having trouble right now."
                tts.speak(response)
        else:
            response = "Sorry, I'm having trouble right now."
            tts.speak(response)

    conversation.append({"role": "assistant", "content": response})

    # 5. Log metrics
    latency.total_ms = (time.perf_counter() - start) * 1000
    print(f"  📊 {latency.summary()}")
    log.info(f"Resources: {snapshot().summary()}")


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