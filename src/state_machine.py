"""Finite State Machine for voice assistant orchestration.

Replaces the procedural loop in main.py with explicit states and transitions.
Each state manages its own resource lifecycle (mic ownership, threads, etc.).

States:
    IDLE           — waiting for wake word / keypress / text input
    SESSION_CHECK  — check inactivity timeout, rotate session if expired
    LISTENING      — STT: record + transcribe (mic owned by STT)
    THINKING       — route + LLM + tool loop + streaming TTS
    SHUTDOWN       — close session, exit
"""

import logging
import sys
import time
from enum import Enum, auto

log = logging.getLogger(__name__)


class State(Enum):
    IDLE = auto()
    SESSION_CHECK = auto()
    LISTENING = auto()
    THINKING = auto()
    SHUTDOWN = auto()


class InputMode(Enum):
    WAKE_WORD = auto()
    KEYBOARD = auto()
    TEXT = auto()


class AssistantFSM:
    """Finite state machine orchestrating the voice assistant pipeline.

    Each ``_state_*()`` method returns ``(next_state, ctx)`` where *ctx* is a
    dict carrying data between states (e.g. transcribed text, latency record).
    """

    def __init__(self, cfg, stt, tts, backends, router, memory, wake_detector=None):
        self.cfg = cfg
        self.stt = stt
        self.tts = tts
        self.backends = backends
        self.router = router
        self.memory = memory
        self.wake_detector = wake_detector

        self.conversation: list[dict] = []
        self.last_interaction_time: float | None = None
        self.session_id: int = memory.open_session()

        # Determine input mode
        if "--text" in sys.argv:
            self.input_mode = InputMode.TEXT
        elif "--no-wake" in sys.argv or wake_detector is None:
            self.input_mode = InputMode.KEYBOARD
        else:
            self.input_mode = InputMode.WAKE_WORD

        # For wake word mode, store the listen() generator
        self._wake_gen = None
        if self.input_mode == InputMode.WAKE_WORD:
            self._wake_gen = self.wake_detector.listen()

        # UI event bus — injected by run_ui() before starting FSMWorker.
        # All emit calls are gated by ``if self.ui_bus`` so console-only
        # mode is fully unaffected.
        self.ui_bus = None
        self._turn_count: int = 0

        # State handler dispatch table
        self._handlers = {
            State.IDLE: self._state_idle,
            State.SESSION_CHECK: self._state_session_check,
            State.LISTENING: self._state_listening,
            State.THINKING: self._state_thinking,
            State.SHUTDOWN: self._state_shutdown,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Dispatch to state handlers until SHUTDOWN."""
        state = State.IDLE
        ctx: dict = {}

        log.info(f"FSM started (input_mode={self.input_mode.name})")

        try:
            while state != State.SHUTDOWN:
                handler = self._handlers[state]
                prev = state
                state, ctx = handler(ctx)
                if state != prev:
                    log.info(f"State: {prev.name} → {state.name}")
                    if self.ui_bus is not None:
                        self.ui_bus.state_changed.emit(state.name)
        finally:
            self.memory.close_session(self.session_id)
            log.info("Session closed on shutdown.")

    # ------------------------------------------------------------------
    # State: IDLE
    # ------------------------------------------------------------------
    def _state_idle(self, ctx: dict) -> tuple[State, dict]:
        """Wait for user trigger based on input mode."""
        if self.input_mode == InputMode.TEXT:
            try:
                text = input("\n[Type message] ").strip()
                if not text:
                    return State.IDLE, {}
                return State.SESSION_CHECK, {"text": text}
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return State.SHUTDOWN, {}

        elif self.input_mode == InputMode.KEYBOARD:
            try:
                input("\n[Press Enter to speak] ")
                return State.SESSION_CHECK, {}
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return State.SHUTDOWN, {}

        else:  # WAKE_WORD
            try:
                from src.main import _maybe_end_session

                while True:
                    val = next(self._wake_gen)
                    if val is None:  # heartbeat — check for session expiry while idle
                        new_sid = _maybe_end_session(
                            self.conversation,
                            self.memory,
                            self.backends,
                            self.cfg,
                            self.last_interaction_time,
                            self.session_id,
                        )
                        if new_sid is not None:
                            self.session_id = new_sid
                    else:
                        break  # wake word detected; mic released
                # Pause any playing stream so STT can hear cleanly; resume() fires
                # at the end of _state_thinking after TTS has finished.
                from src.tools.player import pause

                pause()
                proc = self.tts.play_greeting()
                if proc is not None:
                    proc.wait()  # wait for greeting to finish before recording
                return State.SESSION_CHECK, {}
            except (StopIteration, KeyboardInterrupt):
                return State.SHUTDOWN, {}

    # ------------------------------------------------------------------
    # State: SESSION_CHECK
    # ------------------------------------------------------------------
    def _state_session_check(self, ctx: dict) -> tuple[State, dict]:
        """Check inactivity timeout and rotate session if expired."""
        from src.main import _maybe_end_session

        new_sid = _maybe_end_session(
            self.conversation,
            self.memory,
            self.backends,
            self.cfg,
            self.last_interaction_time,
            self.session_id,
        )
        if new_sid is not None:
            self.session_id = new_sid

        # --text mode provides text directly; skip LISTENING
        if "text" in ctx:
            return State.THINKING, ctx
        return State.LISTENING, ctx

    # ------------------------------------------------------------------
    # State: LISTENING
    # ------------------------------------------------------------------
    def _state_listening(self, ctx: dict) -> tuple[State, dict]:
        """Record and transcribe speech via STT."""
        from src.monitor import LatencyRecord, Timer

        latency = ctx.get("latency", LatencyRecord())

        with Timer() as stt_timer:
            text, rec_time, trans_time = self.stt.record_and_transcribe(
                silence_timeout_s=self.cfg["stt"]["silence_timeout_s"],
            )
        latency.stt_ms = stt_timer.elapsed_ms

        if not text:
            log.info("No speech detected.")
            return State.IDLE, {}

        if self.ui_bus is not None:
            self.ui_bus.user_said.emit(text)

        return State.THINKING, {"text": text, "latency": latency}

    # ------------------------------------------------------------------
    # State: THINKING
    # ------------------------------------------------------------------
    def _state_thinking(self, ctx: dict) -> tuple[State, dict]:
        """Route, run LLM with tools, stream TTS."""
        from src.llm.backends import ClaudeBackend, GeminiBackend, LlamaCppBackend, OllamaBackend
        from src.main import run_llm_with_tools, run_streaming_llm
        from src.monitor import LatencyRecord, Timer, check_thresholds, snapshot
        from src.tools.executor import ALL_TOOLS, VOLATILE_TOOLS

        text = ctx["text"]
        latency = ctx.get("latency", LatencyRecord())
        start = time.perf_counter()

        print(f"\n🎤 You: {text}")

        # Thinking sound — plays in parallel with LLM processing
        thinking_proc = self.tts.play_thinking()

        # Route
        with Timer() as route_timer:
            decision = self.router.route(text)
        latency.router_ms = route_timer.elapsed_ms

        backend = self.router.get_backend(decision.backend_key)
        actual_key = next(
            (k for k, v in self.router.backends.items() if v is backend), decision.backend_key
        )
        log.info(f"Router: {decision.reason} → {backend.name}")

        if self.ui_bus is not None:
            self.ui_bus.model_changed.emit(actual_key, backend.name)

        if not isinstance(backend, (OllamaBackend, LlamaCppBackend)):
            print(f"  [Using {actual_key}]")
        elif actual_key != self.router.default_backend_key:
            print(f"  [Using {actual_key}]")

        # Build messages with memory context
        self.conversation.append({"role": "user", "content": decision.cleaned_text})
        # Include tool call and tool result messages so the small local model sees
        # the full tool-call pattern and doesn't copy canned responses verbatim.
        # Keep last 6 conversational turns to limit context size on the RPi.
        recent = self.conversation[-6:]

        # Tell the model which backend it's running on
        model_info = f"You are running as: {backend.name} (backend: {decision.backend_key})"
        system_prompt = (
            self.memory.build_system_prompt(model_info=model_info) if self.memory else ""
        )

        # LLM + Tools + TTS
        tools_used: set[str] = set()
        tool_delta: list[dict] = []
        try:
            if isinstance(backend, (OllamaBackend, LlamaCppBackend, GeminiBackend, ClaudeBackend)):
                response, tools_used, tool_delta = run_llm_with_tools(
                    backend,
                    list(recent),
                    ALL_TOOLS,
                    system_prompt,
                    self.tts,
                    latency,
                    thinking_proc=thinking_proc,
                    ui_bus=self.ui_bus,
                )
            else:
                response, tools_used, tool_delta = run_streaming_llm(
                    backend,
                    list(recent),
                    ALL_TOOLS,
                    system_prompt,
                    self.tts,
                    latency,
                    thinking_proc=thinking_proc,
                    ui_bus=self.ui_bus,
                )
        except Exception as e:
            log.error(f"LLM failed: {e}")
            fallback = self.router.get_fallback(decision.backend_key)
            if fallback:
                # Find the key for the fallback backend so we can update the UI badge
                fallback_key = next(
                    (k for k, v in self.router.backends.items() if v is fallback),
                    "local",
                )
                log.info(f"Silently falling back to {fallback_key} ({fallback.name})")
                if self.ui_bus is not None:
                    self.ui_bus.model_changed.emit(fallback_key, fallback.name)
                try:
                    _local_backends = (OllamaBackend, LlamaCppBackend, GeminiBackend, ClaudeBackend)
                    if isinstance(fallback, _local_backends):
                        response, tools_used, tool_delta = run_llm_with_tools(
                            fallback,
                            list(recent),
                            ALL_TOOLS,
                            system_prompt,
                            self.tts,
                            latency,
                            thinking_proc=thinking_proc,
                            ui_bus=self.ui_bus,
                        )
                    else:
                        response, tools_used, tool_delta = run_streaming_llm(
                            fallback,
                            list(recent),
                            ALL_TOOLS,
                            system_prompt,
                            self.tts,
                            latency,
                            thinking_proc=thinking_proc,
                            ui_bus=self.ui_bus,
                        )
                except Exception:
                    response = "Sorry, I'm having trouble right now."
                    self.tts.speak(response)
            else:
                response = "Sorry, I'm having trouble right now."
                self.tts.speak(response)

        # Store response in conversation history
        if tools_used & VOLATILE_TOOLS:
            self.conversation.pop()  # Remove user message — no stale data in context
        else:
            # Extend with full delta (tool calls, results, final response) so the
            # local 2B model sees the correct tool-call pattern on the next turn.
            self.conversation.extend(tool_delta)

        # Metrics
        latency.total_ms = (time.perf_counter() - start) * 1000
        print(f"  📊 {latency.summary()}")

        if self.ui_bus is not None:
            self._turn_count += 1
            self.ui_bus.turn_count_updated.emit(self._turn_count)
            self.ui_bus.metrics_updated.emit(
                latency.stt_ms,
                latency.llm_first_token_ms,
                latency.llm_ms,
                float(latency.token_count),
                latency.model_used,
            )

        snap = snapshot()
        log.info(f"Resources: {snap.summary()}")
        thresholds = self.cfg.get("monitor", {}).get("thresholds", {})
        if thresholds:
            alerts = check_thresholds(snap, thresholds)
            for alert in alerts:
                print(f"  ⚠️  {alert}")

        self.last_interaction_time = time.time()

        # Resume any stream that pause() suspended on the wake-word trigger.
        from src.tools.player import resume

        resume()

        return State.IDLE, {}

    # ------------------------------------------------------------------
    # State: SHUTDOWN
    # ------------------------------------------------------------------
    def _state_shutdown(self, ctx: dict) -> tuple[State, dict]:
        """Terminal state. Cleanup handled by run()'s finally block."""
        return State.SHUTDOWN, {}
