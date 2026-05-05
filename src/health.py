"""Unified backend health monitor — single source of truth for all backend availability.

Polls remote Ollama every 60s and cloud backends every 5min via real API checks
(list-models endpoint, zero token cost). Local RPi is always considered healthy.

Router reads .is_up(key) on each route decision.
UI subscribes via add_listener() to receive BackendStatus updates.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

_POLL_INTERVALS: dict[str, int] = {
    "remote": 300,
    "claude": 300,
    "gemini": 300,
}


@dataclass
class BackendStatus:
    key: str      # "local" | "remote" | "claude" | "gemini"
    label: str    # Display label: RPI / GPU / CLAUDE / GEMINI
    online: bool
    model: str
    active: bool = False


class BackendHealthMonitor:
    """Thread-safe health monitor for all LLM backends.

    Start with .start(); stop with .stop(). Safe to read is_up() from any thread.
    """

    def __init__(self, cfg: dict):
        llm = cfg.get("llm", {})
        self._remote_url: str = llm.get("remote_base_url", "http://192.168.1.74:11434")
        self._remote_model: str = llm.get("remote_model", "gemma4:e4b")
        self._local_model: str = llm.get("local", {}).get("model", "local")
        cloud = llm.get("cloud", {})
        self._claude_model: str = cloud.get("claude", {}).get("model", "claude")
        self._gemini_model: str = cloud.get("gemini", {}).get("model", "gemini")

        preferred = llm.get("preferred_backend", "remote")
        self._active_key: str = preferred

        self._lock = threading.Lock()
        self._statuses: dict[str, BackendStatus] = {
            "local":  BackendStatus("local",  "RPI",    True,  self._local_model,
                                    preferred == "local"),
            "remote": BackendStatus("remote", "GPU",    False, self._remote_model,
                                    preferred == "remote"),
            "claude": BackendStatus("claude", "CLAUDE", False, self._claude_model,
                                    preferred == "claude"),
            "gemini": BackendStatus("gemini", "GEMINI", False, self._gemini_model,
                                    preferred == "gemini"),
        }
        self._listeners: list = []
        self._running = False

    def start(self) -> None:
        """Start background polling threads and run an immediate first check."""
        self._running = True
        for key, interval in _POLL_INTERVALS.items():
            threading.Thread(
                target=self._poll_loop, args=(key, interval),
                daemon=True, name=f"health-{key}",
            ).start()
            # Immediate check (non-blocking)
            threading.Thread(
                target=self._check_and_update, args=(key,), daemon=True,
            ).start()

    def stop(self) -> None:
        self._running = False

    def is_up(self, key: str) -> bool:
        if key == "local":
            return True
        with self._lock:
            return self._statuses.get(key, BackendStatus(key, key, False, "")).online

    def set_active(self, key: str) -> None:
        with self._lock:
            self._active_key = key
            for k, s in self._statuses.items():
                self._statuses[k] = BackendStatus(s.key, s.label, s.online, s.model, k == key)
        self._notify()

    def get_statuses(self) -> list[BackendStatus]:
        with self._lock:
            return list(self._statuses.values())

    def add_listener(self, cb) -> None:
        """Register a callback(list[BackendStatus]) called on every status change."""
        self._listeners.append(cb)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self, key: str, interval: int) -> None:
        while self._running:
            time.sleep(interval)
            self._check_and_update(key)

    def _check_and_update(self, key: str) -> None:
        from src.llm.backends import (
            check_claude_availability,
            check_gemini_availability,
            check_ollama_connectivity,
        )

        if key == "remote":
            online = check_ollama_connectivity(self._remote_url, timeout=2.0)
            reason = "" if online else "unreachable"
        elif key == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            online, reason = check_claude_availability(api_key)
        elif key == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            online, reason = check_gemini_availability(api_key)
        else:
            return

        with self._lock:
            prev = self._statuses[key]
            if prev.online != online:
                log.info(
                    f"Backend '{key}' is now {'UP' if online else f'DOWN ({reason})'}"
                )
            self._statuses[key] = BackendStatus(
                prev.key, prev.label, online, prev.model,
                prev.key == self._active_key,
            )

        self._notify()

    def _notify(self) -> None:
        statuses = self.get_statuses()
        for cb in self._listeners:
            try:
                cb(statuses)
            except Exception as e:
                log.warning(f"Health listener error: {e}")
