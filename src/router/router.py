"""Model router — picks the right LLM backend for each query.

Priority:
  1. Explicit user trigger ("use claude to...")
  2. Session preference ("switch to gemini" persists until reset)
  3. Automatic rules (default: remote if reachable, else local)
  4. Fallback: if chosen backend fails, cascade through fallback chain
"""

import logging
import re
import threading
import time
from dataclasses import dataclass

from src.llm.backends import LLMBackend, check_ollama_connectivity

log = logging.getLogger(__name__)


class RemoteAvailabilityMonitor:
    """Background thread that periodically checks if the remote Ollama server is up.

    The router reads `is_up` on every route() call and adjusts default_backend_key
    accordingly, so the assistant automatically recovers when the GPU PC comes back
    online or switches to local when it goes down — without any user intervention.
    """

    def __init__(self, remote_url: str, check_interval: int = 60):
        self._url = remote_url
        self._interval = check_interval
        self._is_up = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="remote-availability")

    def start(self, initial_state: bool) -> None:
        with self._lock:
            self._is_up = initial_state
        self._thread.start()

    @property
    def is_up(self) -> bool:
        with self._lock:
            return self._is_up

    def _loop(self) -> None:
        while True:
            time.sleep(self._interval)
            result = check_ollama_connectivity(self._url, timeout=2.0)
            with self._lock:
                prev = self._is_up
                self._is_up = result
            if result != prev:
                log.info(f"Remote Ollama at {self._url} is now {'UP' if result else 'DOWN'}")


@dataclass
class RouteDecision:
    backend_key: str        # "remote", "local", "claude", "gemini"
    reason: str             # Why this backend was chosen
    cleaned_text: str       # User text with trigger phrase removed


class ModelRouter:
    def __init__(
        self,
        backends: dict[str, LLMBackend],
        triggers: dict[str, list[str]] | None = None,
        default_backend_key: str = "local",
        monitor: RemoteAvailabilityMonitor | None = None,
    ):
        self.backends = backends
        self.triggers = triggers or {}
        self.default_backend_key = default_backend_key
        self.session_preference: str | None = None  # Sticky preference
        self._monitor = monitor

    def route(self, text: str) -> RouteDecision:
        """Decide which backend to use for this text."""
        text_lower = text.lower().strip()

        # 1. Check explicit triggers
        for key, patterns in self.triggers.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Remove trigger phrase from the text
                    cleaned = re.sub(re.escape(pattern), "", text_lower, count=1).strip()
                    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else text

                    # "auto" key resets session preference → route to default
                    if key == "auto":
                        self.session_preference = None
                        log.info("Session preference cleared → automatic routing")
                        return RouteDecision(
                            backend_key=self.default_backend_key,
                            reason=f"explicit trigger: '{pattern}' → auto ({self.default_backend_key})",
                            cleaned_text=cleaned if cleaned else text,
                        )

                    # Any explicit backend selection is sticky
                    self.session_preference = key
                    log.info(f"Session preference set → {key}")

                    return RouteDecision(
                        backend_key=key,
                        reason=f"explicit trigger: '{pattern}'",
                        cleaned_text=cleaned if cleaned else text,
                    )

        # 2. Session preference
        if self.session_preference and self.session_preference in self.backends:
            return RouteDecision(
                backend_key=self.session_preference,
                reason=f"session preference: {self.session_preference}",
                cleaned_text=text,
            )

        # 3. Default: remote (if reachable at startup) or local.
        # Sync with live remote availability — only affects the default path;
        # explicit triggers and session_preference (steps 1 & 2) are unaffected.
        if self._monitor is not None and "remote" in self.backends:
            new_default = "remote" if self._monitor.is_up else "local"
            if new_default != self.default_backend_key:
                log.info(f"Remote availability changed → default is now {new_default}")
                self.default_backend_key = new_default

        return RouteDecision(
            backend_key=self.default_backend_key,
            reason=f"default → {self.default_backend_key}",
            cleaned_text=text,
        )

    def get_backend(self, key: str) -> LLMBackend:
        """Get backend by key, with fallback to default."""
        if key in self.backends:
            return self.backends[key]
        log.warning(f"Backend '{key}' not available, falling back to {self.default_backend_key}")
        return self.backends[self.default_backend_key]

    def get_fallback(self, failed_key: str) -> LLMBackend | None:
        """Get fallback backend when primary fails.

        Fallback chain:
          remote  → local
          local   → (none)
          claude  → default (remote or local)
          gemini  → default (remote or local)
        """
        chain: dict[str, str | None] = {
            "remote": "local",
            "local": None,
            "claude": self.default_backend_key,
            "gemini": self.default_backend_key,
        }
        fallback_key = chain.get(failed_key)
        if fallback_key and fallback_key in self.backends:
            log.info(f"Falling back from {failed_key} → {fallback_key}")
            return self.backends[fallback_key]
        return None
