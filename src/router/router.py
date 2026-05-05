"""Model router — picks the right LLM backend for each query.

Priority:
  1. Explicit user trigger ("use claude to...") — is_explicit=True, no fallback
  2. Session preference ("switch to gemini" persists until reset) — is_explicit=False
  3. Automatic: preferred_backend if healthy, else "local"
"""

import logging
import re
from dataclasses import dataclass

from src.llm.backends import LLMBackend

log = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    backend_key: str   # "remote", "local", "claude", "gemini"
    reason: str        # Why this backend was chosen
    cleaned_text: str  # User text with trigger phrase removed
    is_explicit: bool  # True when a trigger phrase matched this turn


class ModelRouter:
    def __init__(
        self,
        backends: dict[str, LLMBackend],
        triggers: dict[str, list[str]] | None = None,
        preferred_backend_key: str = "local",
        health=None,  # BackendHealthMonitor | None
    ):
        self.backends = backends
        self.triggers = triggers or {}
        self.preferred_backend_key = preferred_backend_key
        self.session_preference: str | None = None
        self._health = health

    def route(self, text: str) -> RouteDecision:
        text_lower = text.lower().strip()

        # 1. Explicit trigger phrase
        for key, patterns in self.triggers.items():
            for pattern in patterns:
                if pattern in text_lower:
                    cleaned = re.sub(re.escape(pattern), "", text_lower, count=1).strip()
                    # Residual noise ≤3 chars (e.g. "GPU" from "switch to remote GPU")
                    # is not a real query — treat as empty so the FSM can skip the LLM.
                    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 3 else ""

                    if key == "auto":
                        self.session_preference = None
                        log.info("Session preference cleared → automatic routing")
                        default = self._default_key()
                        return RouteDecision(
                            backend_key=default,
                            reason=f"explicit trigger: '{pattern}' → auto ({default})",
                            cleaned_text=cleaned or text,
                            is_explicit=True,
                        )

                    self.session_preference = key
                    log.info(f"Session preference set → {key}")
                    return RouteDecision(
                        backend_key=key,
                        reason=f"explicit trigger: '{pattern}'",
                        cleaned_text=cleaned or text,
                        is_explicit=True,
                    )

        # 2. Sticky session preference
        if self.session_preference and self.session_preference in self.backends:
            return RouteDecision(
                backend_key=self.session_preference,
                reason=f"session preference: {self.session_preference}",
                cleaned_text=text,
                is_explicit=False,
            )

        # 3. Automatic: preferred if healthy, else local
        chosen = self._default_key()
        return RouteDecision(
            backend_key=chosen,
            reason=f"default → {chosen}",
            cleaned_text=text,
            is_explicit=False,
        )

    def _default_key(self) -> str:
        """Return preferred backend if healthy, else 'local'."""
        if self._health is None:
            return self.preferred_backend_key
        if self._health.is_up(self.preferred_backend_key):
            return self.preferred_backend_key
        return "local"

    def get_backend(self, key: str) -> LLMBackend:
        """Get backend by key. Falls back to local if key not registered."""
        if key in self.backends:
            return self.backends[key]
        log.warning(f"Backend '{key}' not registered, falling back to local")
        return self.backends["local"]

    def get_fallback(self, failed_key: str) -> LLMBackend | None:
        """Fallback chain for non-explicit failures: preferred → local only.

        Cloud backends (claude, gemini) fall back to the current default.
        Returns None if failed_key is already local.
        """
        if failed_key == "local":
            return None
        fallback_key = "local"
        if fallback_key in self.backends and fallback_key != failed_key:
            log.info(f"Falling back from {failed_key} → {fallback_key}")
            return self.backends[fallback_key]
        return None
