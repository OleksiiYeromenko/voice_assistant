"""Backend availability monitoring for the retro idle status panel."""
from __future__ import annotations

import logging
import os
import socket
import threading
from typing import NamedTuple
from urllib.parse import urlparse

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

log = logging.getLogger(__name__)

_POLL_INTERVAL_MS = 30_000
_TIMEOUT_S = 2.0


class BackendStatus(NamedTuple):
    key: str        # "local" | "remote" | "claude" | "gemini"
    label: str      # Display label: RPI / GPU / CLAUDE / GEMINI
    online: bool
    sub_model: str
    active: bool = False


class BackendMonitor(QObject):
    """Polls each backend for availability every 30 s.

    Lives on the main Qt thread; polling runs in a daemon thread.
    Cross-thread signal emission is safe via Qt's auto-queued connections.
    """

    status_updated = pyqtSignal(list)   # list[BackendStatus]
    net_updated = pyqtSignal(bool)      # internet reachable

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        llm = cfg.get("llm", {})
        self._remote_url = llm.get("remote_base_url", "http://192.168.1.74:11434")
        self._local_url = llm.get("local_base_url", "http://localhost:8080")
        self._local_model = llm.get("local", {}).get("model", "gemma3-4b")
        cloud = llm.get("cloud", {})
        self._claude_model = cloud.get("claude", {}).get("model", "claude-3.5")
        self._gemini_model = cloud.get("gemini", {}).get("model", "gemini-2.5")
        self._active_key: str = "local"
        self._running = True
        self._lock = threading.Lock()
        self._last_statuses: list[BackendStatus] = []

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._schedule_poll)
        self._timer.start()

        # Initial poll (non-blocking)
        self._schedule_poll()

    def set_active(self, key: str):
        """Update which backend is currently active without re-polling."""
        self._active_key = key
        with self._lock:
            updated = [s._replace(active=s.key == key) for s in self._last_statuses]
            self._last_statuses = updated
        if updated:
            self.status_updated.emit(list(updated))

    def stop(self):
        self._running = False
        self._timer.stop()

    def _schedule_poll(self):
        if self._running:
            threading.Thread(target=self._do_poll, daemon=True).start()

    def _do_poll(self):
        if not self._running:
            return

        net = self._check_tcp("8.8.8.8", 53)
        local_ok = self._check_tcp(*self._parse_host_port(self._local_url))
        remote_ok = self._check_tcp(*self._parse_host_port(self._remote_url))
        claude_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        gemini_ok = bool(os.environ.get("GOOGLE_API_KEY"))

        active = self._active_key
        statuses: list[BackendStatus] = [
            BackendStatus("local",  "RPI",    local_ok,  self._local_model,   active == "local"),
            BackendStatus("remote", "GPU",    remote_ok, "phi-4-mini",        active == "remote"),
            BackendStatus("claude", "CLAUDE", claude_ok, self._claude_model,  active == "claude"),
            BackendStatus("gemini", "GEMINI", gemini_ok, self._gemini_model,  active == "gemini"),
        ]

        with self._lock:
            self._last_statuses = statuses

        self.status_updated.emit(statuses)
        self.net_updated.emit(net)

    @staticmethod
    def _parse_host_port(url: str) -> tuple[str, int]:
        try:
            p = urlparse(url)
            return p.hostname or "localhost", p.port or 80
        except Exception:
            return "localhost", 80

    @staticmethod
    def _check_tcp(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=_TIMEOUT_S):
                return True
        except OSError:
            return False
