"""Internet radio playback via mpv.

Radio stations are discovered through the Radio Browser API (free, no key)
and streamed by a background ``mpv`` process. mpv exposes a JSON IPC socket
so we can pause / resume / set volume / quit without killing the process.

State is module-level (same pattern as ``src/tools/timers.py``) and protected
by a single lock so the LLM's tool thread and the state-machine's pause/resume
hooks can't race.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time

import httpx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module state (set by init_player at startup)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_mpv_proc: subprocess.Popen | None = None
_current_station: str | None = None
_current_url: str | None = None  # URL of the active/last stream
_was_playing_before_pause: bool = False
_paused_url: str | None = None  # saved URL so resume() can restart
_paused_station: str | None = None  # saved station name for display

_radio_change_callback = None  # (station_name: str) → None; "" means stopped
_ipc_path: str = "/tmp/va-mpv.sock"
_radio_browser_base: str = "https://de1.api.radio-browser.info"
_search_limit: int = 5
_request_timeout_s: float = 5.0
_audio_device: str | None = None  # e.g. "plughw:2,0" (passed as alsa/<device>)
_preferred_volume: int = 70


def register_radio_callback(cb) -> None:
    """Register a callable(station_name: str) notified on play/stop.

    Called with the station name when playback starts, "" when it stops.
    Not called during pause/resume cycles so the UI doesn't flicker during TTS.
    """
    global _radio_change_callback
    _radio_change_callback = cb


def _notify_radio(station: str) -> None:
    if _radio_change_callback is not None:
        try:
            _radio_change_callback(station)
        except Exception:
            pass


def init_player(cfg: dict) -> None:
    """Capture config values needed by play_radio / set_volume.

    Called once from main.py at startup. After this, the tool functions can be
    invoked by the LLM without needing to pass cfg around.
    """
    global _ipc_path, _radio_browser_base, _search_limit, _request_timeout_s
    global _audio_device, _preferred_volume

    pcfg = cfg.get("player", {}) or {}
    _ipc_path = pcfg.get("ipc_socket", _ipc_path)
    _radio_browser_base = pcfg.get("radio_browser_base", _radio_browser_base)
    _search_limit = int(pcfg.get("search_limit", _search_limit))
    _request_timeout_s = float(pcfg.get("request_timeout_s", _request_timeout_s))
    _preferred_volume = _clamp_volume(pcfg.get("default_volume", _preferred_volume))

    _audio_device = cfg.get("tts", {}).get("aplay_device")

    log.info(
        f"Player initialised: ipc={_ipc_path} volume={_preferred_volume} device={_audio_device}"
    )


# ---------------------------------------------------------------------------
# Public tool functions (exposed to the LLM)
# ---------------------------------------------------------------------------


def play_radio(query: str) -> str:
    """Search Radio Browser for ``query`` and stream the top result via mpv."""
    query = (query or "").strip()
    if not query:
        return "I need a station name or genre to play."

    try:
        stations = _search_radio_browser(query)
    except Exception as e:
        log.error(f"Radio Browser search failed: {e}")
        return f"Couldn't reach the radio directory: {e}"

    if not stations:
        return f"No stations found for '{query}'."

    started_name: str | None = None
    last_err: Exception | None = None
    with _lock:
        _stop_locked()  # stop anything currently playing before starting a new station
        for s in stations:
            url = s.get("url_resolved") or s.get("url")
            name = s.get("name", "").strip() or query
            if not url:
                continue
            try:
                _start_mpv_locked(url, name)
                started_name = name
                break
            except Exception as e:
                last_err = e
                log.warning(f"mpv failed for station '{name}': {e}")

    if started_name is not None:
        _notify_radio(started_name)
        return f"Playing {started_name}."
    return f"Couldn't start any station for '{query}'" + (f": {last_err}." if last_err else ".")


def stop_playback() -> str:
    """Stop the currently playing stream, if any."""
    with _lock:
        # _mpv_proc may be None if pause() already killed mpv to free the device,
        # but _was_playing_before_pause signals that a stream was active.
        if _mpv_proc is None and not _was_playing_before_pause:
            return "Nothing is playing."
        name = _current_station or _paused_station or "playback"
        _stop_locked()  # clears _paused_url / _paused_station too
    _notify_radio("")
    return f"Stopped {name}."


def set_volume(level: int) -> str:
    """Set music-only volume (0-100). TTS is unaffected.

    Preference is remembered across stop/play so the next ``play_radio`` honours it.
    """
    global _preferred_volume
    try:
        level = int(level)
    except (TypeError, ValueError):
        return "Volume must be a number between 0 and 100."
    level = _clamp_volume(level)

    with _lock:
        _preferred_volume = level
        if _mpv_proc is not None:
            try:
                _ipc_command_locked(["set_property", "volume", level])
            except Exception as e:
                log.warning(f"Volume IPC failed: {e}")
                return f"Set volume preference to {level}, but couldn't apply it live."

    return f"Volume set to {level}."


# ---------------------------------------------------------------------------
# Internal API (called by the state machine, not the LLM)
# ---------------------------------------------------------------------------


def is_playing() -> bool:
    with _lock:
        return _mpv_proc is not None and _mpv_proc.poll() is None


def get_radio_status() -> dict:
    """Return current radio state for LLM context injection.

    Returns {"station": str | None, "active": bool} — active is True while
    a stream is playing or paused-for-TTS (i.e. it will resume after THINKING).
    """
    with _lock:
        station = _current_station or _paused_station
        active = station is not None and (_mpv_proc is not None or _was_playing_before_pause)
    return {"station": station, "active": active}


def pause() -> None:
    """Kill the current stream to release the audio device; save state for resume().

    IPC-pausing mpv does not reliably close the exclusive ALSA device, so we
    terminate the process outright and restart it in resume() once TTS is done.
    """
    global _was_playing_before_pause, _paused_url, _paused_station
    was_killed = False
    with _lock:
        if _mpv_proc is None or _mpv_proc.poll() is not None:
            return
        # Read stream info before _stop_locked() clears it.
        url = _current_url
        station = _current_station
        _stop_locked()
        if url:
            _was_playing_before_pause = True
            _paused_url = url
            _paused_station = station
            was_killed = True
    if was_killed:
        # ALSA holds the device briefly after the process exits; give it a moment
        # so that the greeting sound played immediately after can open the device.
        time.sleep(0.15)


def resume() -> None:
    """Resume the stream that pause() suspended.

    Two cases:
    - pause() killed a running stream → restart mpv from the saved URL.
    - play_radio() started mpv with --pause this turn → just IPC-unpause it.
    """
    global _was_playing_before_pause, _paused_url, _paused_station
    resumed_station: str | None = None
    with _lock:
        if not _was_playing_before_pause:
            return
        _was_playing_before_pause = False

        if _paused_url is not None:
            # Restart the stream that was killed in pause().
            url = _paused_url
            station = _paused_station or "radio"
            _paused_url = None
            _paused_station = None
            try:
                _start_mpv_locked(url, station, start_paused=False)
                resumed_station = station
            except Exception as e:
                log.warning(f"Resume (restart) failed: {e}")
        elif _mpv_proc is not None and _mpv_proc.poll() is None:
            # mpv is alive but started with --pause (play_radio this turn); just unpause.
            try:
                _ipc_command_locked(["set_property", "pause", False])
                resumed_station = _current_station
            except Exception as e:
                log.warning(f"Resume IPC failed: {e}")
    if resumed_station:
        _notify_radio(resumed_station)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clamp_volume(v) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return 70
    return max(0, min(100, v))


def _search_radio_browser(query: str) -> list[dict]:
    """Return up to ``_search_limit`` candidate stations, best-first.

    Tries a name search first, then falls back to a tag search if nothing hits.
    ``hidebroken=true`` filters stations Radio Browser knows are down.
    """
    params_common = {
        "hidebroken": "true",
        "order": "clickcount",
        "reverse": "true",
        "limit": str(_search_limit),
    }
    headers = {"User-Agent": "voice-assistant/0.1 (+radio-playback)"}

    def _get(path: str, extra: dict) -> list[dict]:
        url = f"{_radio_browser_base.rstrip('/')}{path}"
        params = {**params_common, **extra}
        r = httpx.get(url, params=params, headers=headers, timeout=_request_timeout_s)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    results = _get("/json/stations/search", {"name": query})
    if not results:
        results = _get("/json/stations/search", {"tag": query})
    return results


def _start_mpv_locked(url: str, station_name: str, start_paused: bool = True) -> None:
    """Launch mpv and wait briefly for the IPC socket to appear.

    Caller must hold ``_lock``. Raises on failure (mpv dies or socket never appears).

    start_paused=True  → started with --pause so TTS can use the audio device in the
                         same turn; resume() will unpause at end of THINKING.
    start_paused=False → starts playing immediately (used by resume() after kill-pause).
    """
    global _mpv_proc, _current_station, _current_url, _was_playing_before_pause

    # Stale socket from a previously killed run would prevent a clean bind.
    try:
        if os.path.exists(_ipc_path):
            os.unlink(_ipc_path)
    except OSError:
        pass

    cmd = [
        "mpv",
        "--no-video",
        "--really-quiet",
        "--idle=no",
        "--no-terminal",
        "--audio-stream-silence=no",
        f"--input-ipc-server={_ipc_path}",
        f"--volume={_preferred_volume}",
    ]
    if start_paused:
        cmd.append("--pause")
    if _audio_device:
        cmd.append(f"--audio-device=alsa/{_audio_device}")
    cmd.append(url)

    log.info(f"Starting mpv for '{station_name}' (paused={start_paused}): {url}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 2s for the IPC socket to appear; bail early if mpv exits.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"mpv exited immediately (code {proc.returncode})")
        if os.path.exists(_ipc_path):
            break
        time.sleep(0.05)
    else:
        # Socket never appeared; kill and error so caller can try next candidate.
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError("mpv did not open IPC socket within 2s")

    _mpv_proc = proc
    _current_station = station_name
    _current_url = url
    # When started paused, signal resume() to unpause at end of THINKING.
    _was_playing_before_pause = start_paused


def _stop_locked() -> None:
    """Terminate the running player. Caller must hold ``_lock``."""
    global _mpv_proc, _current_station, _current_url, _was_playing_before_pause
    global _paused_url, _paused_station

    proc = _mpv_proc
    _mpv_proc = None
    _current_station = None
    _current_url = None
    _was_playing_before_pause = False
    _paused_url = None
    _paused_station = None

    if proc is None:
        return

    # Try a graceful "quit" via IPC first; fall back to SIGTERM/SIGKILL.
    try:
        _send_ipc(["quit"])
    except Exception:
        pass

    try:
        proc.wait(timeout=1.5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    try:
        if os.path.exists(_ipc_path):
            os.unlink(_ipc_path)
    except OSError:
        pass


def _ipc_command_locked(command: list) -> None:
    """Send a single JSON IPC command to mpv. Caller must hold ``_lock``."""
    _send_ipc(command)


def _send_ipc(command: list) -> None:
    """Open the mpv UNIX socket, write one newline-terminated JSON command, close."""
    payload = (json.dumps({"command": command}) + "\n").encode()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect(_ipc_path)
        sock.sendall(payload)
    finally:
        sock.close()
