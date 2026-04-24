"""Thread-safe Qt signal bus for UI ↔ FSM communication.

The FSMWorker thread emits signals here; the main Qt thread receives them.
Qt's queued connection mechanism handles cross-thread delivery automatically.
"""

from PyQt6.QtCore import QObject, pyqtSignal


class UIEventBus(QObject):
    """All signals the FSM emits to drive the UI.

    Emitters: FSMWorker thread (state_machine.py, main.py)
    Receivers: MainWindow widgets (main Qt thread)
    """

    # Current FSM state name, e.g. "IDLE", "LISTENING", "THINKING"
    state_changed = pyqtSignal(str)

    # One LLM response token (streamed)
    text_chunk = pyqtSignal(str)

    # LLM response complete for this turn
    response_complete = pyqtSignal()

    # Transcribed user speech
    user_said = pyqtSignal(str)

    # Tool call lifecycle
    tool_started = pyqtSignal(str)  # tool name
    tool_done = pyqtSignal(str, str)  # tool name, short result (≤80 chars)

    # Latency after each full pipeline pass: stt_ms, ttft_ms, llm_ms, token_count, model_name
    metrics_updated = pyqtSignal(float, float, float, float, str)

    # System resource snapshot: cpu_percent, ram_used_mb, ram_total_mb, temp_celsius|None
    resource_updated = pyqtSignal(float, float, float, object)

    # Active backend changed: backend_key ("remote"/"local"/"claude"/"gemini"), model display name
    model_changed = pyqtSignal(str, str)

    # Conversation turn counter (increments after each assistant response)
    turn_count_updated = pyqtSignal(int)

    # Radio playback changed: station name when playing, "" when stopped
    radio_changed = pyqtSignal(str)

    # FSM has exited — time to close the window
    shutdown_requested = pyqtSignal()
