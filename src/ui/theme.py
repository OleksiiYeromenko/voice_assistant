"""Dark theme color palette and stylesheets for the RPi Touch Display 2 UI."""

# ---------------------------------------------------------------------------
# Background layers
# ---------------------------------------------------------------------------
BG_DEEPEST = "#0D0D1A"    # main window / outermost
BG_SURFACE = "#1A1A2E"    # chat text area, card surfaces
BG_ELEVATED = "#22223A"   # bars, badge backgrounds

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
TEXT_PRIMARY = "#E8E8E8"
TEXT_SECONDARY = "#9090A0"
TEXT_MUTED = "#606070"
TEXT_ACCENT = "#4FC3F7"    # "You:" prefix, interactive highlights

# ---------------------------------------------------------------------------
# State pill — (background, foreground) per state name
# ---------------------------------------------------------------------------
STATE_COLORS: dict[str, tuple[str, str]] = {
    "IDLE":          ("#2D3748", "#718096"),
    "SESSION_CHECK": ("#2D3748", "#718096"),
    "LISTENING":     ("#1B5E20", "#69F0AE"),
    "THINKING":      ("#4A3000", "#FFB300"),
    "INTERRUPTED":   ("#4A1500", "#FF7043"),
    "SHUTDOWN":      ("#1A0000", "#EF5350"),
}

# ---------------------------------------------------------------------------
# Model badge — (background, foreground) per backend key
# ---------------------------------------------------------------------------
MODEL_COLORS: dict[str, tuple[str, str]] = {
    "remote":  ("#006064", "#00E5FF"),   # GPU PC — teal/cyan
    "local":   ("#1B5E20", "#69F0AE"),   # RPi local — green
    "claude":  ("#4A1A00", "#FF9800"),   # Claude — orange
    "gemini":  ("#1A0060", "#B388FF"),   # Gemini — purple
}
MODEL_LABELS: dict[str, str] = {
    "remote": "GPU",
    "local":  "RPi",
    "claude": "Claude",
    "gemini": "Gemini",
}
MODEL_ICONS: dict[str, str] = {
    "remote": "⚡",   # GPU PC — fast/powerful
    "local":  "◉",   # Raspberry Pi — compact local device
    "claude": "☁",   # Claude cloud API
    "gemini": "☁",   # Gemini cloud API
}

# ---------------------------------------------------------------------------
# Idle screen typography
# ---------------------------------------------------------------------------
IDLE_CLOCK_FONT_SIZE = 120
IDLE_DATE_FONT_SIZE = 28

# ---------------------------------------------------------------------------
# Resource thresholds (CPU & temp gauge colors)
# ---------------------------------------------------------------------------
COLOR_OK   = "#4CAF50"   # < 70% CPU / < 60°C
COLOR_WARN = "#FFC107"   # < 85% CPU / < 75°C
COLOR_CRIT = "#F44336"   # ≥ 85% CPU / ≥ 75°C


def cpu_color(pct: float) -> str:
    if pct >= 85:
        return COLOR_CRIT
    if pct >= 70:
        return COLOR_WARN
    return COLOR_OK


def temp_color(celsius: float) -> str:
    if celsius >= 75:
        return COLOR_CRIT
    if celsius >= 60:
        return COLOR_WARN
    return COLOR_OK


# ---------------------------------------------------------------------------
# Main application stylesheet (applied once to QMainWindow)
# ---------------------------------------------------------------------------
MAIN_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DEEPEST};
    color: {TEXT_PRIMARY};
    font-family: "DejaVu Sans";
    font-size: 15px;
}}

QTextEdit {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: none;
    font-size: 18px;
    padding: 8px 12px;
    selection-background-color: #3A3A5A;
}}

QScrollBar:vertical {{
    width: 0px;
    background: transparent;
}}

QScrollBar:horizontal {{
    height: 0px;
    background: transparent;
}}

QProgressBar {{
    background-color: {BG_ELEVATED};
    border: none;
    border-radius: 3px;
    text-align: left;
}}

QProgressBar::chunk {{
    border-radius: 3px;
    background-color: {COLOR_OK};
}}

QFrame[frameShape="4"] {{
    color: #2A2A45;
    background-color: #2A2A45;
    border: none;
    max-height: 1px;
}}
"""
