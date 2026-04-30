# Handoff: Voice Assistant UI — Retro Theme (POONDYK.SYS)

## Overview
A full-screen voice assistant UI designed for a **Raspberry Pi with an 800×480 display**. The interface shows two primary states: **Idle** (clock + system status) and **Active** (listening / thinking / speaking). The "Retro" theme is a CRT terminal aesthetic — pixel font, scanlines, cyan-on-dark-navy palette, magenta accents.

## About the Design File
`Retro Design.html` is a **high-fidelity interactive prototype** built in React/JSX. It is a design reference — not production code to ship directly. Your task is to **recreate this UI in your actual project** (Python + Pygame, Qt, React, or whatever your RPi app uses), matching the visual and behavioural spec below as closely as possible.

Open the HTML in any browser to interact: use the **Tweaks panel** (bottom-right gear icon) to cycle themes, FSM states, and simulate data. The Retro theme is the target.

---

## Fidelity
**High-fidelity.** Colors, typography, spacing, layout, and animations are all final. Recreate pixel-perfectly.

---

## Canvas
- **Resolution:** 800 × 480 px (fixed, no responsive scaling needed for RPi)
- **Background:** `#04040c`
- **Font:** `Press Start 2P` (Google Fonts) — used for ALL text in Retro theme
- **Scanlines overlay:** repeating horizontal lines, `rgba(0,0,0,0.35)` every 2px — covers entire screen, `z-index` above all content
- **CRT flicker:** subtle opacity animation on root container  
  `0%,100% → opacity:1 | 93% → opacity:0.85 | 97% → opacity:0.9` over 4s loop

---

## Color Tokens (Retro Theme)

| Token | Value | Usage |
|---|---|---|
| `bg` | `#04040c` | Screen background |
| `surface` | `#080818` | Log area, input row |
| `elevated` | `#0c0c24` | (unused in retro) |
| `border` | `#0033aa` | All borders/dividers |
| `textPrimary` | `#00ffcc` | Main text, clock |
| `textSecondary` | `#0099aa` | Labels, subtitles |
| `textMuted` | `#003355` | Dim labels, dividers |
| `accent` | `#ff44dd` | Prompt `>`, cursor, ACT badge |
| Header bg | `#000820` | Top & bottom bars |
| CPU OK | `#00ffcc` | Metric value (normal) |
| CPU WARN | `#ffcc00` | Metric value (>65%) |
| CPU CRIT | `#ff4444` | Metric value (>85%) |
| TEMP WARN threshold | 60°C | → `#ffcc00` |
| TEMP CRIT threshold | 72°C | → `#ff4444` |

---

## Screen 1 — IDLE

### Layout (top → bottom, 800×480)

```
┌─────────────────────────────────────────────────────────┐  h:40
│  HEADER BAR                                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│              BIG CLOCK  (flex center)                    │  flex:1
│                                                          │
│           ┌─────────────────────────┐                   │
│           │  SYSTEM STATUS BOX      │  w:480            │
│           ├─────────────────────────┤                   │
│           │  > AWAITING INPUT █     │                   │
│           └─────────────────────────┘                   │
│                                                          │
├─────────────────────────────────────────────────────────┤  h:32
│  BOTTOM BAR                                              │
└─────────────────────────────────────────────────────────┘
```

### Header Bar (h:40, bg:`#000820`, border-bottom: 2px solid `#0033aa`)
- Left: `[ POONDYK.SYS ]` — font-size:8px, color:`#ff44dd`, letter-spacing:2px
- Right: date string e.g. `THU, APR 30, 2026` — font-size:7px, color:`#003355`, letter-spacing:1px

### Big Clock
- Font: `Press Start 2P`, size: **64px**, color: `#00ffcc`, letter-spacing: 8px
- Text shadow: `0 0 20px #00ffcc80, 0 0 40px #00ffcc30, 3px 2px 0px #aa00ff60, -3px -2px 0px #6600cc40`
- Colon blinks every 500ms (opacity 1 ↔ 0, no transition/easing)
- Format: `HH:MM` (24h)

### System Status Box (w:480, centered)
Pixel-style bordered box: `border: 2px solid #0033aa`, `box-shadow: inset -2px -2px 0 0 rgba(0,0,0,0.6), inset 2px 2px 0 0 #0033aa30, 0 0 8px #0033aa40`

**Header:** `SYSTEM STATUS` — font-size:7px, color:`#003355`, letter-spacing:3px, margin-bottom:10px

**Two-column layout inside:**

**Left column — Hardware metrics:**
- Label width: 32px, font-size:7px, color:`#0099aa`, letter-spacing:2px
- ASCII bar: 10 `█` chars, lit=`#00ffcc`, unlit=`#003355`, font-size:7px, letter-spacing:2px
- Value: font-size:7px, width:28px, right-aligned
- Rows: `CPU`, `TEMP`, `RAM`
- Divider: 1px `#0033aa`, margin 3px vertical
- `NET` row: square dot (5×5px) + `ONLINE` text in `#00ff66`, glow `0 0 5px #00ff6690`

**Right column — Backend list** (separated by 1px vertical divider + 18px margin):
- Rows: `RPI`, `GPU`, `CLAUDE`, `GEMINI`
- Label: 46px wide, font-size:7px, color:`#0099aa`, letter-spacing:1px
- Status dot: 5×5px square, online=`#00ff66` / offline=`#ff4444`, glow matching color
- Status text: `ONLINE` / `OFFLIN` (6 chars), font-size:7px, width:44px
- Sub-model: font-size:6px, color:`#003355`, flex:1
- Active backend: `ACT` badge — border:1px solid `#ff44dd`, color:`#ff44dd`, font-size:6px, padding:1px 4px, glow `0 0 5px #ff44dd50`

**Prompt row** (flush below the box, same width, bg:`#020210`, border:2px solid `#0033aa`, border-top:none, padding:9px 20px):
- `>` in `#ff44dd`, font-size:8px, margin-right:8px
- `AWAITING INPUT` in `#ff44dd`, font-size:8px, letter-spacing:3px
- Block cursor: 9×13px, `#ff44dd`, blinks every 500ms

### Bottom Bar (h:32, bg:`#000820`, border-top:2px solid `#0033aa`)
Left → right with `│` dividers (color:`#0033aa`, margin:0 12px):
1. `IP 192.168.1.42` — label muted, value `#00ffcc`, font-size:7px, letter-spacing:2px
2. `UPTIME 03H 14M` — same style, counts up in real time
3. *(if radio playing)* `♪ Radio Złote Przeboje FM` — color:`#4fc3f7`, font-size:7px; ♪ blinks 500ms
4. Right-aligned: `█ IDLE` — square dot blinks, color = IDLE state fg (`#00ffcc`), letter-spacing:2px

---

## Screen 2 — ACTIVE

Four sub-states: `LISTENING`, `THINKING`, `SPEAKING`, `IDLE`

### State Colors

| State | Label | fg | bg |
|---|---|---|---|
| LISTENING | `* REC *` | `#00ff66` | `#040c04` |
| THINKING | `* PROC *` | `#ffcc00` | `#0c0804` |
| SPEAKING | `* OUT *` | `#ff44dd` | `#04040c` |
| IDLE | `** IDLE **` | `#00ffcc` | `#04040c` |

### Layout (top → bottom)

```
┌─────────────────────────────────────────────────────────┐  h:40
│  TOP STATUS BAR                                          │
├─────────────────────────────────────────────────────────┤  h:40
│  USER INPUT LINE                                         │
├─────────────────────────────────────────────────────────┤
│  LOG AREA                                  (flex:1)      │
├─────────────────────────────────────────────────────────┤  h:36
│  BOTTOM STATS BAR                                        │
└─────────────────────────────────────────────────────────┘
```

### Top Status Bar (h:40, bg:`#000820`, border-bottom: 2px solid `{state.fg}`, box-shadow: `0 2px 8px {state.fg}50`)
- **State tag** (left): bordered box `border:2px solid {state.fg}`, bg:`{state.bg}`, padding:3px 10px, font-size:8px, letter-spacing:2px, glow `0 0 10px {fg}60`. Contains: 6×6 blinking square + state label. Square pulses during LISTENING/THINKING (`softpulse` 0.9s).
- **Model badge** (after 12px gap): `border:1px solid {m.fg}60`, padding:3px 8px, font-size:7px, letter-spacing:2px. Format: `RPi / GEMMA3-4B`
- **Time** (right): font-size:9px, color:`#0099aa`, letter-spacing:4px

### User Input Line (h:40, bg:`#040414`, border-bottom:1px solid `#0033aa`)
- `>` prompt in `#ff44dd`, margin-right:10px
- Transcript text: font-size:8px, color:`#00ffcc`, letter-spacing:1px, truncated
- If LISTENING + no transcript: `RECORDING...` in `#003355`, `softpulse` animation; followed by blinking `█` in `#00ff66`
- If no activity: `---` in `#003355`

### Log Area (flex:1, bg:`#080818`, padding:12px 14px, border-left+right:2px solid `#0033aa`)
**Thinking indicator** (THINKING state, no tools/response yet):
- `PROCESSING` + `...` in `#ffcc00`, font-size:7px, letter-spacing:2/4px, `softpulse` animation

**Tool call rows:**
- Container: border `1px solid {done?#00ff6640:#ffcc0040}`, padding:6px 10px, bg done=`#001808` / running=`#100c00`
- Status label: `[DONE]` in `#00ff66` / `[EXEC]` in `#ffcc00`, font-size:7px, letter-spacing:2px
- Result text (when done): font-size:7px, color:`#0099aa`, letter-spacing:1px

**Dashed divider** between tools and response: `border-top:1px dashed #0033aa`

**Response output:**
- Header: `SYS OUTPUT:` in `#ff44dd`, font-size:7px, letter-spacing:2px, margin-bottom:8px
- Text: font-size:11px, color:`#00ffcc`, line-height:2.2, letter-spacing:1px, text-shadow:`0 0 10px #00ffcc60`
- Streaming cursor: `█` at end, blinks every 530ms

### Bottom Stats Bar (h:36, bg:`#000820`, border-top:2px solid `#0033aa`)
Left section — metrics separated by `|` (color:`#0033aa`, margin:0 10px):
- `CPU: 75%` | `TMP: 68C` | `STT: ---` | `TPS: ---`
- Labels: font-size:7px, color:`#003355`, letter-spacing:1px, margin-right:4px
- Values: font-size:7px, color:`#0099aa`, letter-spacing:1px

Right section (after flex:1 spacer):
- *(if radio)* `| ♪ Radio Złote Przeboje FM |` — color:`#4fc3f7`, font-size:7px, ♪ blinks; then `|` divider
- `POONDYK.SYS READY` — font-size:7px, color:`#003355`, letter-spacing:2px

---

## Animations

| Name | Keyframes | Usage |
|---|---|---|
| `blink` | 0/100%→opacity:1, 50%→opacity:0 | Text cursor |
| `softpulse` | 0/100%→opacity:1, 50%→opacity:0.4 | State dot, LISTENING label |
| `crtflicker` | See above | Root container |
| `fadein` | from opacity:0 translateY(3px) → to opacity:1 translateY(0), 0.25s ease | Tool rows, response |

---

## Data / Props

Your app needs to feed the UI these live values:

| Field | Type | Notes |
|---|---|---|
| `fsm` | `IDLE \| LISTENING \| THINKING \| SPEAKING` | Voice assistant state machine |
| `modelKey` | `RPi \| GPU \| Claude \| Gemini` | Active LLM backend |
| `radio` | `string \| null` | Station name, null if not playing |
| `sys.cpu` | `number` | CPU % |
| `sys.temp` | `number` | Temperature °C |
| `sys.ramUsed` | `number` | MB used |
| `sys.ramTotal` | `number` | MB total |
| `transcript` | `string` | STT result of user utterance |
| `tools[]` | `{name, done, result}[]` | Tool calls in progress |
| `response` | `string` | LLM response (streamed) |
| `streamDone` | `bool` | Whether streaming is complete |
| `latency.stt` | `number \| null` | STT latency ms |
| `latency.tps` | `number \| null` | Tokens/sec |

---

## Files in this Package

| File | Purpose |
|---|---|
| `Retro Design.html` | Full interactive prototype — open in browser, use Tweaks panel to explore all states and themes |
| `README.md` | This document |

---

## Tips for Implementation

- **Press Start 2P** must be loaded from Google Fonts before rendering — add `<link>` or equivalent font loader
- The scanlines overlay must sit above ALL content (`z-index` highest) and be `pointer-events:none`
- ASCII progress bars use the Unicode block character `█` repeated N times — no canvas needed
- The 800×480 canvas should be centered/letterboxed if the display resolution differs
- All timers (clock, uptime, blink) should be independent intervals; blink at ~500ms, clock at 1000ms
