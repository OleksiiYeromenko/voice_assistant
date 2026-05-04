#!/usr/bin/env python3
"""End-to-end voice pipeline latency benchmark.

Measures the three stages that define "spoken question → first word of spoken answer":
  1. STT   — faster-whisper transcribes a 5-second audio clip
  2. LLM   — time-to-first-token (TTFT) for each configured backend
  3. TTS   — Piper synthesizes the first response sentence

Produces a per-backend breakdown table suitable for publication.

Each row = one backend.  STT and TTS are identical across backends (they run
locally on the RPi), so those columns will match — but the breakdown makes the
bottleneck visible.

Usage:
    uv run scripts/bench_e2e_latency.py                    # all backends
    uv run scripts/bench_e2e_latency.py --rounds 3         # average over 3
    uv run scripts/bench_e2e_latency.py --skip-stt         # skip STT (no whisper needed)
    uv run scripts/bench_e2e_latency.py --skip-tts         # skip TTS (no piper needed)
    uv run scripts/bench_e2e_latency.py --skip-local       # skip llama.cpp backend
    uv run scripts/bench_e2e_latency.py --skip-remote      # skip Ollama GPU backend
    uv run scripts/bench_e2e_latency.py --skip-claude      # skip Claude Haiku
    uv run scripts/bench_e2e_latency.py --skip-gemini      # skip Gemini Flash
    uv run scripts/bench_e2e_latency.py -o results.json    # save full results to JSON
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Load .env from repo root (API keys for Claude / Gemini)
# ---------------------------------------------------------------------------
_DOTENV = Path(__file__).parent.parent / ".env"
if _DOTENV.exists():
    for _line in _DOTENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip().removeprefix("export").strip()
        _v = _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain what causes thunder in one sentence.",
    "Name three planets in our solar system.",
    "What's two plus two?",
    "Write one sentence about autumn leaves.",
]

SYSTEM_PROMPT = (
    "You are a helpful home voice assistant. "
    "Keep responses concise — 1-3 sentences."
)

# Typical first sentence a voice assistant might speak (for TTS timing)
TTS_SAMPLE_SENTENCE = "The capital of France is Paris, a major European city."

WHISPER_MODEL = os.environ.get("VA_STT_MODEL", "base.en")
PIPER_VOICE = os.environ.get("VA_TTS_VOICE", "./voices/en_US-hfc_male-medium.onnx")
LLAMA_URL = os.environ.get("VA_LLAMA_URL", "http://localhost:8080")
OLLAMA_URL = os.environ.get("VA_OLLAMA_URL", "http://192.168.1.74:11434")
OLLAMA_MODEL = os.environ.get("VA_OLLAMA_MODEL", "gemma4:e4b")
CLAUDE_MODEL = os.environ.get("VA_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.environ.get("VA_GEMINI_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ComponentTime:
    value_s: float | None  # None = skipped or failed
    error: str = ""

    def fmt(self) -> str:
        if self.value_s is None:
            return "—" if not self.error else "ERR"
        return f"{self.value_s:.2f}s"


@dataclass
class BackendResult:
    label: str
    model: str
    hardware: str
    stt: ComponentTime = field(default_factory=lambda: ComponentTime(None))
    llm_ttft: ComponentTime = field(default_factory=lambda: ComponentTime(None))
    tts: ComponentTime = field(default_factory=lambda: ComponentTime(None))

    @property
    def total_s(self) -> float | None:
        parts = [c.value_s for c in (self.stt, self.llm_ttft, self.tts) if c.value_s is not None]
        if not parts:
            return None
        # Sum all measured components (missing ones are replaced with 0 for display)
        vals = [c.value_s if c.value_s is not None else 0.0 for c in (self.stt, self.llm_ttft, self.tts)]
        return sum(vals)

    def fmt_total(self) -> str:
        t = self.total_s
        return f"{t:.2f}s" if t is not None else "—"


# ---------------------------------------------------------------------------
# STT benchmark — faster-whisper
# ---------------------------------------------------------------------------
def _make_test_audio(duration_s: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a realistic speech-like audio snippet (sine sweep + noise)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    freq_sweep = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)
    audio = 0.3 * np.sin(2 * np.pi * freq_sweep * t)
    noise = 0.05 * np.random.randn(len(t)).astype(np.float32)
    return audio + noise


def bench_stt(rounds: int) -> ComponentTime:
    """Return average faster-whisper transcription latency over `rounds` runs."""
    print("\n[STT] faster-whisper")
    print(f"  Model: {WHISPER_MODEL}", end="  ", flush=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("SKIP (faster-whisper not installed)")
        return ComponentTime(None, "faster-whisper not installed")

    print("Loading...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=4)
    except Exception as e:
        print(f"FAILED: {e}")
        return ComponentTime(None, str(e))
    load_s = time.perf_counter() - t0
    print(f"{load_s:.1f}s")

    audio = _make_test_audio()
    times: list[float] = []

    for i in range(rounds):
        print(f"  Run {i + 1}/{rounds}:", end=" ", flush=True)
        t_start = time.perf_counter()
        segments, _info = model.transcribe(audio, beam_size=1, language="en")
        list(segments)  # consume generator
        elapsed = time.perf_counter() - t_start
        times.append(elapsed)
        print(f"{elapsed:.2f}s")

    avg = sum(times) / len(times)
    print(f"  Avg: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# TTS benchmark — Piper
# ---------------------------------------------------------------------------
def bench_tts(rounds: int) -> ComponentTime:
    """Return average Piper synthesis latency for a typical short sentence."""
    voice = PIPER_VOICE
    print(f"\n[TTS] Piper  voice={voice}")

    if not Path(voice).exists():
        print(f"  SKIP (voice not found at {voice})")
        return ComponentTime(None, f"voice not found: {voice}")

    try:
        subprocess.run(["piper", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  SKIP (piper binary not found)")
        return ComponentTime(None, "piper not found")

    times: list[float] = []
    for i in range(rounds):
        print(f"  Run {i + 1}/{rounds}:", end=" ", flush=True)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            t_start = time.perf_counter()
            result = subprocess.run(
                ["piper", "--model", voice, "--output_file", tmp],
                input=TTS_SAMPLE_SENTENCE.encode(),
                capture_output=True,
                timeout=30,
            )
            elapsed = time.perf_counter() - t_start
            if result.returncode != 0:
                print(f"ERROR: {result.stderr.decode()[:80]}")
                continue
            times.append(elapsed)
            print(f"{elapsed:.2f}s")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if not times:
        return ComponentTime(None, "all runs failed")
    avg = sum(times) / len(times)
    print(f"  Avg: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# LLM TTFT — llama.cpp (local RPi)
# ---------------------------------------------------------------------------
def _llama_ttft(client: httpx.Client, base_url: str, prompt: str) -> float:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_budget_tokens": 0,
    }
    t_start = time.perf_counter()
    t_first: float | None = None
    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                    return t_first - t_start
    return (t_first or time.perf_counter()) - t_start


def bench_llama(rounds: int, prompts: list[str]) -> ComponentTime:
    url = LLAMA_URL
    print(f"\n[LLM] llama.cpp (RPi5 local)  {url}")
    try:
        r = httpx.get(f"{url}/v1/models", timeout=5.0)
        r.raise_for_status()
        models = r.json().get("data", [])
        model_name = models[0]["id"] if models else "<unknown>"
        print(f"  Model: {model_name}")
    except Exception as e:
        print(f"  UNREACHABLE: {e}")
        return ComponentTime(None, str(e))

    times: list[float] = []
    with httpx.Client() as client:
        for r_idx in range(rounds):
            if rounds > 1:
                print(f"  Round {r_idx + 1}/{rounds}")
            for prompt in prompts:
                short = prompt[:50]
                print(f"    \"{short}\"", end=" ", flush=True)
                try:
                    ttft = _llama_ttft(client, url, prompt)
                    times.append(ttft)
                    print(f"TTFT {ttft:.2f}s")
                except Exception as e:
                    print(f"ERROR: {e}")

    if not times:
        return ComponentTime(None, "all runs failed")
    avg = sum(times) / len(times)
    print(f"  Avg TTFT: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# LLM TTFT — Ollama (remote GPU)
# ---------------------------------------------------------------------------
def _ollama_chat(client, **kwargs):
    try:
        return client.chat(**kwargs, think=False)
    except TypeError:
        return client.chat(**kwargs)


def bench_ollama(rounds: int, prompts: list[str]) -> ComponentTime:
    url = OLLAMA_URL
    model = OLLAMA_MODEL
    print(f"\n[LLM] Ollama (GPU remote)  {url}  model={model}")
    try:
        import ollama as ollama_lib
    except ImportError:
        print("  SKIP (ollama package not installed)")
        return ComponentTime(None, "ollama not installed")

    client = ollama_lib.Client(host=url)

    # Warmup
    print("  Warming up...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        _ollama_chat(
            client,
            model=model,
            messages=[{"role": "user", "content": "hi /no_think"}],
            options={"num_predict": 1, "num_ctx": 32},
        )
        load_s = time.perf_counter() - t0
        print(f"{load_s:.1f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        return ComponentTime(None, str(e))

    times: list[float] = []
    for r_idx in range(rounds):
        if rounds > 1:
            print(f"  Round {r_idx + 1}/{rounds}")
        for prompt in prompts:
            short = prompt[:50]
            print(f"    \"{short}\"", end=" ", flush=True)
            try:
                t_start = time.perf_counter()
                t_first: float | None = None
                for chunk in _ollama_chat(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + " /no_think"},
                    ],
                    stream=True,
                    options={"temperature": 0.0, "num_ctx": 2048},
                ):
                    content = chunk.get("message", {}).get("content", "")
                    if content and t_first is None:
                        t_first = time.perf_counter()
                        break
                ttft = (t_first - t_start) if t_first else (time.perf_counter() - t_start)
                times.append(ttft)
                print(f"TTFT {ttft:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")

    # Unload
    try:
        client.chat(model=model, messages=[{"role": "user", "content": ""}], keep_alive=0)
    except Exception:
        pass

    if not times:
        return ComponentTime(None, "all runs failed")
    avg = sum(times) / len(times)
    print(f"  Avg TTFT: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# LLM TTFT — Claude Haiku
# ---------------------------------------------------------------------------
def bench_claude(rounds: int, prompts: list[str]) -> ComponentTime:
    model = CLAUDE_MODEL
    print(f"\n[LLM] Claude  model={model}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP (ANTHROPIC_API_KEY not set)")
        return ComponentTime(None, "ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        print("  SKIP (anthropic package not installed)")
        return ComponentTime(None, "anthropic not installed")

    client = anthropic.Anthropic(api_key=api_key)
    times: list[float] = []

    for r_idx in range(rounds):
        if rounds > 1:
            print(f"  Round {r_idx + 1}/{rounds}")
        for prompt in prompts:
            short = prompt[:50]
            print(f"    \"{short}\"", end=" ", flush=True)
            try:
                t_start = time.perf_counter()
                t_first: float | None = None
                with client.messages.stream(
                    model=model,
                    max_tokens=200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for event in stream:
                        if not hasattr(event, "type"):
                            continue
                        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                            if event.delta.text and t_first is None:
                                t_first = time.perf_counter()
                                break
                ttft = (t_first - t_start) if t_first else (time.perf_counter() - t_start)
                times.append(ttft)
                print(f"TTFT {ttft:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")

    if not times:
        return ComponentTime(None, "all runs failed")
    avg = sum(times) / len(times)
    print(f"  Avg TTFT: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# LLM TTFT — Gemini
# ---------------------------------------------------------------------------
def bench_gemini(rounds: int, prompts: list[str]) -> ComponentTime:
    model = GEMINI_MODEL
    print(f"\n[LLM] Gemini  model={model}")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  SKIP (GOOGLE_API_KEY not set)")
        return ComponentTime(None, "GOOGLE_API_KEY not set")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  SKIP (google-genai package not installed)")
        return ComponentTime(None, "google-genai not installed")

    client = genai.Client(api_key=api_key)
    times: list[float] = []

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=200,
    )

    for r_idx in range(rounds):
        if rounds > 1:
            print(f"  Round {r_idx + 1}/{rounds}")
        for prompt in prompts:
            short = prompt[:50]
            print(f"    \"{short}\"", end=" ", flush=True)
            try:
                t_start = time.perf_counter()
                t_first: float | None = None
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config,
                ):
                    if chunk.text and t_first is None:
                        t_first = time.perf_counter()
                        break
                ttft = (t_first - t_start) if t_first else (time.perf_counter() - t_start)
                times.append(ttft)
                print(f"TTFT {ttft:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")

    if not times:
        return ComponentTime(None, "all runs failed")
    avg = sum(times) / len(times)
    print(f"  Avg TTFT: {avg:.2f}s")
    return ComponentTime(avg)


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------
def print_table(results: list[BackendResult], stt_shared: ComponentTime, tts_shared: ComponentTime) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  END-TO-END LATENCY: spoken question → first word of spoken answer")
    print("=" * W)
    print(f"  {'BACKEND':<28} {'HARDWARE':<18} {'STT':>7} {'LLM TTFT':>10} {'TTS 1st':>9} {'TOTAL':>8}")
    print(f"  {'':28} {'':18} {'(transcr)':>7} {'(1st token)':>10} {'(synth)':>9} {'':>8}")
    print("-" * W)

    for r in results:
        stt_val = r.stt.fmt() if r.stt.value_s is not None else stt_shared.fmt()
        tts_val = r.tts.fmt() if r.tts.value_s is not None else tts_shared.fmt()
        llm_val = r.llm_ttft.fmt()

        # Compute total using shared STT/TTS if backend didn't measure them
        stt_s = r.stt.value_s if r.stt.value_s is not None else (stt_shared.value_s or 0.0)
        tts_s = r.tts.value_s if r.tts.value_s is not None else (tts_shared.value_s or 0.0)
        llm_s = r.llm_ttft.value_s or 0.0
        total = stt_s + llm_s + tts_s
        total_fmt = f"{total:.2f}s" if (stt_s + llm_s + tts_s > 0) else "—"

        print(f"  {r.label:<28} {r.hardware:<18} {stt_val:>7} {llm_val:>10} {tts_val:>9} {total_fmt:>8}")

    print("=" * W)
    if stt_shared.value_s is not None:
        print(f"\n  STT note : {WHISPER_MODEL} (faster-whisper, CPU int8) — identical for all backends")
    if tts_shared.value_s is not None:
        print(f"  TTS note : Piper ({Path(PIPER_VOICE).stem}) — identical for all backends")
    print(f"  LLM note : TTFT = time from prompt submission to first streamed token")
    print(f"  Prompts  : {len(PROMPTS)} voice-assistant questions, averaged over rounds")
    print("=" * W)


def save_json(
    results: list[BackendResult],
    stt_shared: ComponentTime,
    tts_shared: ComponentTime,
    path: str,
) -> None:
    data = {
        "stt_shared": {"model": WHISPER_MODEL, "avg_s": stt_shared.value_s, "error": stt_shared.error},
        "tts_shared": {"voice": PIPER_VOICE, "avg_s": tts_shared.value_s, "error": tts_shared.error},
        "backends": [
            {
                "label": r.label,
                "model": r.model,
                "hardware": r.hardware,
                "stt_s": r.stt.value_s,
                "llm_ttft_s": r.llm_ttft.value_s,
                "tts_s": r.tts.value_s,
                "total_s": r.total_s,
                "errors": {
                    k: v
                    for k, v in [
                        ("stt", r.stt.error),
                        ("llm", r.llm_ttft.error),
                        ("tts", r.tts.error),
                    ]
                    if v
                },
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end voice pipeline latency benchmark (STT + LLM + TTS)"
    )
    parser.add_argument("--rounds", type=int, default=2,
                        help="Rounds per prompt per backend (default: 2)")
    parser.add_argument("--skip-stt", action="store_true", help="Skip STT benchmark")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS benchmark")
    parser.add_argument("--skip-local", action="store_true", help="Skip llama.cpp backend")
    parser.add_argument("--skip-remote", action="store_true", help="Skip Ollama GPU backend")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude Haiku backend")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini backend")
    parser.add_argument("--output", "-o", default=None, help="Save results to JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("  Voice Pipeline E2E Latency Benchmark")
    print("=" * 70)
    print(f"  Rounds  : {args.rounds}")
    print(f"  Prompts : {len(PROMPTS)}")
    print(f"  STT     : {'SKIP' if args.skip_stt else WHISPER_MODEL}")
    print(f"  TTS     : {'SKIP' if args.skip_tts else PIPER_VOICE}")
    print(f"  Local   : {'SKIP' if args.skip_local else LLAMA_URL}")
    print(f"  Remote  : {'SKIP' if args.skip_remote else f'{OLLAMA_URL}  {OLLAMA_MODEL}'}")
    print(f"  Claude  : {'SKIP' if args.skip_claude else CLAUDE_MODEL}")
    print(f"  Gemini  : {'SKIP' if args.skip_gemini else GEMINI_MODEL}")
    print("=" * 70)

    # --- shared components (same for all backends) ---
    stt_shared = bench_stt(args.rounds) if not args.skip_stt else ComponentTime(None, "skipped")
    tts_shared = bench_tts(args.rounds) if not args.skip_tts else ComponentTime(None, "skipped")

    # --- per-backend LLM benchmarks ---
    results: list[BackendResult] = []

    if not args.skip_local:
        llm = bench_llama(args.rounds, PROMPTS)
        results.append(BackendResult(
            label="llama.cpp (RPi5 local)",
            model="gemma4-e2b-q4km",
            hardware="RPi5 16GB (CPU)",
            llm_ttft=llm,
        ))

    if not args.skip_remote:
        llm = bench_ollama(args.rounds, PROMPTS)
        results.append(BackendResult(
            label=f"Ollama ({OLLAMA_MODEL})",
            model=OLLAMA_MODEL,
            hardware="GTX 1070 GPU",
            llm_ttft=llm,
        ))

    if not args.skip_claude:
        llm = bench_claude(args.rounds, PROMPTS)
        results.append(BackendResult(
            label="Claude Haiku",
            model=CLAUDE_MODEL,
            hardware="Anthropic cloud",
            llm_ttft=llm,
        ))

    if not args.skip_gemini:
        llm = bench_gemini(args.rounds, PROMPTS)
        results.append(BackendResult(
            label="Gemini Flash",
            model=GEMINI_MODEL,
            hardware="Google cloud",
            llm_ttft=llm,
        ))

    if not results:
        print("\nNo backends selected — nothing to compare.")
        sys.exit(1)

    print_table(results, stt_shared, tts_shared)

    if args.output:
        save_json(results, stt_shared, tts_shared, args.output)


if __name__ == "__main__":
    main()
