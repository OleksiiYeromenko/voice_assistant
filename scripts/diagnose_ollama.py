#!/usr/bin/env python3
"""Diagnose Ollama performance on Raspberry Pi.

Checks all known factors that affect inference speed:
CPU governor, thermal throttling, thread detection, Ollama env vars,
memory bandwidth, KV cache settings, and Ollama version.

Usage:
    python scripts/diagnose_ollama.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def run(cmd: str, timeout: int = 5) -> str | None:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip()
    except Exception:
        return None


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} ──{RESET}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_cpu_info() -> None:
    section("CPU Info")
    model = run("cat /proc/cpuinfo | grep 'Model' | head -1 | cut -d: -f2")
    if model:
        print(f"  Board: {model.strip()}")

    # Current frequency
    freq = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    max_freq = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    if freq and max_freq:
        cur_mhz = int(freq) / 1000
        max_mhz = int(max_freq) / 1000
        print(f"  CPU freq: {cur_mhz:.0f} MHz (max: {max_mhz:.0f} MHz)")
        if cur_mhz < max_mhz * 0.95:
            warn(f"CPU running below max — check governor or thermal throttling")
        else:
            ok(f"CPU at max frequency")

    cores = run("nproc")
    if cores:
        print(f"  Cores: {cores}")


def check_cpu_governor() -> None:
    section("CPU Governor")
    governor = run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor:
        print(f"  Current: {governor}")
        if governor == "performance":
            ok("Governor set to 'performance' — good")
        else:
            warn(f"Governor is '{governor}' — CPU may downclock between tokens")
            print(f"  {YELLOW}FIX:{RESET} echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    else:
        warn("Could not read CPU governor")


def check_thermal() -> None:
    section("Thermal")
    temp = run("cat /sys/class/thermal/thermal_zone0/temp")
    if temp:
        celsius = int(temp) / 1000
        print(f"  CPU temp: {celsius:.1f}°C")
        if celsius > 80:
            bad(f"Thermal throttling likely! ({celsius:.1f}°C > 80°C)")
            print(f"  {RED}FIX:{RESET} Improve cooling (active fan, heatsink)")
        elif celsius > 70:
            warn(f"Getting warm ({celsius:.1f}°C) — may throttle under sustained load")
        else:
            ok(f"Temperature OK")

    # Check for throttling events
    throttled = run("vcgencmd get_throttled 2>/dev/null")
    if throttled:
        print(f"  Throttle status: {throttled}")
        if "0x0" in throttled:
            ok("No throttling detected")
        else:
            bad("Throttling events detected! Check power supply and cooling")
            # Decode common flags
            try:
                val = int(throttled.split("=")[1], 16)
                if val & 0x1:
                    bad("  → Currently under-voltage (bad PSU?)")
                if val & 0x2:
                    bad("  → ARM frequency currently capped")
                if val & 0x4:
                    bad("  → Currently throttled (overheating)")
                if val & 0x8:
                    bad("  → Soft temperature limit active")
                if val & 0x10000:
                    warn("  → Under-voltage has occurred since boot")
                if val & 0x20000:
                    warn("  → ARM frequency capping has occurred since boot")
                if val & 0x40000:
                    warn("  → Throttling has occurred since boot")
                if val & 0x80000:
                    warn("  → Soft temperature limit reached since boot")
            except (ValueError, IndexError):
                pass


def check_memory() -> None:
    section("Memory")
    meminfo = run("grep MemTotal /proc/meminfo")
    if meminfo:
        kb = int(meminfo.split()[1])
        gb = kb / 1024 / 1024
        print(f"  Total RAM: {gb:.1f} GB")

    available = run("grep MemAvailable /proc/meminfo")
    if available:
        kb = int(available.split()[1])
        gb = kb / 1024 / 1024
        print(f"  Available: {gb:.1f} GB")
        if gb < 2:
            bad("Less than 2GB available — models may swap!")
        elif gb < 4:
            warn(f"Only {gb:.1f}GB free — tight for larger models")
        else:
            ok(f"{gb:.1f}GB available")

    # Check swap activity
    swap = run("grep SwapTotal /proc/meminfo")
    swap_used = run("grep SwapFree /proc/meminfo")
    if swap and swap_used:
        total = int(swap.split()[1])
        free = int(swap_used.split()[1])
        used_mb = (total - free) / 1024
        if used_mb > 100:
            bad(f"Swap in use: {used_mb:.0f}MB — inference will be very slow!")
        else:
            ok(f"No significant swap usage ({used_mb:.0f}MB)")


def check_ollama_version() -> None:
    section("Ollama Version")
    version = run("ollama --version 2>/dev/null") or run("ollama -v 2>/dev/null")
    if version:
        print(f"  {version}")
        # Check for known ARM64 regression in 0.9.3+
        # Extract version number
        for part in version.split():
            if part and part[0].isdigit():
                try:
                    parts = part.split(".")
                    major, minor = int(parts[0]), int(parts[1])
                    if major == 0 and minor >= 9:
                        patch = int(parts[2]) if len(parts) > 2 else 0
                        if minor > 9 or (minor == 9 and patch >= 3):
                            warn(f"v{part}: Known ARM64 performance regression (issue #11238)")
                            print(f"  {YELLOW}TIP:{RESET} Consider testing v0.9.2 if speed is worse than expected")
                        else:
                            ok(f"Version pre-0.9.3 — no known ARM64 regression")
                    else:
                        ok(f"Version OK")
                except (ValueError, IndexError):
                    pass
                break
    else:
        bad("Ollama not found in PATH")


def check_ollama_env() -> None:
    section("Ollama Environment Variables")

    # Check systemd override
    override_paths = [
        "/etc/systemd/system/ollama.service.d/override.conf",
        "/etc/systemd/system/ollama.service",
    ]

    env_vars: dict[str, str] = {}
    for path in override_paths:
        content = run(f"cat {path} 2>/dev/null")
        if content:
            print(f"  Config from: {path}")
            for line in content.splitlines():
                if "Environment=" in line:
                    # Parse Environment="KEY=VALUE"
                    eq = line.split("Environment=", 1)[1].strip('"').strip("'")
                    k, _, v = eq.partition("=")
                    env_vars[k] = v

    # Also check current process env
    ollama_pid = run("pgrep -x ollama")
    if ollama_pid:
        pid = ollama_pid.splitlines()[0]
        environ = run(f"cat /proc/{pid}/environ 2>/dev/null | tr '\\0' '\\n' | grep OLLAMA")
        if environ:
            for line in environ.splitlines():
                k, _, v = line.partition("=")
                if k.startswith("OLLAMA_"):
                    env_vars[k] = v

    # Report on critical vars
    critical = {
        "OLLAMA_NUM_PARALLEL": ("1", "default may auto-select >1, multiplying KV cache memory"),
        "OLLAMA_MAX_LOADED_MODELS": ("1", "prevents multiple models competing for RAM"),
        "OLLAMA_FLASH_ATTENTION": ("1", "required for KV cache quantization"),
        "OLLAMA_KV_CACHE_TYPE": ("q8_0", "halves KV cache memory vs f16 default"),
    }

    if not env_vars:
        warn("No Ollama environment variables set — using all defaults")
        print(f"  {YELLOW}FIX:{RESET} sudo systemctl edit ollama")
    for key, (recommended, reason) in critical.items():
        val = env_vars.get(key)
        if val == recommended:
            ok(f"{key}={val}")
        elif val:
            warn(f"{key}={val} (recommended: {recommended} — {reason})")
        else:
            warn(f"{key} not set (recommended: {recommended} — {reason})")


def check_ollama_thread_detection() -> None:
    section("Ollama Thread Detection (ARM64 Bug #11221)")

    # Check if /proc/cpuinfo has x86-style fields
    has_physical_id = run("grep -c 'physical id' /proc/cpuinfo 2>/dev/null")
    has_siblings = run("grep -c 'siblings' /proc/cpuinfo 2>/dev/null")

    if has_physical_id == "0" and has_siblings == "0":
        warn("ARM64 cpuinfo lacks x86 thread fields — Ollama may misdetect thread count")
        print(f"  {YELLOW}FIX:{RESET} Create a Modelfile with: PARAMETER num_thread 4")
    else:
        ok("CPU info has standard thread detection fields")

    # Check Ollama's actual thread usage during idle
    ollama_threads = run("ps -eo pid,nlwp,comm | grep ollama | head -1")
    if ollama_threads:
        parts = ollama_threads.split()
        if len(parts) >= 2:
            print(f"  Ollama process threads: {parts[1]}")


def check_ollama_loaded_models() -> None:
    section("Currently Loaded Models")
    try:
        import ollama
        client = ollama.Client()
        ps = client.ps()
        if ps.models:
            for m in ps.models:
                size_gb = m.size / (1024**3) if hasattr(m, 'size') else 0
                vram = m.size_vram / (1024**3) if hasattr(m, 'size_vram') else 0
                print(f"  {m.model}: {size_gb:.1f}GB (VRAM: {vram:.1f}GB)")
            if len(ps.models) > 1:
                warn(f"{len(ps.models)} models loaded — they compete for memory!")
                print(f"  {YELLOW}FIX:{RESET} Set OLLAMA_MAX_LOADED_MODELS=1")
        else:
            ok("No models currently loaded (cold start)")
    except Exception as e:
        warn(f"Could not query Ollama: {e}")


def check_context_settings() -> None:
    section("Context / KV Cache Impact")
    print("  Context size directly affects speed via KV cache memory pressure.")
    print("  Smaller context → less memory bandwidth wasted → faster tok/s")
    print()
    print("  Recommended for voice assistant:")
    print(f"    num_ctx: 1024-2048 (you have 4096 in config.yaml)")
    warn("Consider reducing num_ctx from 4096 to 2048 in config.yaml")


def check_memory_bandwidth() -> None:
    section("Memory Bandwidth (estimated)")
    # Try to detect RAM type
    ram_info = run("sudo dmidecode -t memory 2>/dev/null | grep -i speed | head -2")
    if ram_info:
        print(f"  {ram_info}")
    else:
        # Pi 5 specific
        print("  RPi 5: LPDDR4X-4267 → theoretical peak ~34 GB/s")
        print("  Real-world usable for LLM inference: ~17-25 GB/s")
        print("  This is the primary bottleneck — not CPU clock speed")

    # Quick bandwidth test if mbw is available
    if shutil.which("mbw"):
        print("  Running quick bandwidth test (mbw)...")
        bw = run("mbw -n 3 256 2>/dev/null | grep AVG | tail -1", timeout=30)
        if bw:
            print(f"  {bw}")
    else:
        print(f"  {YELLOW}TIP:{RESET} Install 'mbw' to measure actual bandwidth: sudo apt install mbw")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"{BOLD}Ollama RPi Performance Diagnostics{RESET}")
    print("=" * 50)

    check_cpu_info()
    check_cpu_governor()
    check_thermal()
    check_memory()
    check_ollama_version()
    check_ollama_env()
    check_ollama_thread_detection()
    check_ollama_loaded_models()
    check_context_settings()
    check_memory_bandwidth()

    # Summary
    section("Recommended Actions (quick wins)")
    print("""
  1. Set CPU governor to performance:
     echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

  2. Set Ollama env vars (sudo systemctl edit ollama):
     [Service]
     Environment="OLLAMA_NUM_PARALLEL=1"
     Environment="OLLAMA_MAX_LOADED_MODELS=1"
     Environment="OLLAMA_FLASH_ATTENTION=1"
     Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
     Then: sudo systemctl daemon-reload && sudo systemctl restart ollama

  3. Reduce num_ctx from 4096 to 2048 in config/config.yaml

  4. Force thread count — create Modelfile with:
     PARAMETER num_thread 4

  5. Pull and benchmark 1B models for 10+ tok/s:
     ollama pull gemma3:1b
     ollama pull qwen3:0.6b
     python scripts/bench_ollama.py gemma3:1b qwen3:0.6b qwen2.5:3b
""")


if __name__ == "__main__":
    main()
