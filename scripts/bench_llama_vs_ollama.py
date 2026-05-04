#!/usr/bin/env python3
"""Ollama vs llama.cpp head-to-head benchmark (RPi5, same model, same prompts).

Both servers are assumed to run on localhost.
Thinking is disabled for both:
  - Ollama:    think=False SDK kwarg + /no_think prompt suffix
  - llama.cpp: chat_template_kwargs={"enable_thinking": false} + thinking_budget_tokens=0
               (most reliable: start llama-server with --reasoning-budget 0)

Usage:
    uv run scripts/bench_vs.py
    uv run scripts/bench_vs.py --ollama-model gemma4:e2b --rounds 2
    uv run scripts/bench_vs.py --ollama-url http://localhost:11434 \\
                                --llama-url  http://localhost:8080
    uv run scripts/bench_vs.py -o results.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx
import ollama

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain what causes thunder in one sentence.",
    "Name three planets in our solar system.",
    "What's two plus two?",
    "Write one sentence about autumn leaves.",
]

SYSTEM_PROMPT = "You are a helpful home voice assistant. Keep responses concise — 1-3 sentences."
NUM_CTX = 2048
MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Run:
    prompt: str
    ttft_s: float
    total_s: float
    tokens: int
    tok_per_s: float


@dataclass
class BackendResult:
    label: str
    runs: list[Run] = field(default_factory=list)
    load_s: float = 0.0
    error: str = ""

    @property
    def avg_ttft(self) -> float:
        return sum(r.ttft_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def avg_tok_s(self) -> float:
        return sum(r.tok_per_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def avg_total(self) -> float:
        return sum(r.total_s for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.runs)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
def _ollama_chat(client: ollama.Client, **kwargs):
    try:
        return client.chat(**kwargs, think=False)
    except TypeError:
        return client.chat(**kwargs)


def ollama_warmup(client: ollama.Client, model: str) -> float:
    t0 = time.perf_counter()
    _ollama_chat(
        client,
        model=model,
        messages=[{"role": "user", "content": "hi /no_think"}],
        options={"num_predict": 1, "num_ctx": 32},
    )
    return time.perf_counter() - t0


def ollama_bench_prompt(client: ollama.Client, model: str, prompt: str) -> Run:
    t_start = time.perf_counter()
    t_first: float | None = None
    tokens = 0

    for chunk in _ollama_chat(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + " /no_think"},
        ],
        stream=True,
        options={"temperature": 0.0, "num_ctx": NUM_CTX},
    ):
        content = chunk.get("message", {}).get("content", "")
        if content and t_first is None:
            t_first = time.perf_counter()
        if chunk.get("done"):
            tokens = chunk.get("eval_count", tokens)

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else total
    return Run(prompt=prompt, ttft_s=ttft, total_s=total, tokens=tokens,
               tok_per_s=tokens / total if total > 0 else 0.0)


def bench_ollama(url: str, model: str, rounds: int) -> BackendResult:
    label = f"Ollama ({model})"
    result = BackendResult(label=label)
    client = ollama.Client(host=url)

    print(f"\n[Ollama] {url}  model={model}")
    print("  Warming up...", end=" ", flush=True)
    try:
        result.load_s = ollama_warmup(client, model)
        print(f"{result.load_s:.1f}s")
    except Exception as e:
        result.error = str(e)
        print(f"FAILED: {e}")
        return result

    for r in range(rounds):
        if rounds > 1:
            print(f"  Round {r + 1}/{rounds}")
        for prompt in PROMPTS:
            short = prompt[:48] + ("..." if len(prompt) > 48 else "")
            print(f"    \"{short}\"", end=" ", flush=True)
            try:
                run = ollama_bench_prompt(client, model, prompt)
                result.runs.append(run)
                print(f"TTFT {run.ttft_s:.2f}s  {run.tok_per_s:.1f} tok/s  "
                      f"{run.tokens} tok  total {run.total_s:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")

    # Unload model from RAM
    try:
        client.chat(model=model, messages=[{"role": "user", "content": ""}], keep_alive=0)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------
def llama_check(base_url: str) -> str | None:
    try:
        r = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        return models[0]["id"] if models else "<unknown>"
    except Exception:
        return None


def llama_bench_prompt(client: httpx.Client, base_url: str, prompt: str) -> Run:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_budget_tokens": 0,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: float | None = None
    completion_tokens = 0
    generated_tokens = 0

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
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                generated_tokens += 1

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first is not None else total
    tokens = completion_tokens if completion_tokens > 0 else generated_tokens

    return Run(prompt=prompt, ttft_s=ttft, total_s=total, tokens=tokens,
               tok_per_s=tokens / total if total > 0 else 0.0)


def bench_llama(url: str, rounds: int) -> BackendResult:
    print(f"\n[llama.cpp] {url}")
    model_name = llama_check(url)
    if model_name is None:
        result = BackendResult(label=f"llama.cpp @ {url}")
        result.error = f"Cannot reach server at {url}"
        print(f"  ERROR: {result.error}")
        return result

    label = f"llama.cpp ({model_name})"
    result = BackendResult(label=label)

    print(f"  Model: {model_name}  (no warmup needed — server keeps model loaded)")

    with httpx.Client() as client:
        for r in range(rounds):
            if rounds > 1:
                print(f"  Round {r + 1}/{rounds}")
            for prompt in PROMPTS:
                short = prompt[:48] + ("..." if len(prompt) > 48 else "")
                print(f"    \"{short}\"", end=" ", flush=True)
                try:
                    run = llama_bench_prompt(client, url, prompt)
                    result.runs.append(run)
                    print(f"TTFT {run.ttft_s:.2f}s  {run.tok_per_s:.1f} tok/s  "
                          f"{run.tokens} tok  total {run.total_s:.2f}s")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Output: side-by-side comparison table
# ---------------------------------------------------------------------------
def _w(label: str, max_len: int = 24) -> str:
    return (label[:max_len - 1] + "…") if len(label) > max_len else label


def print_table(ollama_r: BackendResult, llama_r: BackendResult) -> None:
    W = 82
    print("\n" + "=" * W)
    print(f"  {'METRIC':<22}  {'OLLAMA':>22}  {'LLAMA.CPP':>22}  {'WINNER':>8}")
    print("-" * W)

    def row(metric, oval, lval, unit="", higher_better=True):
        try:
            ov = float(oval)
            lv = float(lval)
            if higher_better:
                winner = "ollama" if ov > lv else ("llama.cpp" if lv > ov else "tie")
            else:
                winner = "ollama" if ov < lv else ("llama.cpp" if lv < ov else "tie")
        except (ValueError, TypeError):
            winner = "-"
            ov = lv = None

        os = f"{oval}{unit}" if unit else str(oval)
        ls = f"{lval}{unit}" if unit else str(lval)
        print(f"  {metric:<22}  {os:>22}  {ls:>22}  {winner:>8}")

    def fmt(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}"

    # Summary metrics
    if ollama_r.runs:
        row("Avg TTFT (s)",       fmt(ollama_r.avg_ttft), fmt(llama_r.avg_ttft) if llama_r.runs else "N/A",    higher_better=False)
        row("Avg tok/s",          fmt(ollama_r.avg_tok_s, 1), fmt(llama_r.avg_tok_s, 1) if llama_r.runs else "N/A", higher_better=True)
        row("Avg response (s)",   fmt(ollama_r.avg_total), fmt(llama_r.avg_total) if llama_r.runs else "N/A",  higher_better=False)
        row("Model load (s)",     fmt(ollama_r.load_s), "N/A (server-managed)",                                higher_better=False)
        row("Total tokens out",   str(ollama_r.total_tokens), str(llama_r.total_tokens) if llama_r.runs else "N/A", higher_better=False)

    print("=" * W)

    # Per-prompt breakdown
    if ollama_r.runs and llama_r.runs:
        print(f"\n  {'PROMPT':<40}  {'OLLAMA':>10}  {'LLAMA.CPP':>10}  {'WINNER':>9}")
        print(f"  {'(tok/s per prompt)':<40}  {'tok/s':>10}  {'tok/s':>10}  {'':>9}")
        print("  " + "-" * 74)
        paired = zip(ollama_r.runs[:len(PROMPTS)], llama_r.runs[:len(PROMPTS)])
        for o_run, l_run in paired:
            short = (o_run.prompt[:38] + "…") if len(o_run.prompt) > 39 else o_run.prompt
            winner = "ollama" if o_run.tok_per_s > l_run.tok_per_s else (
                "llama.cpp" if l_run.tok_per_s > o_run.tok_per_s else "tie"
            )
            print(f"  {short:<40}  {o_run.tok_per_s:>10.1f}  {l_run.tok_per_s:>10.1f}  {winner:>9}")

    # Overall winner
    if ollama_r.runs and llama_r.runs:
        print()
        if ollama_r.avg_tok_s > llama_r.avg_tok_s:
            diff = (ollama_r.avg_tok_s / llama_r.avg_tok_s - 1) * 100
            print(f"  Overall: Ollama is {diff:.0f}% faster in throughput  "
                  f"({ollama_r.avg_tok_s:.1f} vs {llama_r.avg_tok_s:.1f} tok/s)")
        elif llama_r.avg_tok_s > ollama_r.avg_tok_s:
            diff = (llama_r.avg_tok_s / ollama_r.avg_tok_s - 1) * 100
            print(f"  Overall: llama.cpp is {diff:.0f}% faster in throughput  "
                  f"({llama_r.avg_tok_s:.1f} vs {ollama_r.avg_tok_s:.1f} tok/s)")
        else:
            print("  Overall: tie")

    print("=" * W)


def save_json(ollama_r: BackendResult, llama_r: BackendResult, path: str) -> None:
    def serialize(r: BackendResult) -> dict:
        return {
            "label": r.label,
            "load_s": round(r.load_s, 3),
            "avg_ttft_s": round(r.avg_ttft, 3),
            "avg_tok_s": round(r.avg_tok_s, 1),
            "avg_total_s": round(r.avg_total, 3),
            "total_tokens": r.total_tokens,
            "error": r.error,
            "runs": [
                {
                    "prompt": run.prompt,
                    "ttft_s": round(run.ttft_s, 3),
                    "total_s": round(run.total_s, 3),
                    "tokens": run.tokens,
                    "tok_per_s": round(run.tok_per_s, 1),
                }
                for run in r.runs
            ],
        }
    with open(path, "w") as f:
        json.dump({"ollama": serialize(ollama_r), "llama_cpp": serialize(llama_r)}, f, indent=2)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head: Ollama vs llama.cpp on RPi5 (same prompts, no thinking)"
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--ollama-model", default="gemma4:e2b",
                        help="Ollama model name (default: gemma4:e2b)")
    parser.add_argument("--llama-url", default="http://localhost:8080",
                        help="llama.cpp server URL (default: http://localhost:8080)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Rounds per prompt — results are averaged (default: 1)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("  Ollama vs llama.cpp — RPi5 head-to-head benchmark")
    print("=" * 60)
    print(f"  Prompts : {len(PROMPTS)}   Rounds: {args.rounds}")
    print(f"  Ollama  : {args.ollama_url}  model={args.ollama_model}")
    print(f"  llama.cpp: {args.llama_url}")
    print(f"  Thinking: DISABLED for both")
    print(f"  Note: for llama.cpp start server with --reasoning-budget 0")
    print("=" * 60)

    ollama_result = bench_ollama(args.ollama_url, args.ollama_model, args.rounds)
    llama_result = bench_llama(args.llama_url, args.rounds)

    if not ollama_result.runs and not llama_result.runs:
        print("\nBoth backends failed — nothing to compare.")
        sys.exit(1)

    print_table(ollama_result, llama_result)

    if args.output:
        save_json(ollama_result, llama_result, args.output)


if __name__ == "__main__":
    main()
