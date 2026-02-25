#!/usr/bin/env python3
"""Benchmark all locally available Ollama models.

Measures time-to-first-token (TTFT), tokens/second, and total latency
for a set of prompts representative of a voice assistant workload.

Usage:
    python scripts/bench_ollama.py                  # bench all models
    python scripts/bench_ollama.py qwen2.5:3b       # bench specific model(s)
    python scripts/bench_ollama.py --rounds 3        # average over 3 rounds
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import ollama

# ---------------------------------------------------------------------------
# Test prompts — short, voice-assistant-style queries
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in two sentences.",
    "Write a short four-line poem about rain.",
]

SYSTEM_PROMPT = (
    "You are a helpful home voice assistant. "
    "Keep responses concise — 1-3 sentences."
)

NUM_CTX = 2048  # small context for consistent benchmarking


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    prompt: str
    ttft_s: float  # time to first token
    total_s: float  # wall-clock time
    tok_count: int  # generated tokens
    tok_per_s: float  # tokens / second


@dataclass
class ModelResult:
    model: str
    runs: list[RunResult] = field(default_factory=list)
    load_s: float = 0.0  # cold-load time

    @property
    def avg_ttft(self) -> float:
        return sum(r.ttft_s for r in self.runs) / len(self.runs) if self.runs else 0

    @property
    def avg_tok_s(self) -> float:
        return sum(r.tok_per_s for r in self.runs) / len(self.runs) if self.runs else 0

    @property
    def avg_total(self) -> float:
        return sum(r.total_s for r in self.runs) / len(self.runs) if self.runs else 0

    @property
    def total_tokens(self) -> int:
        return sum(r.tok_count for r in self.runs)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def list_models(client: ollama.Client) -> list[str]:
    """Return names of all locally available models."""
    resp = client.list()
    return sorted(m.model for m in resp.models)


def warm_model(client: ollama.Client, model: str) -> float:
    """Load model into memory and return load time in seconds."""
    t0 = time.perf_counter()
    client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1, "num_ctx": 32},
    )
    return time.perf_counter() - t0


def bench_prompt(client: ollama.Client, model: str, prompt: str) -> RunResult:
    """Stream a single prompt and measure performance."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    t_start = time.perf_counter()
    t_first: float | None = None
    tok_count = 0

    for chunk in client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": 0.7, "num_ctx": NUM_CTX},
    ):
        content = chunk.get("message", {}).get("content", "")
        if content and t_first is None:
            t_first = time.perf_counter()
        if chunk.get("done"):
            tok_count = chunk.get("eval_count", tok_count)

    t_end = time.perf_counter()
    total = t_end - t_start
    ttft = (t_first - t_start) if t_first else total

    return RunResult(
        prompt=prompt,
        ttft_s=ttft,
        total_s=total,
        tok_count=tok_count,
        tok_per_s=tok_count / total if total > 0 else 0,
    )


def bench_model(
    client: ollama.Client,
    model: str,
    prompts: list[str],
    rounds: int,
) -> ModelResult:
    """Run all prompts N rounds against one model."""
    result = ModelResult(model=model)

    # Warm up (cold load)
    print(f"  Loading model...", end=" ", flush=True)
    result.load_s = warm_model(client, model)
    print(f"{result.load_s:.1f}s")

    for r in range(rounds):
        if rounds > 1:
            print(f"  Round {r + 1}/{rounds}")
        for prompt in prompts:
            short = prompt[:50] + ("..." if len(prompt) > 50 else "")
            print(f"    \"{short}\"", end=" ", flush=True)
            run = bench_prompt(client, model, prompt)
            result.runs.append(run)
            print(
                f"-> {run.tok_count} tok, "
                f"TTFT {run.ttft_s:.2f}s, "
                f"{run.tok_per_s:.1f} tok/s, "
                f"total {run.total_s:.2f}s"
            )

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_summary(results: list[ModelResult]) -> None:
    """Print a ranked comparison table."""
    # Sort by avg tokens/sec descending
    ranked = sorted(results, key=lambda r: r.avg_tok_s, reverse=True)

    print("\n" + "=" * 78)
    print(f"{'MODEL':<30} {'LOAD':>6} {'TTFT':>7} {'TOK/S':>7} {'AVG_T':>7} {'TOKENS':>7}")
    print("-" * 78)
    for r in ranked:
        print(
            f"{r.model:<30} "
            f"{r.load_s:>5.1f}s "
            f"{r.avg_ttft:>6.2f}s "
            f"{r.avg_tok_s:>6.1f} "
            f"{r.avg_total:>6.2f}s "
            f"{r.total_tokens:>7}"
        )
    print("=" * 78)

    best = ranked[0]
    print(f"\nFastest: {best.model} at {best.avg_tok_s:.1f} tok/s (TTFT {best.avg_ttft:.2f}s)")


def save_json(results: list[ModelResult], path: str) -> None:
    """Save detailed results to JSON."""
    data = []
    for r in results:
        data.append({
            "model": r.model,
            "load_s": round(r.load_s, 3),
            "avg_ttft_s": round(r.avg_ttft, 3),
            "avg_tok_s": round(r.avg_tok_s, 1),
            "avg_total_s": round(r.avg_total, 3),
            "total_tokens": r.total_tokens,
            "runs": [
                {
                    "prompt": run.prompt,
                    "ttft_s": round(run.ttft_s, 3),
                    "total_s": round(run.total_s, 3),
                    "tok_count": run.tok_count,
                    "tok_per_s": round(run.tok_per_s, 1),
                }
                for run in r.runs
            ],
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Ollama models")
    parser.add_argument(
        "models", nargs="*",
        help="Model name(s) to test (default: all local models)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of rounds per prompt (default: 1)",
    )
    parser.add_argument(
        "--url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    client = ollama.Client(host=args.url)

    # Determine which models to bench
    if args.models:
        models = args.models
    else:
        models = list_models(client)
        if not models:
            print("No models found. Pull some models first: ollama pull <model>")
            sys.exit(1)

    print(f"Benchmarking {len(models)} model(s), {args.rounds} round(s), {len(PROMPTS)} prompts each\n")

    results: list[ModelResult] = []
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model}")
        try:
            result = bench_model(client, model, PROMPTS, args.rounds)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}\n")

    if results:
        print_summary(results)
        if args.output:
            save_json(results, args.output)


if __name__ == "__main__":
    main()
