#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark: Python 3.14 performance advantages for guardrail workloads.

Measures concrete improvements that Python 3.14 brings over 3.12:

1. **Free-threaded parallel CPU-bound rails** — On 3.14t (no-GIL), multiple
   CPU-bound guardrail checks (regex scanning, hashing, PII detection) run
   truly in parallel via ThreadPoolExecutor, achieving near-linear speedup.
   On standard builds, the GIL serialises them.

2. **Deferred annotation evaluation (PEP 649)** — Python 3.14 evaluates
   type annotations lazily, reducing import time for annotation-heavy
   modules like config.py and llmrails.py.

3. **Incremental GC** — Python 3.14's incremental garbage collector reduces
   pause times, improving tail latency (p99) under sustained load.

4. **Faster asyncio internals** — Python 3.13+ and 3.14 optimise the
   event loop, benefiting async rail orchestration.

Run all benchmarks:
    python -m benchmarks.bench_py314_advantages

Run specific section:
    python -m benchmarks.bench_py314_advantages --section threading
    python -m benchmarks.bench_py314_advantages --section import
    python -m benchmarks.bench_py314_advantages --section gc
    python -m benchmarks.bench_py314_advantages --section asyncio

Output JSON for CI:
    python -m benchmarks.bench_py314_advantages --output /tmp/py314-bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gc
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP = 3
ROUNDS = 10
SAMPLE_TEXT = (
    "My social security number is 123-45-6789 and my email is user@example.com. "
    "Can you help me write a cover letter for a software engineering position? "
    "My credit card number is 4111-1111-1111-1111 and phone is 555-123-4567."
)

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CREDIT_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_PATTERNS = [_SSN_RE, _EMAIL_RE, _PHONE_RE, _CREDIT_RE]


def cpu_rail_work(text: str, rounds: int = 1000) -> dict:
    """Simulate a CPU-bound guardrail check: regex + hashing."""
    found = False
    for pat in _PATTERNS:
        if pat.search(text):
            found = True

    tokens = text.split()
    h = text.encode("utf-8")
    for _ in range(rounds):
        h = hashlib.sha256(h).digest()

    return {"found_pii": found, "tokens": len(tokens), "hash": h.hex()[:16]}


def _python_info() -> dict:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    build = "free-threaded" if gil_disabled else "standard"
    return {"version": ver, "build": build, "gil_disabled": gil_disabled}


# ---------------------------------------------------------------------------
# Section 1: Free-threaded parallel CPU rails
# ---------------------------------------------------------------------------


def bench_threading(rail_counts=(1, 2, 4, 8), cpu_rounds: int = 2000) -> list[dict]:
    """Compare sequential vs thread-pool execution of CPU-bound rails."""
    results = []

    for n_rails in rail_counts:
        # --- Sequential ---
        seq_times = []
        for _ in range(WARMUP):
            for _ in range(n_rails):
                cpu_rail_work(SAMPLE_TEXT, cpu_rounds)

        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(n_rails):
                cpu_rail_work(SAMPLE_TEXT, cpu_rounds)
            seq_times.append(time.perf_counter() - t0)

        # --- Threaded (ThreadPoolExecutor) ---
        thr_times = []
        workers = min(n_rails, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fn = functools.partial(cpu_rail_work, SAMPLE_TEXT, cpu_rounds)

            for _ in range(WARMUP):
                list(pool.map(lambda _: fn(), range(n_rails)))

            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                list(pool.map(lambda _: fn(), range(n_rails)))
                thr_times.append(time.perf_counter() - t0)

        # --- Threaded via asyncio + run_in_executor ---
        async def _async_threaded():
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=workers)
            tasks = [
                loop.run_in_executor(executor, functools.partial(cpu_rail_work, SAMPLE_TEXT, cpu_rounds))
                for _ in range(n_rails)
            ]
            await asyncio.gather(*tasks)
            executor.shutdown(wait=False)

        async_times = []
        for _ in range(WARMUP):
            asyncio.run(_async_threaded())
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            asyncio.run(_async_threaded())
            async_times.append(time.perf_counter() - t0)

        seq_mean = statistics.mean(seq_times) * 1000
        thr_mean = statistics.mean(thr_times) * 1000
        async_mean = statistics.mean(async_times) * 1000
        speedup_thr = seq_mean / thr_mean if thr_mean > 0 else 0
        speedup_async = seq_mean / async_mean if async_mean > 0 else 0

        results.append({
            "test": "cpu_parallel",
            "rails": n_rails,
            "cpu_rounds": cpu_rounds,
            "sequential_ms": round(seq_mean, 2),
            "threadpool_ms": round(thr_mean, 2),
            "async_executor_ms": round(async_mean, 2),
            "speedup_threadpool": round(speedup_thr, 2),
            "speedup_async_executor": round(speedup_async, 2),
            "workers": workers,
        })

        print(
            f"  {n_rails} rails: seq={seq_mean:.1f}ms  "
            f"tpool={thr_mean:.1f}ms ({speedup_thr:.2f}x)  "
            f"async={async_mean:.1f}ms ({speedup_async:.2f}x)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 2: Import time (deferred annotations benefit)
# ---------------------------------------------------------------------------


def bench_import(modules=None, iterations: int = 5) -> list[dict]:
    """Measure import time for key modules via subprocess."""
    if modules is None:
        modules = [
            "nemoguardrails",
            "nemoguardrails.rails.llm.config",
            "nemoguardrails.rails.llm.llmrails",
            "nemoguardrails.rails.llm.dag_scheduler",
        ]

    results = []
    for mod in modules:
        times = []
        for _ in range(iterations):
            cmd = [sys.executable, "-c", f"import time; t=time.perf_counter(); import {mod}; print(time.perf_counter()-t)"]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
                times.append(float(out.strip()) * 1000)
            except Exception as e:
                print(f"  Warning: failed to import {mod}: {e}")
                break

        if times:
            mean_ms = statistics.mean(times)
            results.append({
                "test": "import_time",
                "module": mod,
                "mean_ms": round(mean_ms, 1),
                "min_ms": round(min(times), 1),
                "max_ms": round(max(times), 1),
                "stdev_ms": round(statistics.stdev(times), 1) if len(times) > 1 else 0,
            })
            print(f"  {mod}: {mean_ms:.1f}ms (min={min(times):.1f}, max={max(times):.1f})")

    return results


# ---------------------------------------------------------------------------
# Section 3: GC pause impact on tail latency
# ---------------------------------------------------------------------------


def bench_gc_pauses(iterations: int = 5000, cpu_rounds: int = 200) -> list[dict]:
    """Measure p99 latency with and without GC pressure.

    Python 3.14's incremental GC should reduce p99 spikes.
    """
    results = []

    for label, create_garbage in [("no_gc_pressure", False), ("gc_pressure", True)]:
        latencies = []
        gc.collect()

        for i in range(iterations):
            if create_garbage and i % 10 == 0:
                # Create reference cycles to trigger GC
                _garbage = []
                for _ in range(100):
                    a: dict[str, Any] = {}
                    b: dict[str, Any] = {"ref": a}
                    a["ref"] = b
                    _garbage.append(a)

            t0 = time.perf_counter_ns()
            cpu_rail_work(SAMPLE_TEXT, cpu_rounds)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]
        mean = statistics.mean(latencies)

        results.append({
            "test": "gc_tail_latency",
            "scenario": label,
            "iterations": iterations,
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(mean, 3),
            "p99_p50_ratio": round(p99 / p50, 2) if p50 > 0 else 0,
        })
        print(
            f"  {label}: p50={p50:.3f}ms  p95={p95:.3f}ms  "
            f"p99={p99:.3f}ms  ratio={p99/p50:.2f}x"
        )

    return results


# ---------------------------------------------------------------------------
# Section 4: Asyncio overhead
# ---------------------------------------------------------------------------


def bench_asyncio(task_counts=(10, 50, 100, 500), iterations: int = 100) -> list[dict]:
    """Measure asyncio.gather overhead for parallel rail dispatch."""
    results = []

    async def noop_rail():
        return {"ok": True}

    async def light_rail():
        await asyncio.sleep(0)
        return {"ok": True}

    for n_tasks in task_counts:
        # Noop tasks — pure gather overhead
        noop_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            await_result = asyncio.run(_gather_n(noop_rail, n_tasks))
            noop_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Light tasks — minimal await
        light_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            asyncio.run(_gather_n(light_rail, n_tasks))
            light_times.append((time.perf_counter_ns() - t0) / 1e6)

        noop_mean = statistics.mean(noop_times)
        light_mean = statistics.mean(light_times)

        results.append({
            "test": "asyncio_gather",
            "tasks": n_tasks,
            "noop_mean_ms": round(noop_mean, 3),
            "light_mean_ms": round(light_mean, 3),
            "overhead_per_task_us": round((noop_mean / n_tasks) * 1000, 2),
        })
        print(
            f"  {n_tasks} tasks: noop={noop_mean:.3f}ms  "
            f"light={light_mean:.3f}ms  "
            f"overhead={noop_mean/n_tasks*1000:.1f}us/task"
        )

    return results


async def _gather_n(coro_fn, n):
    tasks = [asyncio.create_task(coro_fn()) for _ in range(n)]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Python 3.14 performance advantages benchmark")
    parser.add_argument("--output", "-o", default=None, help="JSON output path")
    parser.add_argument(
        "--section", "-s",
        choices=["threading", "import", "gc", "asyncio", "all"],
        default="all",
        help="Which benchmark section to run",
    )
    args = parser.parse_args()

    info = _python_info()
    print(f"Python {info['version']} ({info['build']})")
    print(f"GIL disabled: {info['gil_disabled']}")
    print()

    all_results: dict[str, Any] = {"python": info, "sections": {}}

    if args.section in ("threading", "all"):
        print("=== Section 1: CPU-bound parallel rail execution ===")
        all_results["sections"]["threading"] = bench_threading()
        print()

    if args.section in ("import", "all"):
        print("=== Section 2: Import time (deferred annotations) ===")
        all_results["sections"]["import"] = bench_import()
        print()

    if args.section in ("gc", "all"):
        print("=== Section 3: GC tail latency ===")
        all_results["sections"]["gc"] = bench_gc_pauses()
        print()

    if args.section in ("asyncio", "all"):
        print("=== Section 4: Asyncio gather overhead ===")
        all_results["sections"]["asyncio"] = bench_asyncio()
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "threading" in all_results["sections"]:
        thr = all_results["sections"]["threading"]
        best = max(thr, key=lambda r: r["speedup_threadpool"])
        if info["gil_disabled"]:
            print(f"  Thread parallelism: {best['speedup_threadpool']:.2f}x speedup "
                  f"({best['rails']} rails, free-threaded)")
        else:
            print(f"  Thread parallelism: {best['speedup_threadpool']:.2f}x "
                  f"({best['rails']} rails, GIL limits parallelism)")

    if "gc" in all_results["sections"]:
        gc_results = all_results["sections"]["gc"]
        if len(gc_results) >= 2:
            ratio = gc_results[1]["p99_p50_ratio"]
            print(f"  GC p99/p50 ratio under pressure: {ratio:.2f}x")

    if "asyncio" in all_results["sections"]:
        aio = all_results["sections"]["asyncio"]
        if aio:
            print(f"  Asyncio overhead: {aio[0]['overhead_per_task_us']:.1f}us/task "
                  f"({aio[0]['tasks']} tasks)")

    print()

    blob = json.dumps(all_results, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(blob + "\n")
        print(f"Results written to {args.output}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
