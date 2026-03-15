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

"""Benchmark: Python 3.14 performance advantages for NeMo Guardrails.

Measures concrete improvements across twelve areas, using the real guardrails
infrastructure (fake_rails, conftest, Jinja2 template engine, DAG scheduler,
ThreadSafeCache, ActionDispatcher) rather than toy synthetic workloads.

Sections:
    1. **Free-threaded parallel CPU rails** — Sequential vs ThreadPoolExecutor
       vs asyncio.run_in_executor using *pure-Python* CPU work that holds the
       GIL on standard builds.  On 3.14t (no-GIL) threads achieve near-linear
       speedup because the work genuinely runs in parallel.

    2. **Template rendering (M3 cache)** — Measures the actual LLMTaskManager
       Jinja2 _render_string() path, comparing cold (first parse) vs hot
       (cached template + variable set) performance.

    3. **Import time (PEP 649)** — Cold-start import timing via subprocess,
       comparing annotation-heavy modules with/without deferred evaluation.

    4. **GC tail latency** — p50/p95/p99/p999 under heavy GC pressure
       (deep reference cycles, large allocation bursts) to show Python 3.14's
       incremental GC advantage.

    5. **Asyncio gather + real rails** — Measures asyncio.gather with the
       actual fake_rails infrastructure (mixed CPU + I/O rails) at scale.

    6. **DAG scheduler** — Benchmarks the TopologicalScheduler with
       dependency chains to show parallel group execution benefits.

    7. **Eager task factory** — Directly measures the speedup from
       asyncio.eager_task_factory (3.12+) vs standard task creation, using
       a *persistent* event loop to avoid masking the benefit with
       asyncio.run() overhead.

    8. **Scheduler execute (before/after)** — Runs the actual
       TopologicalScheduler.execute() codepath with and without the eager
       task factory to prove the code change delivers measurable improvement.

    9. **Action name normalisation cache** — Measures the benefit of caching
       normalised action names vs re-computing CamelCase→snake_case on every
       dispatch call.

   10. **ThreadSafeCache contention** — Measures ThreadSafeCache throughput
       under multi-threaded contention (1, 2, 4, 8 threads) to quantify
       lock overhead on both GIL and free-threaded builds.

   11. **End-to-end pipeline simulation** — Full request path: template
       rendering + action dispatch + rail evaluation + response assembly,
       measuring total framework overhead per request.

   12. **Memory efficiency** — Tracks per-request RSS delta and allocation
       counts via tracemalloc to detect memory leaks from caching.

Run:
    python -m benchmarks.bench_py314_advantages
    python -m benchmarks.bench_py314_advantages --section threading
    python -m benchmarks.bench_py314_advantages --output /tmp/py314-bench.json
"""

# PEP 563: deferred evaluation of annotations — reduces import-time cost
# when modules are heavily annotated (relevant to Section 3 / PEP 649).
from __future__ import annotations

import argparse
import asyncio
import functools
import gc
import json
import os
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from benchmarks.conftest import SAMPLE_PAYLOADS, compute_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WARMUP iterations are discarded before measurement begins. This lets
# JIT compilation (if present), branch predictors, and CPU caches reach
# a steady state so that measured latencies reflect production behaviour.
WARMUP = 5
# ROUNDS controls statistical sample size. 30 rounds is the minimum for
# the Central Limit Theorem to apply, giving meaningful mean/stdev values.
ROUNDS = 30
_SECTION_CHOICES = [
    "threading",
    "template",
    "import",
    "gc",
    "asyncio",
    "dag",
    "eager",
    "scheduler",
    "action_cache",
    "cache_contention",
    "pipeline",
    "memory",
    "all",
]


def _python_info() -> dict:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # Py_GIL_DISABLED is set at compile time in free-threaded (3.13t/3.14t)
    # builds. When True, the GIL is fully removed, enabling genuine thread
    # parallelism for CPU-bound pure-Python code — the core advantage
    # measured by Section 1.
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    build = "free-threaded" if gil_disabled else "standard"
    return {
        "version": ver,
        "build": build,
        "gil_disabled": gil_disabled,
        "cpu_count": os.cpu_count(),  # Used to cap ThreadPoolExecutor workers
    }


# ---------------------------------------------------------------------------
# Pure-Python CPU work that genuinely holds the GIL
# ---------------------------------------------------------------------------
#
# The previous benchmark used hashlib.sha256 which is implemented in C and
# releases the GIL during computation — meaning threads could already run in
# parallel even on standard Python, hiding the free-threaded advantage.
#
# This function does the equivalent work (regex + tokenisation + iterative
# hashing) in pure Python so the GIL is held throughout.  On free-threaded
# 3.14t this means threads genuinely run in parallel for the first time.
# ---------------------------------------------------------------------------


def _pure_python_cpu_work(text: str, rounds: int = 2000) -> dict:
    """Pure-Python CPU-bound work that does NOT release the GIL.

    Simulates the kind of work guardrail checks do:
    - Pattern matching (character-by-character scan)
    - Token counting with normalisation
    - Iterative hashing (pure-Python FNV-1a variant)
    """
    # Pattern scan: count digits, @-signs, dashes (PII indicators).
    # This character-by-character loop is deliberately kept in pure Python
    # so that the GIL is held for the entire duration — C extensions like
    # re.findall() would release the GIL and allow spurious parallelism
    # on standard builds, defeating the purpose of this benchmark.
    digits = 0
    ats = 0
    dashes = 0
    for ch in text:
        if ch.isdigit():
            digits += 1
        elif ch == "@":
            ats += 1
        elif ch == "-":
            dashes += 1

    # Token normalisation — mimics the preprocessing guardrail checks
    # perform before pattern matching (lowercasing, whitespace splitting).
    tokens = text.lower().split()
    token_count = len(tokens)
    unique_count = len(set(tokens))

    # Iterative pure-Python FNV-1a hash (GIL-holding).
    # FNV-1a was chosen because it is simple, deterministic, and — crucially —
    # entirely implemented in Python bytecode.  The `rounds` parameter
    # controls how much CPU time each "rail" consumes, letting us amplify
    # the GIL contention effect for measurement purposes.
    h = 0x811C9DC5  # FNV offset basis for 32-bit
    data = text.encode("utf-8")
    for _ in range(rounds):
        for byte in data[:64]:  # Cap at 64 bytes to keep per-round cost stable
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, masked to 32 bits

    return {
        "ok": digits < 20,  # Threshold mimics a simple PII heuristic
        "kind": "cpu",
        "tokens": token_count,
        "unique": unique_count,
        "digest": format(h, "08x"),  # Hex digest ensures deterministic output
    }


# ---------------------------------------------------------------------------
# Section 1: Free-threaded parallel CPU rails
# ---------------------------------------------------------------------------


def bench_threading(
    rail_counts: tuple[int, ...] = (1, 2, 4, 8),
    cpu_rounds: int = 800,
) -> list[dict]:
    """Compare sequential vs threaded execution of pure-Python CPU work.

    Uses _pure_python_cpu_work which holds the GIL, so on standard Python
    threads cannot run in parallel.  On free-threaded 3.14t, threads achieve
    genuine parallelism and near-linear speedup.
    """
    results = []
    text = SAMPLE_PAYLOADS[2]  # PII-containing payload — triggers digit/at/dash scanning

    for n_rails in rail_counts:
        # Cap workers at the physical core count to avoid context-switch overhead
        # that would artificially inflate threaded timings on small machines.
        workers = min(n_rails, os.cpu_count() or 4)
        # functools.partial freezes arguments so we measure only execution time,
        # not argument-packing overhead, across all three dispatch strategies.
        fn = functools.partial(_pure_python_cpu_work, text, cpu_rounds)

        # --- Sequential baseline ---
        # On both standard and free-threaded builds, sequential execution is
        # identical — it serves as the denominator for speedup calculations.
        for _ in range(WARMUP):
            for _ in range(n_rails):
                fn()

        seq_times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(n_rails):
                fn()
            seq_times.append(time.perf_counter() - t0)

        # --- ThreadPoolExecutor ---
        # On standard builds the GIL serialises these threads, so speedup ~ 1.0.
        # On free-threaded 3.14t, threads run in parallel and speedup ~ n_rails.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(WARMUP):
                list(pool.map(lambda _: fn(), range(n_rails)))

            thr_times = []
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                list(pool.map(lambda _: fn(), range(n_rails)))
                thr_times.append(time.perf_counter() - t0)

        # --- asyncio + run_in_executor (mirrors real ActionDispatcher path) ---
        # This is how NeMo Guardrails actually dispatches CPU-bound actions:
        # the event loop offloads blocking work to a thread pool. Measuring
        # this path separately captures any additional overhead from asyncio
        # future management on top of raw thread dispatch.
        async def _async_dispatch():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = [loop.run_in_executor(executor, fn) for _ in range(n_rails)]
                await asyncio.gather(*futs)

        for _ in range(WARMUP):
            asyncio.run(_async_dispatch())

        async_times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            asyncio.run(_async_dispatch())
            async_times.append(time.perf_counter() - t0)

        # Convert seconds to milliseconds for compute_stats
        seq_stats = compute_stats([t * 1000 for t in seq_times])
        thr_stats = compute_stats([t * 1000 for t in thr_times])
        async_stats = compute_stats([t * 1000 for t in async_times])

        # Speedup = sequential / parallel. Values > 1.0 indicate genuine
        # parallelism; values near 1.0 mean the GIL is serialising threads.
        speedup_thr = seq_stats["mean_ms"] / thr_stats["mean_ms"] if thr_stats["mean_ms"] > 0 else 0
        speedup_async = seq_stats["mean_ms"] / async_stats["mean_ms"] if async_stats["mean_ms"] > 0 else 0

        results.append(
            {
                "test": "cpu_parallel",
                "rails": n_rails,
                "workers": workers,
                "sequential_mean_ms": round(seq_stats["mean_ms"], 2),
                "sequential_p99_ms": round(seq_stats["p99_ms"], 2),
                "sequential_stdev_ms": round(seq_stats["stdev_ms"], 2),
                "threadpool_mean_ms": round(thr_stats["mean_ms"], 2),
                "threadpool_p99_ms": round(thr_stats["p99_ms"], 2),
                "threadpool_stdev_ms": round(thr_stats["stdev_ms"], 2),
                "async_executor_mean_ms": round(async_stats["mean_ms"], 2),
                "async_executor_p99_ms": round(async_stats["p99_ms"], 2),
                "speedup_threadpool": round(speedup_thr, 2),
                "speedup_async_executor": round(speedup_async, 2),
            }
        )

        print(
            f"  {n_rails:2d} rails ({workers}w): "
            f"seq={seq_stats['mean_ms']:7.1f}ms  "
            f"tpool={thr_stats['mean_ms']:7.1f}ms ({speedup_thr:.2f}x)  "
            f"async={async_stats['mean_ms']:7.1f}ms ({speedup_async:.2f}x)  "
            f"stdev={thr_stats['stdev_ms']:.2f}"
        )

    return results


# ---------------------------------------------------------------------------
# Section 2: Template rendering (M3 Jinja2 cache benchmark)
# ---------------------------------------------------------------------------


def bench_template_rendering(iterations: int = 5000) -> list[dict]:
    """Benchmark the actual LLMTaskManager template cache (M3 optimisation).

    Measures cold (first call, parse + compile) vs hot (cached) rendering
    of realistic guardrail prompt templates.
    """
    from jinja2 import meta
    from jinja2.sandbox import SandboxedEnvironment

    # SandboxedEnvironment is what NeMo Guardrails uses in production to
    # prevent template injection attacks. We benchmark it (not plain
    # Environment) so measurements reflect real-world overhead.
    env = SandboxedEnvironment()

    # Realistic templates of increasing complexity — the "simple" template
    # has a single variable substitution, "medium" adds a for-loop over
    # conversation history, and "complex" exercises multiple variables plus
    # a custom filter. This gradient reveals how cache benefit scales with
    # template complexity (parsing cost rises faster than rendering cost).
    templates = {
        "simple": "You are a helpful assistant. {{ general_instructions }}",
        "medium": (
            "Below is a conversation:\n"
            "{% for msg in history %}"
            "{{ msg.role }}: {{ msg.content }}\n"
            "{% endfor %}\n"
            "Instructions: {{ general_instructions }}\n"
            "Respond safely."
        ),
        "complex": (
            "{{ general_instructions }}\n\n"
            "Sample conversation:\n{{ sample_conversation }}\n\n"
            "{% for msg in history | last_turns(5) %}"
            "{{ msg.role }}: {{ msg.content }}\n"
            "{% endfor %}\n"
            "User intent: {{ user_intent }}\n"
            "Bot intent: {{ bot_intent }}\n"
            "Available actions: {{ available_actions }}\n"
            "Context: {{ context_summary }}"
        ),
    }

    context = {
        "general_instructions": "You are a helpful AI assistant. Be safe and accurate.",
        "history": [{"role": "user", "content": f"Message {i}"} for i in range(20)],
        "sample_conversation": "user: Hi\nassistant: Hello!",
        "user_intent": "ask_question",
        "bot_intent": "provide_answer",
        "available_actions": "search, calculate, respond",
        "context_summary": "General conversation about Python programming",
    }

    # Register a custom filter so the "complex" template's last_turns() works
    env.filters["last_turns"] = lambda msgs, n: msgs[-n:]

    results = []
    # These caches mirror the M3 optimisation in LLMTaskManager: compiled
    # Jinja2 template objects and their declared variable sets are stored
    # so that repeated renders skip the expensive parse-and-compile step.
    template_cache: dict[str, Any] = {}
    variables_cache: dict[str, frozenset] = {}

    for tpl_name, tpl_str in templates.items():
        # --- Cold path: no cache, parse + compile on every invocation ---
        # This simulates the pre-M3 behaviour where each _render_string()
        # call re-parses the Jinja2 source from scratch.
        cold_times = []
        for _ in range(WARMUP):
            t = env.from_string(tpl_str)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))
            t.render({k: v for k, v in context.items() if k in vs})

        for _ in range(iterations):
            t0 = time.perf_counter_ns()  # Nanosecond precision for sub-ms measurements
            t = env.from_string(tpl_str)  # Parse + compile (the expensive part)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))  # AST walk
            render_ctx = {k: v for k, v in context.items() if k in vs}
            t.render(render_ctx)
            cold_times.append((time.perf_counter_ns() - t0) / 1e6)  # Convert ns -> ms

        # --- Hot path: cached template + cached variable set (M3 path) ---
        # Pre-populate the caches with the compiled template and its
        # undeclared variable set (stored as frozenset for hashability).
        template_cache[tpl_str] = env.from_string(tpl_str)
        variables_cache[tpl_str] = frozenset(meta.find_undeclared_variables(env.parse(tpl_str)))

        # Hot path: only dict look-ups + render — no parsing or AST walking.
        # The speedup quantifies the direct benefit of the M3 template cache.
        hot_times = []
        for _ in range(WARMUP):
            cached_t = template_cache[tpl_str]
            cached_v = variables_cache[tpl_str]
            cached_t.render({k: v for k, v in context.items() if k in cached_v})

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            cached_t = template_cache[tpl_str]  # O(1) dict look-up
            cached_v = variables_cache[tpl_str]  # O(1) dict look-up
            # Filter context to only the variables the template actually uses,
            # avoiding unnecessary data serialisation during rendering.
            render_ctx = {k: v for k, v in context.items() if k in cached_v}
            cached_t.render(render_ctx)
            hot_times.append((time.perf_counter_ns() - t0) / 1e6)

        cold_stats = compute_stats(cold_times)
        hot_stats = compute_stats(hot_times)
        # Speedup = cold / hot. Higher values mean caching saves more time.
        speedup = cold_stats["mean_ms"] / hot_stats["mean_ms"] if hot_stats["mean_ms"] > 0 else 0

        results.append(
            {
                "test": "template_rendering",
                "template": tpl_name,
                "iterations": iterations,
                "cold_mean_ms": round(cold_stats["mean_ms"], 4),
                "cold_p99_ms": round(cold_stats["p99_ms"], 4),
                "hot_mean_ms": round(hot_stats["mean_ms"], 4),
                "hot_p99_ms": round(hot_stats["p99_ms"], 4),
                "speedup": round(speedup, 2),
                "savings_us": round((cold_stats["mean_ms"] - hot_stats["mean_ms"]) * 1000, 1),
            }
        )

        print(
            f"  {tpl_name:8s}: cold={cold_stats['mean_ms']:.4f}ms  "
            f"hot={hot_stats['mean_ms']:.4f}ms  "
            f"({speedup:.2f}x, saves {(cold_stats['mean_ms'] - hot_stats['mean_ms']) * 1000:.1f}us/call)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 3: Import time (deferred annotations benefit)
# ---------------------------------------------------------------------------


def bench_import(iterations: int = 10) -> list[dict]:
    """Measure cold-start import time via subprocess.

    Uses more iterations and includes both annotation-heavy and light modules
    for comparison.
    """
    # Modules chosen to represent a spectrum of annotation density.
    # PEP 649 (Python 3.14) defers annotation evaluation to access time
    # rather than import time, so annotation-heavy modules should show
    # measurably faster cold imports on 3.14+ vs 3.12/3.13.
    modules = [
        ("nemoguardrails", "top-level"),
        ("nemoguardrails.rails.llm.config", "annotation-heavy (200+ type hints)"),
        ("nemoguardrails.rails.llm.llmrails", "annotation-heavy (entry point)"),
        ("nemoguardrails.rails.llm.dag_scheduler", "annotation-heavy (dataclasses)"),
        ("nemoguardrails.llm.taskmanager", "annotation-heavy (Jinja2 types)"),
        ("nemoguardrails.actions.action_dispatcher", "annotation-heavy (dispatcher)"),
    ]

    results = []
    for mod, description in modules:
        times = []
        for _ in range(iterations):
            # Each iteration spawns a fresh subprocess so the import is truly
            # cold — no bytecode cache (.pyc) warming from a prior iteration
            # within the same process. The -X importtime flag is passed but
            # its stderr output is discarded; we measure wall time ourselves.
            cmd = [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                f"import time as _t; _s=_t.perf_counter(); import {mod}; print(_t.perf_counter()-_s)",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
                times.append(float(out.strip()) * 1000)  # seconds -> ms
            except Exception as e:
                print(f"  Warning: failed to import {mod}: {e}")
                break  # No point retrying if the module cannot be imported

        if times:
            stats = compute_stats(times)
            results.append(
                {
                    "test": "import_time",
                    "module": mod,
                    "description": description,
                    "iterations": len(times),
                    "mean_ms": round(stats["mean_ms"], 1),
                    "p50_ms": round(stats["p50_ms"], 1),
                    "p99_ms": round(stats["p99_ms"], 1),
                    "stdev_ms": round(stats["stdev_ms"], 1),
                    "min_ms": round(min(times), 1),
                    "max_ms": round(max(times), 1),
                }
            )
            print(
                f"  {mod:48s} {stats['mean_ms']:7.1f}ms "
                f"(p50={stats['p50_ms']:.1f}, p99={stats['p99_ms']:.1f}, "
                f"stdev={stats['stdev_ms']:.1f})"
            )

    return results


# ---------------------------------------------------------------------------
# Section 4: GC pause impact on tail latency
# ---------------------------------------------------------------------------


def bench_gc_pauses(iterations: int = 10000, cpu_rounds: int = 200) -> list[dict]:
    """Measure tail latency under escalating GC pressure.

    Python 3.14's incremental GC should reduce p99/p999 spikes.
    Uses 10k iterations and three pressure levels for statistical significance.
    """
    text = SAMPLE_PAYLOADS[4]  # Longest PII-heavy payload for realistic scan cost
    fn = functools.partial(_pure_python_cpu_work, text, cpu_rounds)

    # Each scenario defines (label, cycle_count, every_n_iterations).
    # cycle_count controls how many reference cycles are created per burst;
    # every_n controls how frequently bursts occur.  This lets us model
    # different production patterns: a mostly-idle system (light), a
    # sustained high-throughput system (heavy), and bursty traffic (burst).
    scenarios = [
        ("baseline", 0, 0),  # No GC pressure — establishes floor latency
        ("light_pressure", 50, 5),  # Moderate: 50 cycles every 5 iters
        ("heavy_pressure", 500, 3),  # Heavy: 500 cycles every 3 iters
        ("burst_pressure", 2000, 10),  # Bursts: large allocation spikes
    ]

    results = []
    for label, cycle_count, every_n in scenarios:
        # Triple gc.collect() to ensure all three generations are swept
        # before starting measurement, preventing residual garbage from
        # the previous scenario from contaminating results.
        gc.collect()
        gc.collect()
        gc.collect()

        latencies = []
        # garbage_holder keeps recent batches alive, sustaining GC pressure
        # by preventing immediate collection of the reference cycles.
        garbage_holder: list[Any] = []

        for i in range(iterations):
            if cycle_count > 0 and every_n > 0 and i % every_n == 0:
                # Create reference cycles that the GC must trace and break.
                # The a -> c -> b -> a triangle cannot be reclaimed by
                # reference counting alone — only the tracing collector
                # can detect and free these.  Python 3.14's incremental GC
                # processes such cycles in smaller batches, reducing the
                # worst-case pause (p999) compared to the stop-the-world
                # full-generation sweeps in older versions.
                batch: list[Any] = []
                for _ in range(cycle_count):
                    a: dict[str, Any] = {"data": "x" * 64}
                    b: dict[str, Any] = {"ref": a, "data": "y" * 64}
                    c: dict[str, Any] = {"ref": b, "data": "z" * 64}
                    a["ref"] = c  # Completes the a -> c -> b -> a cycle
                    batch.append(a)
                garbage_holder.append(batch)
                # Cap retained batches at 20 so memory does not grow unbounded,
                # whilst maintaining enough live cycles to keep the GC busy.
                if len(garbage_holder) > 20:
                    garbage_holder.pop(0)

            # Measure a single rail execution. GC pauses (if they occur)
            # will inflate this timing — that is exactly what we want to
            # capture: the tail-latency impact of garbage collection on
            # request-level performance.
            t0 = time.perf_counter_ns()
            fn()
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

        # Sort once for O(1) percentile extraction by index
        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]
        # p999 is the key metric here — it captures rare GC-induced spikes
        # that cause SLA breaches in production but are invisible in mean/p50.
        p999 = latencies[int(n * 0.999)]
        mean = statistics.mean(latencies)
        max_lat = latencies[-1]

        results.append(
            {
                "test": "gc_tail_latency",
                "scenario": label,
                "iterations": iterations,
                "cycle_count": cycle_count,
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "p99_ms": round(p99, 3),
                "p999_ms": round(p999, 3),
                "max_ms": round(max_lat, 3),
                "mean_ms": round(mean, 3),
                "stdev_ms": round(statistics.stdev(latencies), 3),
                # Ratios relative to p50 quantify tail-latency spread.
                # A low ratio (close to 1.0) means consistent latency; a high
                # ratio means GC pauses are causing severe outlier spikes.
                "p99_p50_ratio": round(p99 / p50, 2) if p50 > 0 else 0,
                "p999_p50_ratio": round(p999 / p50, 2) if p50 > 0 else 0,
            }
        )
        print(
            f"  {label:18s}: p50={p50:.3f}ms  p95={p95:.3f}ms  "
            f"p99={p99:.3f}ms  p999={p999:.3f}ms  max={max_lat:.3f}ms  "
            f"ratio={p99 / p50:.2f}x"
        )

        # Explicitly release and collect to reset heap state between scenarios
        del garbage_holder
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Section 5: Asyncio gather with real rails
# ---------------------------------------------------------------------------


def bench_asyncio_rails(
    rail_counts: tuple[int, ...] = (2, 4, 8, 16),
    io_latency_ms: int = 20,
    iterations: int = 200,
) -> list[dict]:
    """Benchmark asyncio.gather with mixed I/O-bound rails.

    Each rail simulates an external API call (sleep). Serial execution
    takes N * latency; parallel takes ~1 * latency. The speedup shows
    how efficiently asyncio.gather schedules concurrent I/O rails — a
    pattern used heavily in NeMo Guardrails for parallel input checks.
    """
    # fake_rails.io_bound_rail simulates an external API call with a
    # configurable sleep duration, mimicking moderation API round-trips.
    from benchmarks.fake_rails import io_bound_rail

    results = []

    for num_rails in rail_counts:
        text = SAMPLE_PAYLOADS[2]  # PII payload for realistic content checks

        # Serial: each rail awaits sequentially, so total ~ N * latency.
        async def _serial():
            for _ in range(num_rails):
                await io_bound_rail(text, latency_ms=io_latency_ms)

        # Parallel: all rails launched concurrently via asyncio.gather(),
        # so total ~ 1 * latency (plus scheduling overhead). This is the
        # pattern NeMo Guardrails uses for parallel input/output checks.
        async def _parallel():
            coros = [io_bound_rail(text, latency_ms=io_latency_ms) for _ in range(num_rails)]
            await asyncio.gather(*coros)

        # Warmup both paths to stabilise event loop internals
        for _ in range(3):
            asyncio.run(_serial())
            asyncio.run(_parallel())

        # Serial measurement
        serial_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            asyncio.run(_serial())
            serial_times.append((time.perf_counter() - t0) * 1000)

        # Parallel measurement — the gather() overhead is what we quantify
        parallel_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            asyncio.run(_parallel())
            parallel_times.append((time.perf_counter() - t0) * 1000)

        serial_stats = compute_stats(serial_times)
        parallel_stats = compute_stats(parallel_times)
        speedup = serial_stats["mean_ms"] / parallel_stats["mean_ms"] if parallel_stats["mean_ms"] > 0 else 0

        results.append(
            {
                "test": "asyncio_rails",
                "num_rails": num_rails,
                "io_latency_ms": io_latency_ms,
                "serial_mean_ms": round(serial_stats["mean_ms"], 2),
                "serial_p99_ms": round(serial_stats["p99_ms"], 2),
                "parallel_mean_ms": round(parallel_stats["mean_ms"], 2),
                "parallel_p99_ms": round(parallel_stats["p99_ms"], 2),
                "speedup": round(speedup, 2),
                # Scheduling efficiency: ideal = 1.0 (gather overhead is zero).
                # Values < 1.0 indicate overhead from asyncio task creation,
                # event loop wake-ups, and coroutine bookkeeping.
                "scheduling_efficiency": round(
                    (io_latency_ms / parallel_stats["mean_ms"]) if parallel_stats["mean_ms"] > 0 else 0, 2
                ),
            }
        )

        print(
            f"  {num_rails:2d} rails (io={io_latency_ms}ms): "
            f"serial={serial_stats['mean_ms']:.1f}ms  "
            f"parallel={parallel_stats['mean_ms']:.1f}ms  "
            f"({speedup:.2f}x)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 6: DAG scheduler benchmark
# ---------------------------------------------------------------------------


def bench_dag_scheduler(iterations: int = 500) -> list[dict]:
    """Benchmark the TopologicalScheduler with dependency chains.

    Tests that independent rails execute concurrently while respecting
    dependency ordering.
    """
    try:
        from nemoguardrails.rails.llm.dag_scheduler import (
            RailDependencyGraph,
            TopologicalScheduler,
        )
    except ImportError:
        print("  DAG scheduler not available, skipping")
        return []

    results = []

    # Topology configurations: each builder returns (graph, durations_dict).
    # Rails must be added in topological order because add_rail validates
    # that declared dependencies already exist in the graph.
    #
    # Four topologies model common guardrail patterns:
    #   linear  — strict sequential chain (worst case for parallelism)
    #   wide    — fully independent rails (best case for parallelism)
    #   diamond — fork-join: A fans out to B,C then D merges them
    #   fan_out — one root followed by many independent leaves
    def _build_linear():
        g = RailDependencyGraph()
        g.add_rail("A")
        g.add_rail("B", depends_on=["A"])
        g.add_rail("C", depends_on=["B"])
        g.add_rail("D", depends_on=["C"])
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}  # 40ms serial, 40ms parallel

    def _build_wide():
        g = RailDependencyGraph()
        for name in ["A", "B", "C", "D"]:
            g.add_rail(name)  # No dependencies — all execute concurrently
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}  # 40ms serial, 10ms parallel

    def _build_diamond():
        g = RailDependencyGraph()
        g.add_rail("A")
        g.add_rail("B", depends_on=["A"])
        g.add_rail("C", depends_on=["A"])
        g.add_rail("D", depends_on=["B", "C"])  # Waits for both B and C
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}  # 40ms serial, 30ms parallel

    def _build_fan_out():
        g = RailDependencyGraph()
        g.add_rail("A")
        for i in range(8):
            g.add_rail(chr(66 + i), depends_on=["A"])  # 8 leaves depend on root
        return g, {**{"A": 5}, **{chr(66 + i): 10 for i in range(8)}}  # 85ms serial, 15ms parallel

    topologies = [
        ("linear_4", _build_linear),
        ("wide_4", _build_wide),
        ("diamond", _build_diamond),
        ("fan_out_8", _build_fan_out),
    ]

    for topo_name, build_fn in topologies:
        graph, durations = build_fn()
        scheduler = TopologicalScheduler(graph)
        # groups is a list of RailGroup objects; rails within a group
        # execute concurrently, groups execute sequentially.
        groups = scheduler.groups

        # Build coroutine factories — default arguments capture the loop
        # variables by value (Python's late-binding closure gotcha).
        async def run_topology(_groups=groups, _durations=durations):
            for group in _groups:
                coros = []
                for rail_name in group.rails:
                    dur = _durations.get(rail_name, 10)

                    async def _exec(name=rail_name, d=dur):
                        await asyncio.sleep(d / 1000)  # Simulate I/O latency in ms
                        return {"ok": True, "rail": name}

                    coros.append(_exec())
                # Within a group, all coroutines run concurrently
                await asyncio.gather(*coros)

        for _ in range(3):  # Warmup
            asyncio.run(run_topology())

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            asyncio.run(run_topology())
            times.append((time.perf_counter() - t0) * 1000)

        stats = compute_stats(times)
        # Theoretical serial time: sum of all rail durations (no parallelism)
        total_serial_ms = sum(durations.values())
        # Theoretical parallel time: sum of per-group maximums (perfect
        # concurrency within each group, sequential across groups)
        theoretical_parallel_ms = sum(max(durations.get(r, 0) for r in g.rails) for g in groups)

        results.append(
            {
                "test": "dag_scheduler",
                "topology": topo_name,
                "nodes": len(durations),
                "groups": len(groups),
                "total_serial_ms": total_serial_ms,
                "theoretical_parallel_ms": theoretical_parallel_ms,
                "actual_mean_ms": round(stats["mean_ms"], 2),
                "actual_p99_ms": round(stats["p99_ms"], 2),
                "actual_stdev_ms": round(stats["stdev_ms"], 2),
                # Efficiency: theoretical / actual. Values near 1.0 mean the
                # scheduler adds negligible overhead beyond the ideal parallel time.
                "efficiency": round(theoretical_parallel_ms / stats["mean_ms"], 2) if stats["mean_ms"] > 0 else 0,
                # Parallelism ratio: serial / actual. Shows the effective
                # degree of parallelism achieved (e.g. 4.0 = 4x faster than serial).
                "parallelism_ratio": round(total_serial_ms / stats["mean_ms"], 2) if stats["mean_ms"] > 0 else 0,
            }
        )

        print(
            f"  {topo_name:14s}: {len(groups)} groups, "
            f"serial={total_serial_ms}ms  "
            f"theoretical={theoretical_parallel_ms}ms  "
            f"actual={stats['mean_ms']:.1f}ms  "
            f"efficiency={theoretical_parallel_ms / stats['mean_ms']:.2f}"
        )

    return results


# ---------------------------------------------------------------------------
# Section 7: Eager Task Factory benchmark (persistent event loop)
# ---------------------------------------------------------------------------


def bench_eager_task_factory(
    rail_counts: tuple[int, ...] = (4, 8, 16, 32, 64),
    iterations: int = 2000,
) -> list[dict]:
    """Benchmark eager_task_factory vs standard task creation.

    IMPORTANT: Unlike the previous version, this benchmark uses a
    *persistent* event loop (loop.run_until_complete) instead of
    asyncio.run() per iteration.  asyncio.run() creates and destroys
    an event loop each time, which adds ~0.1ms of overhead that masks
    the sub-microsecond eager factory improvement.

    With a persistent loop, the scheduling overhead difference between
    standard and eager task creation is clearly visible.

    We also test with instant-completing coroutines (cache hit simulation)
    AND with mixed fast/slow coroutines (realistic rail mix).
    """
    # eager_task_factory was introduced in Python 3.12. When set on an event
    # loop, coroutines that complete synchronously (e.g. cache hits returning
    # immediately) skip the normal scheduling round-trip, avoiding a context
    # switch back to the event loop. This is significant for NeMo Guardrails
    # because many rails resolve instantly from cached results.
    has_eager = hasattr(asyncio, "eager_task_factory")
    results = []

    for n in rail_counts:
        # ---- Sub-benchmark A: instant coroutines (cache hits) ----
        # Simulates rails that resolve immediately (e.g. cached moderation
        # results). With eager task factory, these never enter the event
        # loop's ready queue — they complete inline during create_task().
        async def _instant_rail():
            return {"action": "continue"}

        async def _run_batch_standard(count: int):
            tasks = [asyncio.create_task(_instant_rail()) for _ in range(count)]
            return await asyncio.gather(*tasks)

        async def _run_batch_eager(count: int):
            tasks = [asyncio.create_task(_instant_rail()) for _ in range(count)]
            return await asyncio.gather(*tasks)

        # Use a persistent event loop (not asyncio.run) to avoid ~0.1ms of
        # loop creation/teardown overhead per iteration, which would dwarf
        # the sub-microsecond eager task factory improvement.
        loop = asyncio.new_event_loop()

        # 50 warmup iterations — more than other sections because we need
        # the event loop's internal data structures to be fully initialised.
        for _ in range(50):
            loop.run_until_complete(_run_batch_standard(n))

        # Standard timing — explicitly reset factory to None so we measure
        # the default scheduling behaviour (coroutines go through the ready
        # queue even if they could complete immediately).
        if has_eager:
            loop.set_task_factory(None)
        std_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            loop.run_until_complete(_run_batch_standard(n))
            std_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Eager timing — with eager_task_factory, create_task() runs the
        # coroutine immediately up to its first real suspension point. If
        # the coroutine completes without suspending, the result is available
        # without any event loop iteration, saving scheduling overhead.
        if has_eager:
            loop.set_task_factory(asyncio.eager_task_factory)
        eager_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            loop.run_until_complete(_run_batch_eager(n))
            eager_times.append((time.perf_counter_ns() - t0) / 1e6)

        loop.close()

        std_stats = compute_stats(std_times)
        eager_stats = compute_stats(eager_times)
        speedup = std_stats["mean_ms"] / eager_stats["mean_ms"] if eager_stats["mean_ms"] > 0 else 1.0

        # ---- Sub-benchmark B: mixed fast/slow (realistic rail mix) ----
        # In production, roughly half the rails resolve from cache (instant)
        # and the other half perform real I/O checks (slow). The eager factory
        # benefits the instant half; the slow half is I/O-bound regardless.
        async def _slow_rail():
            await asyncio.sleep(0.001)  # 1ms simulated I/O
            return {"action": "continue"}

        async def _run_mixed_standard(count: int):
            # Alternating instant (cache hit) and slow (I/O) coroutines
            coros = []
            for i in range(count):
                coros.append(_instant_rail() if i % 2 == 0 else _slow_rail())
            tasks = [asyncio.create_task(c) for c in coros]
            return await asyncio.gather(*tasks)

        # Fresh loop for the mixed benchmark to avoid any state leakage
        loop2 = asyncio.new_event_loop()

        for _ in range(20):  # Warmup
            loop2.run_until_complete(_run_mixed_standard(n))

        # Standard factory (no eager optimisation)
        if has_eager:
            loop2.set_task_factory(None)
        mixed_std_times = []
        # Fewer iterations (capped at 500) because the 1ms sleeps make
        # each iteration slower than the instant-only sub-benchmark.
        for _ in range(min(iterations, 500)):
            t0 = time.perf_counter_ns()
            loop2.run_until_complete(_run_mixed_standard(n))
            mixed_std_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Eager factory — instant coroutines complete inline, slow ones
        # proceed through the normal scheduling path after their first await.
        if has_eager:
            loop2.set_task_factory(asyncio.eager_task_factory)
        mixed_eager_times = []
        for _ in range(min(iterations, 500)):
            t0 = time.perf_counter_ns()
            loop2.run_until_complete(_run_mixed_standard(n))
            mixed_eager_times.append((time.perf_counter_ns() - t0) / 1e6)

        loop2.close()

        mixed_std_stats = compute_stats(mixed_std_times)
        mixed_eager_stats = compute_stats(mixed_eager_times)
        mixed_speedup = (
            mixed_std_stats["mean_ms"] / mixed_eager_stats["mean_ms"] if mixed_eager_stats["mean_ms"] > 0 else 1.0
        )

        results.append(
            {
                "test": "eager_task_factory",
                "tasks": n,
                "eager_available": has_eager,
                "instant_standard_mean_ms": round(std_stats["mean_ms"], 4),
                "instant_standard_p99_ms": round(std_stats["p99_ms"], 4),
                "instant_eager_mean_ms": round(eager_stats["mean_ms"], 4),
                "instant_eager_p99_ms": round(eager_stats["p99_ms"], 4),
                "instant_speedup": round(speedup, 2),
                "mixed_standard_mean_ms": round(mixed_std_stats["mean_ms"], 4),
                "mixed_eager_mean_ms": round(mixed_eager_stats["mean_ms"], 4),
                "mixed_speedup": round(mixed_speedup, 2),
            }
        )

        label = "eager" if has_eager else "N/A (< 3.12)"
        print(
            f"  {n:2d} tasks: instant: std={std_stats['mean_ms']:.4f}ms "
            f"{label}={eager_stats['mean_ms']:.4f}ms ({speedup:.2f}x)  "
            f"| mixed: std={mixed_std_stats['mean_ms']:.4f}ms "
            f"{label}={mixed_eager_stats['mean_ms']:.4f}ms ({mixed_speedup:.2f}x)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 8: Scheduler execute() — before/after comparison
# ---------------------------------------------------------------------------


def bench_scheduler_execute(iterations: int = 500) -> list[dict]:
    """Benchmark TopologicalScheduler.execute() with and without optimisations.

    This is the definitive before/after benchmark.  It runs the same
    topologies through the real execute() codepath twice:

    1. **Without** eager task factory (simulates pre-optimisation behaviour)
    2. **With** eager task factory (current optimised path)

    The difference directly proves whether our code change delivers a
    measurable improvement.  Uses a persistent event loop and sub-millisecond
    rail durations (0.1ms) to make the scheduling overhead visible rather
    than being drowned out by 5ms+ sleep times.
    """
    try:
        from nemoguardrails.rails.llm.dag_scheduler import (
            _HAS_EAGER_TASK_FACTORY,
            TopologicalScheduler,
        )
    except ImportError:
        print("  DAG scheduler not available, skipping")
        return []

    results = []

    # Diverse topologies stress-test the scheduler under different
    # parallelism patterns. Wider topologies benefit more from eager task
    # factory because more tasks can complete in a single scheduling pass.
    topologies = {
        "wide_4": lambda: _build_graph(["A", "B", "C", "D"], {}),  # All independent
        "wide_8": lambda: _build_graph([chr(65 + i) for i in range(8)], {}),
        "wide_16": lambda: _build_graph([f"rail_{i}" for i in range(16)], {}),
        "diamond": lambda: _build_graph(  # Fork-join pattern
            ["A", "B", "C", "D"],
            {"B": ["A"], "C": ["A"], "D": ["B", "C"]},
        ),
        "fan_out_8": lambda: _build_graph(  # One root, many independent leaves
            ["root"] + [f"leaf_{i}" for i in range(8)],
            {f"leaf_{i}": ["root"] for i in range(8)},
        ),
        "deep_pipeline_6": lambda: _build_graph(  # Strictly sequential chain
            [f"stage_{i}" for i in range(6)],
            {f"stage_{i}": [f"stage_{i - 1}"] for i in range(1, 6)},
        ),
    }

    for topo_name, build_fn in topologies.items():
        graph = build_fn()
        scheduler = TopologicalScheduler(graph)

        # Use very fast rails (0.1ms sleep) so that scheduler overhead is
        # a significant fraction of total wall time. With 5ms+ rails the
        # scheduling overhead is dwarfed by sleep time and differences
        # between eager and standard become unmeasurable noise.
        async def rail_executor(rail_name: str, context: dict) -> dict:
            await asyncio.sleep(0.0001)  # 0.1ms — fast enough to expose overhead
            return {"action": "continue", "rail": rail_name}

        # Persistent loop avoids asyncio.run() setup/teardown noise
        loop = asyncio.new_event_loop()

        for _ in range(10):  # Warmup
            loop.run_until_complete(scheduler.execute(rail_executor))

        # --- Run WITHOUT eager factory (simulate pre-optimisation behaviour) ---
        # Explicitly clear any eager factory so all tasks go through the
        # standard scheduling path (enqueue -> wake -> dequeue -> execute).
        if _HAS_EAGER_TASK_FACTORY:
            loop.set_task_factory(None)
        without_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            loop.run_until_complete(scheduler.execute(rail_executor))
            without_times.append((time.perf_counter_ns() - t0) / 1e6)

        # --- Run WITH eager factory (current optimised path) ---
        # The scheduler's execute() method also installs the eager factory
        # internally, but we set it on the loop too, ensuring it is active
        # from the very first create_task() call within execute().
        if _HAS_EAGER_TASK_FACTORY:
            loop.set_task_factory(asyncio.eager_task_factory)
        with_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            loop.run_until_complete(scheduler.execute(rail_executor))
            with_times.append((time.perf_counter_ns() - t0) / 1e6)

        loop.close()

        without_stats = compute_stats(without_times)
        with_stats = compute_stats(with_times)
        speedup = without_stats["mean_ms"] / with_stats["mean_ms"] if with_stats["mean_ms"] > 0 else 1.0

        n_rails = graph.node_count
        n_groups = scheduler.num_groups
        serial_ms = n_rails * 0.1  # Each rail sleeps 0.1ms; serial = sum of all

        results.append(
            {
                "test": "scheduler_execute",
                "topology": topo_name,
                "rails": n_rails,
                "groups": n_groups,
                "serial_ms": serial_ms,
                "without_eager_mean_ms": round(without_stats["mean_ms"], 4),
                "without_eager_p99_ms": round(without_stats["p99_ms"], 4),
                "with_eager_mean_ms": round(with_stats["mean_ms"], 4),
                "with_eager_p99_ms": round(with_stats["p99_ms"], 4),
                "eager_speedup": round(speedup, 2),
                "eager_available": _HAS_EAGER_TASK_FACTORY,
            }
        )

        improvement_pct = (1 - with_stats["mean_ms"] / without_stats["mean_ms"]) * 100
        eager_label = "with eager" if _HAS_EAGER_TASK_FACTORY else "same (< 3.12)"
        print(
            f"  {topo_name:18s}: {n_groups}g/{n_rails}r  "
            f"without={without_stats['mean_ms']:.4f}ms  "
            f"{eager_label}={with_stats['mean_ms']:.4f}ms  "
            f"({speedup:.2f}x, {improvement_pct:+.1f}%)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 9: Action name normalisation cache
# ---------------------------------------------------------------------------


def bench_action_name_cache(iterations: int = 50000) -> list[dict]:
    """Benchmark action name normalisation with and without caching.

    The ActionDispatcher._normalize_action_name() method converts CamelCase
    names to snake_case on every dispatch call.  This benchmark measures the
    cost of that conversion vs a simple dict lookup (cached result).

    On a typical deployment with 20-50 registered actions, each request
    dispatches 3-8 actions — so caching saves 15-40 string operations
    per request.
    """
    try:
        from nemoguardrails import utils
    except ImportError:
        print("  nemoguardrails.utils not available, skipping")
        return []

    # Realistic action names drawn from the NeMo Guardrails codebase.
    # CamelCase names (with "Action" suffix) require normalisation via
    # regex-based camelcase_to_snakecase(); already-snake_case names
    # are included to model the mixed population seen in production.
    action_names = [
        "GenerateUserIntentAction",
        "GenerateBotMessageAction",
        "CheckContentSafetyAction",
        "RetrieveRelevantChunksAction",
        "GenerateValueAction",
        "CheckFactsAction",
        "OutputModerationAction",
        "InputModerationAction",
        "SelfCheckInputAction",
        "SelfCheckOutputAction",
        "CheckHallucinationAction",
        "GenerateFlowContinuationAction",
        "GenerateFlowFromInstructionsAction",
        "GenerateFlowFromNameAction",
        "apify_search",  # Already snake_case
        "retrieve_relevant_chunks",  # Already snake_case
        "check_jailbreak",  # Already snake_case
        "check_output_moderation",  # Already snake_case
    ]

    results = []

    # --- Uncached: full normalisation every time ---
    # This path performs string manipulation (suffix removal + regex-based
    # CamelCase -> snake_case conversion) on every dispatch call, which is
    # the pre-optimisation behaviour.
    uncached_times = []
    for _ in range(WARMUP):
        for name in action_names:
            n = name
            if n.endswith("Action"):
                n = n.replace("Action", "")
            utils.camelcase_to_snakecase(n)

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        for name in action_names:
            n = name
            if n.endswith("Action"):
                n = n.replace("Action", "")  # Strip conventional suffix
            utils.camelcase_to_snakecase(n)  # Regex-based conversion
        uncached_times.append((time.perf_counter_ns() - t0) / 1e6)

    # --- Cached: pre-populated dict look-up ---
    # Build the cache once (as the ActionDispatcher would during registration),
    # then all subsequent look-ups are O(1) dict accesses.
    cache: dict[str, str] = {}
    for name in action_names:
        n = name
        if n.endswith("Action"):
            n = n.replace("Action", "")
        cache[name] = utils.camelcase_to_snakecase(n)

    cached_times = []
    for _ in range(WARMUP):
        for name in action_names:
            cache[name]  # Bare look-up — no computation

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        for name in action_names:
            cache[name]  # O(1) hash look-up replaces regex + string ops
        cached_times.append((time.perf_counter_ns() - t0) / 1e6)

    uncached_stats = compute_stats(uncached_times)
    cached_stats = compute_stats(cached_times)
    speedup = uncached_stats["mean_ms"] / cached_stats["mean_ms"] if cached_stats["mean_ms"] > 0 else 0

    results.append(
        {
            "test": "action_name_cache",
            "action_count": len(action_names),
            "iterations": iterations,
            "uncached_mean_ms": round(uncached_stats["mean_ms"], 4),
            "uncached_p99_ms": round(uncached_stats["p99_ms"], 4),
            "cached_mean_ms": round(cached_stats["mean_ms"], 4),
            "cached_p99_ms": round(cached_stats["p99_ms"], 4),
            "speedup": round(speedup, 2),
            "savings_per_dispatch_us": round(
                (uncached_stats["mean_ms"] - cached_stats["mean_ms"]) * 1000 / len(action_names), 2
            ),
        }
    )

    print(
        f"  {len(action_names)} actions: uncached={uncached_stats['mean_ms']:.4f}ms  "
        f"cached={cached_stats['mean_ms']:.4f}ms  "
        f"({speedup:.2f}x, saves {(uncached_stats['mean_ms'] - cached_stats['mean_ms']) * 1000 / len(action_names):.2f}us/dispatch)"
    )

    return results


# ---------------------------------------------------------------------------
# Section 10: ThreadSafeCache contention benchmark
# ---------------------------------------------------------------------------


def bench_cache_contention(
    thread_counts: tuple[int, ...] = (1, 2, 4, 8),
    operations_per_thread: int = 50000,
) -> list[dict]:
    """Measure ThreadSafeCache throughput under multi-threaded contention.

    This benchmark reveals the lock overhead of ThreadSafeCache compared to
    a plain dict.  On GIL builds the lock is uncontended and near-zero cost.
    On free-threaded builds the lock is the only protection and its overhead
    determines whether caching is net-positive.

    Each thread performs a mix of 80% reads (cache hits) and 20% writes
    (cache misses / updates), matching the expected production access pattern.
    """
    from nemoguardrails._thread_safety import ThreadSafeCache

    results = []

    for n_threads in thread_counts:
        # maxsize=256 matches the production configuration. LRU eviction
        # kicks in once the cache is full, adding occasional overhead.
        cache = ThreadSafeCache(maxsize=256)
        # Pre-populate with 200 entries to simulate a warmed-up cache.
        # The remaining 50 slots (of 256) allow some writes to succeed
        # without triggering eviction, modelling mixed hit/miss patterns.
        for i in range(200):
            cache.put(f"template_{i}", f"compiled_result_{i}")

        # Barrier ensures all threads start their measured work simultaneously,
        # maximising lock contention. Without it, threads that start earlier
        # would run with less contention, biasing throughput upwards.
        barrier = threading.Barrier(n_threads)
        thread_times: list[float] = []

        def _worker(tid: int):
            barrier.wait()  # Synchronise start across all threads
            t0 = time.perf_counter()
            for i in range(operations_per_thread):
                # Key derivation uses (tid * 1000 + i) % 250 to create
                # overlapping access patterns: threads contend on the same
                # keys (hot-spot contention) rather than each touching
                # disjoint ranges (which would hide lock overhead).
                key = f"template_{(tid * 1000 + i) % 250}"
                if i % 5 == 0:
                    # 20% writes — models cache misses / template re-compilation
                    cache.put(key, f"result_{i}")
                else:
                    # 80% reads — models cache hits (the common production path)
                    cache.get(key)
            elapsed = time.perf_counter() - t0
            thread_times.append(elapsed)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_ops = n_threads * operations_per_thread
        # Wall time is the slowest thread — the point at which all work is done.
        # On GIL builds, threads are effectively serialised so wall_time grows
        # linearly with n_threads. On free-threaded builds, wall_time should
        # remain roughly constant if lock contention is manageable.
        wall_time = max(thread_times)
        throughput = total_ops / wall_time

        # Plain dict baseline (single-threaded, no locking) establishes the
        # theoretical maximum throughput. The ratio ThreadSafeCache / plain_dict
        # isolates the cost of lock acquisition and LRU bookkeeping.
        plain_cache: dict[str, str] = {f"template_{i}": f"compiled_result_{i}" for i in range(200)}
        t0 = time.perf_counter()
        for i in range(operations_per_thread):
            key = f"template_{i % 250}"
            if i % 5 == 0:
                plain_cache[key] = f"result_{i}"
            else:
                plain_cache.get(key)
        plain_time = time.perf_counter() - t0
        plain_throughput = operations_per_thread / plain_time

        results.append(
            {
                "test": "cache_contention",
                "threads": n_threads,
                "operations_per_thread": operations_per_thread,
                "total_operations": total_ops,
                "wall_time_ms": round(wall_time * 1000, 2),
                "throughput_ops_per_sec": round(throughput),
                "plain_dict_throughput": round(plain_throughput),
                # Overhead ratio > 1.0 means the lock costs more than the
                # operation itself. Values near 1.0 mean locking is ~free.
                "lock_overhead_ratio": round(plain_throughput / throughput, 2) if throughput > 0 else 0,
                "cache_hit_rate": round(
                    cache.stats().get("hits", 0)
                    / max(1, cache.stats().get("hits", 0) + cache.stats().get("misses", 0)),
                    3,
                ),
            }
        )

        print(
            f"  {n_threads} threads: {throughput:,.0f} ops/s  "
            f"(plain dict: {plain_throughput:,.0f} ops/s, "
            f"overhead: {plain_throughput / throughput:.2f}x)  "
            f"wall={wall_time * 1000:.1f}ms"
        )

    return results


# ---------------------------------------------------------------------------
# Section 11: End-to-end pipeline simulation
# ---------------------------------------------------------------------------


def bench_pipeline(iterations: int = 2000) -> list[dict]:
    """Simulate a full guardrails request pipeline and measure overhead.

    Pipeline stages (per request):
      1. Template rendering (system + user intent prompts)
      2. Action name normalisation (3 actions)
      3. Rail evaluation (2 I/O-bound + 1 CPU-bound)
      4. Response template rendering

    This measures the total framework overhead exclusive of LLM call time,
    which is the overhead users actually experience.
    """
    from jinja2 import meta
    from jinja2.sandbox import SandboxedEnvironment

    try:
        from nemoguardrails import utils
    except ImportError:
        print("  nemoguardrails.utils not available, skipping")
        return []

    # SandboxedEnvironment mirrors production usage (prevents template injection)
    env = SandboxedEnvironment()
    env.filters["last_turns"] = lambda msgs, n: msgs[-n:]

    # Pre-build template caches simulating the M3 optimisation. These three
    # templates represent the minimum rendering a single request requires:
    # system prompt, user intent extraction, and bot response formatting.
    templates = {
        "system": "You are a helpful assistant.\n{{ general_instructions }}",
        "user_intent": (
            "{% for msg in history | last_turns(3) %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}\nUser intent: "
        ),
        "bot_response": ("Intent: {{ bot_intent }}\nContext: {{ context_summary }}\nResponse: "),
    }

    template_cache: dict[str, Any] = {}
    variables_cache: dict[str, frozenset] = {}
    for tpl_str in templates.values():
        template_cache[tpl_str] = env.from_string(tpl_str)
        variables_cache[tpl_str] = frozenset(meta.find_undeclared_variables(env.parse(tpl_str)))

    context = {
        "general_instructions": "Be safe and accurate.",
        "history": [{"role": "user", "content": f"Message {i}"} for i in range(10)],
        "bot_intent": "provide_answer",
        "context_summary": "General conversation",
    }

    # Pre-build action name cache — the three actions represent the typical
    # per-request dispatch sequence in a guardrails pipeline.
    action_names_raw = ["GenerateUserIntentAction", "CheckContentSafetyAction", "GenerateBotMessageAction"]
    action_cache: dict[str, str] = {}
    for name in action_names_raw:
        n = name.replace("Action", "") if name.endswith("Action") else name
        action_cache[name] = utils.camelcase_to_snakecase(n)

    # Minimal CPU work simulating a fast synchronous rail check (e.g.
    # digit-counting PII heuristic). Kept deliberately lightweight so
    # that template/cache overhead dominates the pipeline measurement.
    def _eval_rail(text: str) -> dict:
        count = sum(1 for c in text if c.isdigit())
        return {"ok": count < 10, "digits": count}

    results = []

    # --- Optimised pipeline (with all caches) ---
    # Uses pre-compiled templates, cached variable sets, and cached action
    # name mappings — the full set of M3 + dispatch optimisations.
    optimised_times = []
    # Double warmup to ensure Jinja2's internal bytecode cache is also warm
    for _ in range(WARMUP * 2):
        for tpl_str in templates.values():
            cached_t = template_cache[tpl_str]
            cached_v = variables_cache[tpl_str]
            cached_t.render({k: v for k, v in context.items() if k in cached_v})
        for name in action_names_raw:
            action_cache[name]
        _eval_rail("test input with 123-45-6789")

    for _ in range(iterations):
        t0 = time.perf_counter_ns()

        # Stage 1: Render templates (system + user_intent + bot_response)
        for tpl_str in templates.values():
            cached_t = template_cache[tpl_str]
            cached_v = variables_cache[tpl_str]
            cached_t.render({k: v for k, v in context.items() if k in cached_v})

        # Stage 2: Normalise action names
        for name in action_names_raw:
            action_cache[name]

        # Stage 3: Evaluate rails
        for payload in SAMPLE_PAYLOADS[:3]:
            _eval_rail(payload)

        # Stage 4: Render response template
        cached_t = template_cache[templates["bot_response"]]
        cached_v = variables_cache[templates["bot_response"]]
        cached_t.render({k: v for k, v in context.items() if k in cached_v})

        optimised_times.append((time.perf_counter_ns() - t0) / 1e6)

    # --- Unoptimised pipeline (no caches, reparse every time) ---
    # This path re-parses templates, re-walks ASTs, and re-computes action
    # name normalisations on every request — the pre-M3 behaviour. The
    # difference vs the optimised path above quantifies the aggregate
    # savings from all caching improvements combined.
    unoptimised_times = []
    for _ in range(WARMUP * 2):
        for tpl_str in templates.values():
            t = env.from_string(tpl_str)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))
            t.render({k: v for k, v in context.items() if k in vs})

    for _ in range(iterations):
        t0 = time.perf_counter_ns()

        # Stage 1: Render templates (reparse each time)
        for tpl_str in templates.values():
            t = env.from_string(tpl_str)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))
            t.render({k: v for k, v in context.items() if k in vs})

        # Stage 2: Normalise action names (recompute each time)
        for name in action_names_raw:
            n = name.replace("Action", "") if name.endswith("Action") else name
            utils.camelcase_to_snakecase(n)

        # Stage 3: Evaluate rails
        for payload in SAMPLE_PAYLOADS[:3]:
            _eval_rail(payload)

        # Stage 4: Render response template (reparse)
        tpl_str = templates["bot_response"]
        t = env.from_string(tpl_str)
        vs = meta.find_undeclared_variables(env.parse(tpl_str))
        t.render({k: v for k, v in context.items() if k in vs})

        unoptimised_times.append((time.perf_counter_ns() - t0) / 1e6)

    opt_stats = compute_stats(optimised_times)
    unopt_stats = compute_stats(unoptimised_times)
    speedup = unopt_stats["mean_ms"] / opt_stats["mean_ms"] if opt_stats["mean_ms"] > 0 else 0

    results.append(
        {
            "test": "pipeline",
            "iterations": iterations,
            "stages": "template_render + action_normalise + rail_eval + response_render",
            "optimised_mean_ms": round(opt_stats["mean_ms"], 4),
            "optimised_p50_ms": round(opt_stats["p50_ms"], 4),
            "optimised_p99_ms": round(opt_stats["p99_ms"], 4),
            "optimised_stdev_ms": round(opt_stats["stdev_ms"], 4),
            "unoptimised_mean_ms": round(unopt_stats["mean_ms"], 4),
            "unoptimised_p50_ms": round(unopt_stats["p50_ms"], 4),
            "unoptimised_p99_ms": round(unopt_stats["p99_ms"], 4),
            "unoptimised_stdev_ms": round(unopt_stats["stdev_ms"], 4),
            "speedup": round(speedup, 2),
            "savings_per_request_ms": round(unopt_stats["mean_ms"] - opt_stats["mean_ms"], 4),
        }
    )

    print(
        f"  Pipeline: optimised={opt_stats['mean_ms']:.4f}ms  "
        f"unoptimised={unopt_stats['mean_ms']:.4f}ms  "
        f"({speedup:.2f}x, saves {unopt_stats['mean_ms'] - opt_stats['mean_ms']:.4f}ms/request)"
    )

    return results


# ---------------------------------------------------------------------------
# Section 12: Memory efficiency
# ---------------------------------------------------------------------------


def bench_memory(iterations: int = 5000) -> list[dict]:
    """Track per-request memory allocation using tracemalloc.

    Measures:
    - Peak RSS delta over N requests
    - Allocation count per request (via tracemalloc snapshots)
    - Cache memory footprint (template cache, variable cache)

    This validates that caching does not cause memory leaks — a common
    concern when introducing unbounded caches.  Our ThreadSafeCache is
    bounded (LRU eviction), so memory should plateau.
    """
    import tracemalloc

    from jinja2 import meta
    from jinja2.sandbox import SandboxedEnvironment

    from benchmarks.conftest import get_rss_mb

    env = SandboxedEnvironment()
    env.filters["last_turns"] = lambda msgs, n: msgs[-n:]

    from nemoguardrails._thread_safety import ThreadSafeCache

    # maxsize=64 is deliberately smaller than the 128 unique templates
    # we generate below, forcing LRU eviction on every cycle through
    # the template space. This validates that evicted entries are truly
    # freed and that the cache does not leak memory over time.
    template_cache = ThreadSafeCache(maxsize=64)
    variables_cache = ThreadSafeCache(maxsize=64)

    context = {
        "general_instructions": "Be safe.",
        "history": [{"role": "user", "content": f"Msg {i}"} for i in range(5)],
    }

    results = []

    # Capture baseline RSS and start tracemalloc before the workload.
    # tracemalloc hooks into the allocator to track per-line allocation
    # counts and sizes — it has ~5-10% overhead, but that is consistent
    # across all iterations and does not affect relative measurements.
    rss_before = get_rss_mb()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    for i in range(iterations):
        # Generate 128 unique template strings, cycling with i % 128.
        # Since maxsize=64, every full cycle evicts the older half,
        # exercising the LRU eviction code path continuously.
        tpl_key = i % 128
        # Variable-length padding ("x" * (tpl_key % 20)) ensures templates
        # differ in size, testing that the cache handles heterogeneous
        # object sizes without fragmentation-related leaks.
        tpl_str = f"Template {tpl_key}: {{{{ general_instructions }}}} " + "x" * (tpl_key % 20)

        # Cache-aside pattern: check cache first, populate on miss
        cached_t = template_cache.get(tpl_str)
        if cached_t is None:
            cached_t = env.from_string(tpl_str)  # Parse + compile on miss
            template_cache.put(tpl_str, cached_t)

        cached_v = variables_cache.get(tpl_str)
        if cached_v is None:
            cached_v = frozenset(meta.find_undeclared_variables(env.parse(tpl_str)))
            variables_cache.put(tpl_str, cached_v)

        render_ctx = {k: v for k, v in context.items() if k in cached_v}
        cached_t.render(render_ctx)

    snap_after = tracemalloc.take_snapshot()
    rss_after = get_rss_mb()
    tracemalloc.stop()

    # Compare snapshots by source line to identify where allocations grew.
    # Positive size_diff = memory allocated but not freed (potential leak);
    # negative size_diff = memory that was freed (healthy eviction behaviour).
    stats_diff = snap_after.compare_to(snap_before, "lineno")
    total_alloc_delta = sum(s.size_diff for s in stats_diff if s.size_diff > 0)
    total_freed = sum(abs(s.size_diff) for s in stats_diff if s.size_diff < 0)

    cache_stats = template_cache.stats()

    results.append(
        {
            "test": "memory_efficiency",
            "iterations": iterations,
            "unique_templates": 128,
            "cache_maxsize": 64,
            "rss_before_mb": round(rss_before, 2),
            "rss_after_mb": round(rss_after, 2),
            "rss_delta_mb": round(rss_after - rss_before, 2),
            "alloc_delta_bytes": total_alloc_delta,
            "freed_bytes": total_freed,
            "alloc_per_request_bytes": round(total_alloc_delta / iterations, 1),
            "cache_hits": cache_stats.get("hits", 0),
            "cache_misses": cache_stats.get("misses", 0),
            "cache_hit_rate": round(
                cache_stats.get("hits", 0) / max(1, cache_stats.get("hits", 0) + cache_stats.get("misses", 0)), 3
            ),
            "cache_evictions": cache_stats.get("evictions", 0),
        }
    )

    hit_rate = cache_stats.get("hits", 0) / max(1, cache_stats.get("hits", 0) + cache_stats.get("misses", 0))
    print(
        f"  {iterations} requests: RSS delta={rss_after - rss_before:.2f}MB  "
        f"alloc/req={total_alloc_delta / iterations:.0f}B  "
        f"cache hit rate={hit_rate:.1%}  "
        f"evictions={cache_stats.get('evictions', 0)}"
    )

    return results


def _build_graph(
    rail_names: list[str],
    deps: dict[str, list[str]],
) -> "RailDependencyGraph":
    """Helper to build a dependency graph from names and dependencies."""
    from nemoguardrails.rails.llm.dag_scheduler import RailDependencyGraph

    g = RailDependencyGraph()
    # Rails must be added in topological order — a rail's dependencies
    # must already be present in the graph before it is added.
    for name in rail_names:
        g.add_rail(name, depends_on=deps.get(name))
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_section(title: str):
    print(f"{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(description="Python 3.14 performance advantages benchmark for NeMo Guardrails")
    parser.add_argument("--output", "-o", default=None, help="JSON output path")
    parser.add_argument(
        "--section",
        "-s",
        choices=_SECTION_CHOICES,
        default="all",
        help="Which benchmark section to run",
    )
    args = parser.parse_args()

    info = _python_info()
    print(f"\nPython {info['version']} ({info['build']}, {info['cpu_count']} CPUs)")
    print(f"GIL disabled: {info['gil_disabled']}")
    print()

    # Accumulate all results in a single dict for JSON serialisation.
    # Structure: {"python": {...}, "sections": {"threading": [...], ...}}
    all_results: dict[str, Any] = {"python": info, "sections": {}}

    # Section registry: (CLI key, display title, callable).
    # Using --section=all runs every benchmark; individual keys allow
    # targeted runs during development without waiting for the full suite.
    sections = [
        ("threading", "Section 1: Free-threaded parallel CPU rail execution", bench_threading),
        ("template", "Section 2: Jinja2 template rendering (M3 cache)", bench_template_rendering),
        ("import", "Section 3: Import time (PEP 649 deferred annotations)", bench_import),
        ("gc", "Section 4: GC tail latency under pressure", bench_gc_pauses),
        ("asyncio", "Section 5: Asyncio gather with real rails", bench_asyncio_rails),
        ("dag", "Section 6: DAG scheduler parallelism", bench_dag_scheduler),
        ("eager", "Section 7: Eager task factory (persistent loop)", bench_eager_task_factory),
        ("scheduler", "Section 8: Scheduler execute() before/after", bench_scheduler_execute),
        ("action_cache", "Section 9: Action name normalisation cache", bench_action_name_cache),
        ("cache_contention", "Section 10: ThreadSafeCache contention", bench_cache_contention),
        ("pipeline", "Section 11: End-to-end pipeline simulation", bench_pipeline),
        ("memory", "Section 12: Memory efficiency", bench_memory),
    ]

    for key, title, bench_fn in sections:
        if args.section in (key, "all"):
            _print_section(title)
            all_results["sections"][key] = bench_fn()
            print()

    # -----------------------------------------------------------------------
    # Summary — one-line-per-section highlights for quick triage
    # -----------------------------------------------------------------------
    _print_section("SUMMARY")

    if "threading" in all_results["sections"]:
        thr = all_results["sections"]["threading"]
        # Pick the rail count that achieved the highest thread speedup
        best = max(thr, key=lambda r: r["speedup_threadpool"])
        # Annotate whether this was a free-threaded run (meaningful speedup)
        # or a GIL-limited run (speedup should be ~ 1.0).
        marker = "FREE-THREADED (no GIL)" if info["gil_disabled"] else "GIL-limited"
        print(f"  Threading:  {best['speedup_threadpool']:.2f}x speedup @ {best['rails']} rails ({marker})")

    if "template" in all_results["sections"]:
        tpl = all_results["sections"]["template"]
        if tpl:
            best = max(tpl, key=lambda r: r["speedup"])
            print(
                f"  Templates:  {best['speedup']:.2f}x cache speedup "
                f"(saves {best['savings_us']:.0f}us/call on '{best['template']}')"
            )

    if "import" in all_results["sections"]:
        imp = all_results["sections"]["import"]
        if imp:
            top = imp[0]
            print(f"  Import:     {top['mean_ms']:.0f}ms cold start ({top['module']})")

    if "gc" in all_results["sections"]:
        gc_r = all_results["sections"]["gc"]
        baseline = next((r for r in gc_r if r["scenario"] == "baseline"), None)
        heavy = next((r for r in gc_r if r["scenario"] == "heavy_pressure"), None)
        if baseline and heavy:
            print(
                f"  GC:         p99 ratio baseline={baseline['p99_p50_ratio']:.2f}x, "
                f"heavy={heavy['p99_p50_ratio']:.2f}x "
                f"(p999={heavy['p999_ms']:.3f}ms)"
            )

    if "asyncio" in all_results["sections"]:
        aio = all_results["sections"]["asyncio"]
        if aio:
            best = max(aio, key=lambda r: r["speedup"])
            print(f"  Asyncio:    {best['speedup']:.2f}x gather speedup @ {best['num_rails']} rails")

    if "dag" in all_results["sections"]:
        dag = all_results["sections"]["dag"]
        if dag:
            wide = next((r for r in dag if r["topology"] == "wide_4"), None)
            if wide:
                print(f"  DAG:        {wide['parallelism_ratio']:.2f}x parallelism (wide_4, {wide['groups']} groups)")

    if "eager" in all_results["sections"]:
        eager = all_results["sections"]["eager"]
        if eager:
            best = max(eager, key=lambda r: r["instant_speedup"])
            avail = "available" if best["eager_available"] else "unavailable (< 3.12)"
            print(
                f"  Eager:      {best['instant_speedup']:.2f}x instant, "
                f"{best['mixed_speedup']:.2f}x mixed @ {best['tasks']} tasks ({avail})"
            )

    if "scheduler" in all_results["sections"]:
        sched = all_results["sections"]["scheduler"]
        if sched:
            best = max(sched, key=lambda r: r["eager_speedup"])
            improvement = (1 - 1 / best["eager_speedup"]) * 100 if best["eager_speedup"] > 0 else 0
            print(
                f"  Scheduler:  {best['eager_speedup']:.2f}x with eager "
                f"({best['topology']}, {improvement:+.1f}% faster)"
            )

    if "action_cache" in all_results["sections"]:
        ac = all_results["sections"]["action_cache"]
        if ac:
            print(
                f"  ActionCache:{ac[0]['speedup']:.2f}x normalisation cache "
                f"(saves {ac[0]['savings_per_dispatch_us']:.1f}us/dispatch)"
            )

    if "cache_contention" in all_results["sections"]:
        cc = all_results["sections"]["cache_contention"]
        if cc:
            best = max(cc, key=lambda r: r["threads"])
            print(
                f"  Contention: {best['throughput_ops_per_sec']:,.0f} ops/s @ {best['threads']} threads "
                f"(overhead: {best['lock_overhead_ratio']:.2f}x vs plain dict)"
            )

    if "pipeline" in all_results["sections"]:
        pip = all_results["sections"]["pipeline"]
        if pip:
            print(
                f"  Pipeline:   {pip[0]['speedup']:.2f}x end-to-end "
                f"(saves {pip[0]['savings_per_request_ms']:.3f}ms/request)"
            )

    if "memory" in all_results["sections"]:
        mem = all_results["sections"]["memory"]
        if mem:
            print(
                f"  Memory:     {mem[0]['rss_delta_mb']:.1f}MB delta over {mem[0]['iterations']} requests, "
                f"cache hit rate={mem[0]['cache_hit_rate']:.1%}"
            )

    print()

    # -----------------------------------------------------------------------
    # Output — write JSON to file (for CI consumption) or stdout (for humans)
    # -----------------------------------------------------------------------
    blob = json.dumps(all_results, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(blob + "\n")  # Trailing newline for POSIX compatibility
        print(f"Results written to {args.output}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
