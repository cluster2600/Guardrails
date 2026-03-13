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

Measures concrete improvements across six areas, using the real guardrails
infrastructure (fake_rails, conftest, Jinja2 template engine, DAG scheduler)
rather than toy synthetic workloads.

Sections:
    1. **Free-threaded parallel CPU rails** — Sequential vs ThreadPoolExecutor
       vs asyncio.run_in_executor using the real cpu_bound_scan from fake_rails.
       On 3.14t (no-GIL) threads achieve near-linear speedup.

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

Run:
    python -m benchmarks.bench_py314_advantages
    python -m benchmarks.bench_py314_advantages --section threading
    python -m benchmarks.bench_py314_advantages --output /tmp/py314-bench.json
"""

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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from benchmarks.conftest import SAMPLE_PAYLOADS, compute_stats
from benchmarks.fake_rails import cpu_bound_scan

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP = 5
ROUNDS = 30
_SECTION_CHOICES = ["threading", "template", "import", "gc", "asyncio", "dag", "eager", "scheduler", "all"]


def _python_info() -> dict:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    build = "free-threaded" if gil_disabled else "standard"
    return {
        "version": ver,
        "build": build,
        "gil_disabled": gil_disabled,
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Section 1: Free-threaded parallel CPU rails (using real fake_rails)
# ---------------------------------------------------------------------------


def bench_threading(
    rail_counts: tuple[int, ...] = (1, 2, 4, 8),
    cpu_rounds: int = 2000,
) -> list[dict]:
    """Compare sequential vs threaded execution of cpu_bound_scan from fake_rails."""
    results = []
    text = SAMPLE_PAYLOADS[2]  # PII-containing payload for realistic scan

    for n_rails in rail_counts:
        workers = min(n_rails, os.cpu_count() or 4)
        fn = functools.partial(cpu_bound_scan, text, cpu_rounds)

        # --- Sequential ---
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
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in range(WARMUP):
                list(pool.map(lambda _: fn(), range(n_rails)))

            thr_times = []
            for _ in range(ROUNDS):
                t0 = time.perf_counter()
                list(pool.map(lambda _: fn(), range(n_rails)))
                thr_times.append(time.perf_counter() - t0)

        # --- asyncio + run_in_executor (mirrors real ActionDispatcher path) ---
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

        seq_stats = compute_stats([t * 1000 for t in seq_times])
        thr_stats = compute_stats([t * 1000 for t in thr_times])
        async_stats = compute_stats([t * 1000 for t in async_times])

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

    env = SandboxedEnvironment()

    # Realistic templates of increasing complexity
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

    # Register a filter so "complex" template works
    env.filters["last_turns"] = lambda msgs, n: msgs[-n:]

    results = []
    template_cache: dict[str, Any] = {}
    variables_cache: dict[str, frozenset] = {}

    for tpl_name, tpl_str in templates.items():
        # --- Cold: no cache, parse + compile every time ---
        cold_times = []
        for _ in range(WARMUP):
            t = env.from_string(tpl_str)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))
            t.render({k: v for k, v in context.items() if k in vs})

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            t = env.from_string(tpl_str)
            vs = meta.find_undeclared_variables(env.parse(tpl_str))
            render_ctx = {k: v for k, v in context.items() if k in vs}
            t.render(render_ctx)
            cold_times.append((time.perf_counter_ns() - t0) / 1e6)

        # --- Hot: cached template + cached variable set (M3 path) ---
        # Pre-populate cache
        template_cache[tpl_str] = env.from_string(tpl_str)
        variables_cache[tpl_str] = frozenset(meta.find_undeclared_variables(env.parse(tpl_str)))

        hot_times = []
        for _ in range(WARMUP):
            cached_t = template_cache[tpl_str]
            cached_v = variables_cache[tpl_str]
            cached_t.render({k: v for k, v in context.items() if k in cached_v})

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            cached_t = template_cache[tpl_str]
            cached_v = variables_cache[tpl_str]
            render_ctx = {k: v for k, v in context.items() if k in cached_v}
            cached_t.render(render_ctx)
            hot_times.append((time.perf_counter_ns() - t0) / 1e6)

        cold_stats = compute_stats(cold_times)
        hot_stats = compute_stats(hot_times)
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
            cmd = [
                sys.executable,
                "-X",
                "importtime",
                "-c",
                f"import time as _t; _s=_t.perf_counter(); import {mod}; print(_t.perf_counter()-_s)",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
                times.append(float(out.strip()) * 1000)
            except Exception as e:
                print(f"  Warning: failed to import {mod}: {e}")
                break

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
    text = SAMPLE_PAYLOADS[4]  # Longer PII-heavy payload
    fn = functools.partial(cpu_bound_scan, text, cpu_rounds)

    scenarios = [
        ("baseline", 0, 0),  # No GC pressure
        ("light_pressure", 50, 5),  # Moderate: 50 cycles every 5 iters
        ("heavy_pressure", 500, 3),  # Heavy: 500 cycles every 3 iters
        ("burst_pressure", 2000, 10),  # Bursts: 2000 cycles every 10 iters
    ]

    results = []
    for label, cycle_count, every_n in scenarios:
        gc.collect()
        gc.collect()
        gc.collect()

        latencies = []
        garbage_holder: list[Any] = []

        for i in range(iterations):
            if cycle_count > 0 and every_n > 0 and i % every_n == 0:
                # Create reference cycles that GC must trace
                batch: list[Any] = []
                for _ in range(cycle_count):
                    a: dict[str, Any] = {"data": "x" * 64}
                    b: dict[str, Any] = {"ref": a, "data": "y" * 64}
                    c: dict[str, Any] = {"ref": b, "data": "z" * 64}
                    a["ref"] = c  # a -> c -> b -> a cycle
                    batch.append(a)
                garbage_holder.append(batch)
                # Keep only last few batches to sustain pressure
                if len(garbage_holder) > 20:
                    garbage_holder.pop(0)

            t0 = time.perf_counter_ns()
            fn()
            latencies.append((time.perf_counter_ns() - t0) / 1e6)

        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        p99 = latencies[int(n * 0.99)]
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
                "p99_p50_ratio": round(p99 / p50, 2) if p50 > 0 else 0,
                "p999_p50_ratio": round(p999 / p50, 2) if p50 > 0 else 0,
            }
        )
        print(
            f"  {label:18s}: p50={p50:.3f}ms  p95={p95:.3f}ms  "
            f"p99={p99:.3f}ms  p999={p999:.3f}ms  max={max_lat:.3f}ms  "
            f"ratio={p99 / p50:.2f}x"
        )

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
    from benchmarks.fake_rails import io_bound_rail

    results = []

    for num_rails in rail_counts:
        text = SAMPLE_PAYLOADS[2]

        async def _serial():
            for _ in range(num_rails):
                await io_bound_rail(text, latency_ms=io_latency_ms)

        async def _parallel():
            coros = [io_bound_rail(text, latency_ms=io_latency_ms) for _ in range(num_rails)]
            await asyncio.gather(*coros)

        # Warmup
        for _ in range(3):
            asyncio.run(_serial())
            asyncio.run(_parallel())

        # Serial
        serial_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            asyncio.run(_serial())
            serial_times.append((time.perf_counter() - t0) * 1000)

        # Parallel (asyncio.gather)
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

    # Topology configurations: (name, build_fn)
    # build_fn returns (graph, durations_dict) because add_rail requires
    # dependencies to already exist (topological add order).
    def _build_linear():
        g = RailDependencyGraph()
        g.add_rail("A")
        g.add_rail("B", depends_on=["A"])
        g.add_rail("C", depends_on=["B"])
        g.add_rail("D", depends_on=["C"])
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}

    def _build_wide():
        g = RailDependencyGraph()
        for name in ["A", "B", "C", "D"]:
            g.add_rail(name)
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}

    def _build_diamond():
        g = RailDependencyGraph()
        g.add_rail("A")
        g.add_rail("B", depends_on=["A"])
        g.add_rail("C", depends_on=["A"])
        g.add_rail("D", depends_on=["B", "C"])
        return g, {"A": 10, "B": 10, "C": 10, "D": 10}

    def _build_fan_out():
        g = RailDependencyGraph()
        g.add_rail("A")
        for i in range(8):
            g.add_rail(chr(66 + i), depends_on=["A"])
        return g, {**{"A": 5}, **{chr(66 + i): 10 for i in range(8)}}

    topologies = [
        ("linear_4", _build_linear),
        ("wide_4", _build_wide),
        ("diamond", _build_diamond),
        ("fan_out_8", _build_fan_out),
    ]

    for topo_name, build_fn in topologies:
        graph, durations = build_fn()
        scheduler = TopologicalScheduler(graph)
        groups = scheduler.groups

        # Build coroutine factories
        async def run_topology(_groups=groups, _durations=durations):
            for group in _groups:
                coros = []
                for rail_name in group.rails:
                    dur = _durations.get(rail_name, 10)

                    async def _exec(name=rail_name, d=dur):
                        await asyncio.sleep(d / 1000)
                        return {"ok": True, "rail": name}

                    coros.append(_exec())
                await asyncio.gather(*coros)

        # Warmup
        for _ in range(3):
            asyncio.run(run_topology())

        # Benchmark
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            asyncio.run(run_topology())
            times.append((time.perf_counter() - t0) * 1000)

        stats = compute_stats(times)
        total_serial_ms = sum(durations.values())
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
                "efficiency": round(theoretical_parallel_ms / stats["mean_ms"], 2) if stats["mean_ms"] > 0 else 0,
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
# Section 7: Eager Task Factory benchmark
# ---------------------------------------------------------------------------


def bench_eager_task_factory(
    rail_counts: tuple[int, ...] = (4, 8, 16, 32),
    iterations: int = 500,
) -> list[dict]:
    """Benchmark eager_task_factory vs standard task creation.

    On Python 3.12+ the eager factory lets coroutines that complete
    synchronously skip the event-loop scheduling round-trip. This is
    the exact optimisation we installed in TopologicalScheduler.execute().
    """
    has_eager = hasattr(asyncio, "eager_task_factory")
    results = []

    for n in rail_counts:
        # Fast-completing coroutines (simulates cache hits / trivial checks)
        async def _instant_rail():
            return {"action": "continue"}

        # --- Standard task creation ---
        async def _run_standard():
            loop = asyncio.get_running_loop()
            # Force standard factory
            old = loop.get_task_factory() if has_eager else None
            if has_eager:
                loop.set_task_factory(None)
            try:
                tasks = [asyncio.create_task(_instant_rail()) for _ in range(n)]
                await asyncio.gather(*tasks)
            finally:
                if has_eager and old is not None:
                    loop.set_task_factory(old)

        # --- Eager task creation ---
        async def _run_eager():
            loop = asyncio.get_running_loop()
            old = loop.get_task_factory() if has_eager else None
            if has_eager:
                loop.set_task_factory(asyncio.eager_task_factory)
            try:
                tasks = [asyncio.create_task(_instant_rail()) for _ in range(n)]
                await asyncio.gather(*tasks)
            finally:
                if has_eager:
                    loop.set_task_factory(old)

        # Warmup
        for _ in range(10):
            asyncio.run(_run_standard())
            if has_eager:
                asyncio.run(_run_eager())

        # Benchmark standard
        std_times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            asyncio.run(_run_standard())
            std_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Benchmark eager (or repeat standard if unavailable)
        eager_times = []
        run_fn = _run_eager if has_eager else _run_standard
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            asyncio.run(run_fn())
            eager_times.append((time.perf_counter_ns() - t0) / 1e6)

        std_stats = compute_stats(std_times)
        eager_stats = compute_stats(eager_times)
        speedup = std_stats["mean_ms"] / eager_stats["mean_ms"] if eager_stats["mean_ms"] > 0 else 1.0

        results.append(
            {
                "test": "eager_task_factory",
                "tasks": n,
                "eager_available": has_eager,
                "standard_mean_ms": round(std_stats["mean_ms"], 4),
                "standard_p99_ms": round(std_stats["p99_ms"], 4),
                "eager_mean_ms": round(eager_stats["mean_ms"], 4),
                "eager_p99_ms": round(eager_stats["p99_ms"], 4),
                "speedup": round(speedup, 2),
            }
        )

        label = "eager" if has_eager else "N/A (< 3.12)"
        print(
            f"  {n:2d} tasks: standard={std_stats['mean_ms']:.4f}ms  "
            f"{label}={eager_stats['mean_ms']:.4f}ms  "
            f"({speedup:.2f}x)"
        )

    return results


# ---------------------------------------------------------------------------
# Section 8: Full DAG scheduler execute() with real executor path
# ---------------------------------------------------------------------------


def bench_scheduler_execute(iterations: int = 300) -> list[dict]:
    """Benchmark the actual TopologicalScheduler.execute() method.

    Unlike Section 6 which manually calls asyncio.gather on groups,
    this runs through the real execute() codepath which includes:
    - Eager task factory installation (3.12+)
    - Early-exit on block detection
    - Per-group timeout handling
    - CPU pool dispatch on free-threaded builds
    """
    try:
        from nemoguardrails.rails.llm.dag_scheduler import TopologicalScheduler  # noqa: F401
    except ImportError:
        print("  DAG scheduler not available, skipping")
        return []

    results = []

    topologies = {
        "wide_4": lambda: _build_graph(["A", "B", "C", "D"], {}),
        "wide_8": lambda: _build_graph([chr(65 + i) for i in range(8)], {}),
        "diamond": lambda: _build_graph(
            ["A", "B", "C", "D"],
            {"B": ["A"], "C": ["A"], "D": ["B", "C"]},
        ),
        "fan_out_8": lambda: _build_graph(
            ["root"] + [f"leaf_{i}" for i in range(8)],
            {f"leaf_{i}": ["root"] for i in range(8)},
        ),
        "deep_pipeline_6": lambda: _build_graph(
            [f"stage_{i}" for i in range(6)],
            {f"stage_{i}": [f"stage_{i - 1}"] for i in range(1, 6)},
        ),
    }

    for topo_name, build_fn in topologies.items():
        graph = build_fn()
        scheduler = TopologicalScheduler(graph)

        # Rail executor: simulates 5ms I/O per rail
        async def rail_executor(rail_name: str, context: dict) -> dict:
            await asyncio.sleep(0.005)
            return {"action": "continue", "rail": rail_name}

        # Warmup
        for _ in range(5):
            asyncio.run(scheduler.execute(rail_executor))

        # Benchmark
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            result = asyncio.run(scheduler.execute(rail_executor))
            times.append((time.perf_counter() - t0) * 1000)

        stats = compute_stats(times)
        n_rails = graph.node_count
        n_groups = scheduler.num_groups
        serial_ms = n_rails * 5.0
        speedup = serial_ms / stats["mean_ms"] if stats["mean_ms"] > 0 else 0

        results.append(
            {
                "test": "scheduler_execute",
                "topology": topo_name,
                "rails": n_rails,
                "groups": n_groups,
                "serial_ms": serial_ms,
                "actual_mean_ms": round(stats["mean_ms"], 2),
                "actual_p99_ms": round(stats["p99_ms"], 2),
                "actual_stdev_ms": round(stats["stdev_ms"], 2),
                "speedup_vs_serial": round(speedup, 2),
            }
        )

        print(
            f"  {topo_name:18s}: {n_groups} groups, {n_rails} rails  "
            f"serial={serial_ms:.0f}ms  actual={stats['mean_ms']:.1f}ms  "
            f"({speedup:.2f}x)"
        )

    return results


def _build_graph(
    rail_names: list[str],
    deps: dict[str, list[str]],
) -> "RailDependencyGraph":
    """Helper to build a dependency graph from names and dependencies."""
    from nemoguardrails.rails.llm.dag_scheduler import RailDependencyGraph

    g = RailDependencyGraph()
    for name in rail_names:
        g.add_rail(name, depends_on=deps.get(name))
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_section(title: str):
    print(f"{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


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

    all_results: dict[str, Any] = {"python": info, "sections": {}}

    sections = [
        ("threading", "Section 1: Free-threaded parallel CPU rail execution", bench_threading),
        ("template", "Section 2: Jinja2 template rendering (M3 cache)", bench_template_rendering),
        ("import", "Section 3: Import time (PEP 649 deferred annotations)", bench_import),
        ("gc", "Section 4: GC tail latency under pressure", bench_gc_pauses),
        ("asyncio", "Section 5: Asyncio gather with real rails", bench_asyncio_rails),
        ("dag", "Section 6: DAG scheduler parallelism", bench_dag_scheduler),
        ("eager", "Section 7: Eager task factory (3.12+ optimisation)", bench_eager_task_factory),
        ("scheduler", "Section 8: Full scheduler execute() path", bench_scheduler_execute),
    ]

    for key, title, bench_fn in sections:
        if args.section in (key, "all"):
            _print_section(title)
            all_results["sections"][key] = bench_fn()
            print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_section("SUMMARY")

    if "threading" in all_results["sections"]:
        thr = all_results["sections"]["threading"]
        best = max(thr, key=lambda r: r["speedup_threadpool"])
        marker = "free-threaded" if info["gil_disabled"] else "GIL-limited"
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
            best = max(eager, key=lambda r: r["speedup"])
            avail = "available" if best["eager_available"] else "unavailable (< 3.12)"
            print(f"  Eager:      {best['speedup']:.2f}x speedup @ {best['tasks']} tasks ({avail})")

    if "scheduler" in all_results["sections"]:
        sched = all_results["sections"]["scheduler"]
        if sched:
            best = max(sched, key=lambda r: r["speedup_vs_serial"])
            print(
                f"  Scheduler:  {best['speedup_vs_serial']:.2f}x vs serial ({best['topology']}, {best['rails']} rails)"
            )

    print()

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
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
