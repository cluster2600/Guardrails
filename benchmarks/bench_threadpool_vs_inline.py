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

"""Benchmark: thread-pool dispatch vs inline execution for CPU-bound rail actions.

Compares three execution strategies for CPU-bound guardrail checks:

1. **Inline** -- direct ``await`` of the async wrapper, which blocks the
   event loop for the duration of the CPU work.
2. **Thread pool** -- ``loop.run_in_executor()`` dispatches the synchronous
   scan to the default thread-pool executor.
3. **@cpu_bound decorator** -- uses :class:`RailThreadPool` from
   ``nemoguardrails.rails.llm.thread_pool`` to dispatch decorated sync
   functions.

Each mode is exercised with varying parallelism (1, 2, 4, 8 concurrent
rails) and CPU intensity (100, 500, 2 000 hashing rounds).

Usage:
    python benchmarks/bench_threadpool_vs_inline.py
    python benchmarks/bench_threadpool_vs_inline.py --output results.json

Environment variables:
    BENCH_WARMUP   -- number of warmup rounds (default: 2)
    BENCH_ROUNDS   -- number of measurement rounds per scenario (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Sequence

# -- project imports ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.conftest import PerfResult, compute_stats, write_results  # noqa: E402
from benchmarks.fake_rails import (  # noqa: E402
    cpu_bound_rail,
    cpu_bound_rail_threaded,
    cpu_bound_scan,
)
from benchmarks.scenarios import THROUGHPUT_THREADPOOL, BenchmarkScenario  # noqa: E402
from nemoguardrails.rails.llm.thread_pool import (  # noqa: E402
    RailThreadPool,
    cpu_bound,
    is_free_threaded,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = int(os.environ.get("BENCH_WARMUP", "2"))
BENCH_ROUNDS = int(os.environ.get("BENCH_ROUNDS", "5"))
SAMPLE_TEXT = (
    "My social security number is 123-45-6789 and my email is user@example.com. "
    "Can you help me write a cover letter for a software engineering position?"
)

# ---------------------------------------------------------------------------
# @cpu_bound decorated variant of cpu_bound_scan
# ---------------------------------------------------------------------------


@cpu_bound
def cpu_bound_scan_decorated(text: str, rounds: int = 500) -> dict:
    """Identical to ``cpu_bound_scan`` but decorated with ``@cpu_bound``."""
    return cpu_bound_scan(text, rounds)


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


async def run_inline(text: str, num_rails: int, rounds: int) -> float:
    """Run *num_rails* CPU-bound rails inline (blocking the event loop).

    All rails are launched as concurrent tasks, but because
    ``cpu_bound_rail`` calls the sync scan directly inside the coroutine
    the event loop is blocked serially.
    """
    start = time.perf_counter()
    tasks = [asyncio.create_task(cpu_bound_rail(text, rounds)) for _ in range(num_rails)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start


async def run_threadpool(text: str, num_rails: int, rounds: int) -> float:
    """Run *num_rails* CPU-bound rails via ``loop.run_in_executor()``."""
    start = time.perf_counter()
    tasks = [asyncio.create_task(cpu_bound_rail_threaded(text, rounds)) for _ in range(num_rails)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start


async def run_decorator(text: str, num_rails: int, rounds: int, pool: RailThreadPool) -> float:
    """Run *num_rails* CPU-bound rails via the ``@cpu_bound`` / RailThreadPool path."""
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(pool.dispatch(cpu_bound_scan_decorated, text, rounds=rounds)) for _ in range(num_rails)
    ]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Single-scenario runner
# ---------------------------------------------------------------------------

MODES = ("inline", "threadpool", "decorator")


async def bench_scenario(scenario: BenchmarkScenario) -> list[PerfResult]:
    """Run all three execution modes for *scenario* and return PerfResults."""
    num_rails = scenario.num_rails
    rounds = scenario.cpu_work_units
    iterations = scenario.iterations

    pool = RailThreadPool(max_workers=max(num_rails, 4))
    results: list[PerfResult] = []

    for mode in MODES:
        # -- warmup ----------------------------------------------------------
        for _ in range(WARMUP_ROUNDS):
            if mode == "inline":
                await run_inline(SAMPLE_TEXT, num_rails, rounds)
            elif mode == "threadpool":
                await run_threadpool(SAMPLE_TEXT, num_rails, rounds)
            else:
                await run_decorator(SAMPLE_TEXT, num_rails, rounds, pool)

        # -- measurement -----------------------------------------------------
        latencies_ms: list[float] = []
        wall_start = time.perf_counter()
        for _ in range(iterations):
            if mode == "inline":
                elapsed = await run_inline(SAMPLE_TEXT, num_rails, rounds)
            elif mode == "threadpool":
                elapsed = await run_threadpool(SAMPLE_TEXT, num_rails, rounds)
            else:
                elapsed = await run_decorator(SAMPLE_TEXT, num_rails, rounds, pool)
            latencies_ms.append(elapsed * 1000.0)
        wall_s = time.perf_counter() - wall_start

        stats = compute_stats(latencies_ms)
        rps = iterations / wall_s if wall_s > 0 else 0.0

        results.append(
            PerfResult(
                scenario=f"{scenario.name}__{mode}",
                count=stats.get("count", iterations),
                p50_ms=stats.get("p50_ms", 0.0),
                p95_ms=stats.get("p95_ms", 0.0),
                p99_ms=stats.get("p99_ms", 0.0),
                mean_ms=stats.get("mean_ms", 0.0),
                stdev_ms=stats.get("stdev_ms", 0.0),
                rps=rps,
                wall_time_s=wall_s,
                extra={
                    "mode": mode,
                    "num_rails": num_rails,
                    "cpu_rounds": rounds,
                    "gil_disabled": is_free_threaded(),
                    "pool_workers": pool.max_workers,
                },
            )
        )

    pool.shutdown(wait=True)
    return results


# ---------------------------------------------------------------------------
# Baseline for speedup calculation
# ---------------------------------------------------------------------------


def _compute_speedup(results: list[PerfResult]) -> dict[str, float]:
    """Compute speedup of threadpool and decorator modes vs inline baseline.

    Returns a mapping of ``scenario_name -> speedup`` for every non-inline
    result.  Speedup is defined as ``inline_mean / mode_mean``.
    """
    # Group by scenario prefix (everything before ``__<mode>``)
    inline_by_prefix: dict[str, float] = {}
    for r in results:
        if r.scenario.endswith("__inline"):
            prefix = r.scenario.rsplit("__", 1)[0]
            inline_by_prefix[prefix] = r.mean_ms

    speedups: dict[str, float] = {}
    for r in results:
        if r.scenario.endswith("__inline"):
            speedups[r.scenario] = 1.0
            continue
        prefix = r.scenario.rsplit("__", 1)[0]
        baseline = inline_by_prefix.get(prefix)
        if baseline and r.mean_ms > 0:
            speedups[r.scenario] = round(baseline / r.mean_ms, 2)
        else:
            speedups[r.scenario] = 0.0
    return speedups


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------


def print_summary(results: list[PerfResult]) -> None:
    """Print a human-readable summary table to stdout."""
    gil_status = "DISABLED (free-threaded)" if is_free_threaded() else "ENABLED"
    print()
    print("=" * 88)
    print("  Thread-pool vs Inline Benchmark Summary")
    print(f"  Python {sys.version}")
    print(f"  GIL: {gil_status}")
    print("=" * 88)
    print()

    speedups = _compute_speedup(results)

    header = f"{'Scenario':<40} {'Mean ms':>9} {'P95 ms':>9} {'RPS':>9} {'Speedup':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        sp = speedups.get(r.scenario, 0.0)
        sp_str = f"{sp:.2f}x"
        print(f"{r.scenario:<40} {r.mean_ms:>9.2f} {r.p95_ms:>9.2f} {r.rps:>9.1f} {sp_str:>9}")

    print()
    if is_free_threaded():
        print("NOTE: Free-threaded build -- thread-pool and decorator modes can achieve real CPU parallelism.")
    else:
        print(
            "NOTE: Standard GIL build -- thread-pool dispatch still avoids "
            "event-loop starvation but cannot speed up pure-Python CPU work."
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark thread-pool dispatch vs inline execution.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write JSON results (default: stdout only).",
    )
    parser.add_argument(
        "--scenarios",
        "-s",
        nargs="*",
        default=None,
        help="Run only scenarios whose name contains one of these substrings.",
    )
    return parser.parse_args()


async def async_main(
    scenarios: Sequence[BenchmarkScenario],
    name_filters: list[str] | None = None,
) -> list[PerfResult]:
    """Run all (or filtered) scenarios and return aggregated results."""
    all_results: list[PerfResult] = []

    for sc in scenarios:
        if name_filters and not any(f in sc.name for f in name_filters):
            continue
        print(
            f">>> Running scenario: {sc.name}  "
            f"(rails={sc.num_rails}, rounds={sc.cpu_work_units}, "
            f"iters={sc.iterations})"
        )
        sc_results = await bench_scenario(sc)
        all_results.extend(sc_results)

    return all_results


def main() -> None:
    args = parse_args()

    results = asyncio.run(async_main(THROUGHPUT_THREADPOOL, name_filters=args.scenarios))

    if not results:
        print("No scenarios matched. Nothing to report.", file=sys.stderr)
        sys.exit(1)

    print_summary(results)
    print()
    print("--- JSON output ---")
    write_results(results, output=args.output)


if __name__ == "__main__":
    main()
