#!/usr/bin/env python3
"""Benchmark: GIL vs free-threaded Python for parallel guardrail checks.

Measures the performance difference between sequential and threaded
execution of CPU-bound guardrail-like workloads.  On standard Python
(with GIL), threads cannot achieve true parallelism for CPU work.
On Python 3.14t (free-threaded / no-GIL), threads run in parallel.

Usage:
    python benchmarks/bench_threading.py

Environment variables:
    BENCH_ITERATIONS  - number of iterations per rail check (default: 200000)
    BENCH_RAIL_COUNT  - number of parallel rail checks (default: 4)
"""

import json
import os
import statistics
import sys
import sysconfig
import threading
import time

ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "200000"))
RAIL_COUNT = int(os.environ.get("BENCH_RAIL_COUNT", "4"))
WARMUP_ROUNDS = 2
BENCH_ROUNDS = 5


def simulated_rail_check(iterations: int) -> float:
    """Simulate a CPU-bound guardrail check.

    Performs regex-like pattern matching and string processing
    typical of input/output rail evaluation.
    """
    total = 0.0
    text = "The user asked about financial advice and investment strategies for retirement planning"
    patterns = ["financial", "investment", "password", "credit card", "ssn"]

    for i in range(iterations):
        # Simulate pattern matching (CPU-bound)
        for pattern in patterns:
            if pattern in text:
                total += len(pattern)

        # Simulate token counting
        tokens = text.split()
        total += len(tokens)

        # Simulate hash-based deduplication check
        h = hash(text + str(i))
        total += h % 100

    return total


def run_sequential(rail_count: int, iterations: int) -> float:
    """Run rail checks sequentially."""
    start = time.perf_counter()
    results = []
    for _ in range(rail_count):
        results.append(simulated_rail_check(iterations))
    elapsed = time.perf_counter() - start
    return elapsed


def run_threaded(rail_count: int, iterations: int) -> float:
    """Run rail checks in parallel threads."""
    results = [None] * rail_count
    threads = []

    def worker(idx):
        results[idx] = simulated_rail_check(iterations)

    start = time.perf_counter()
    for i in range(rail_count):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start
    return elapsed


def main():
    gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    build_type = "free-threaded (no-GIL)" if gil_disabled else "standard (with GIL)"

    print(f"Python {sys.version}")
    print(f"Build: {build_type}")
    print(f"Iterations per rail: {ITERATIONS:,}")
    print(f"Parallel rail count: {RAIL_COUNT}")
    print(f"Benchmark rounds: {BENCH_ROUNDS}")
    print()

    # Warmup
    for _ in range(WARMUP_ROUNDS):
        run_sequential(1, ITERATIONS // 10)
        run_threaded(1, ITERATIONS // 10)

    # Benchmark sequential
    seq_times = []
    for _ in range(BENCH_ROUNDS):
        seq_times.append(run_sequential(RAIL_COUNT, ITERATIONS))

    # Benchmark threaded
    thr_times = []
    for _ in range(BENCH_ROUNDS):
        thr_times.append(run_threaded(RAIL_COUNT, ITERATIONS))

    seq_mean = statistics.mean(seq_times)
    seq_stdev = statistics.stdev(seq_times) if len(seq_times) > 1 else 0
    thr_mean = statistics.mean(thr_times)
    thr_stdev = statistics.stdev(thr_times) if len(thr_times) > 1 else 0
    speedup = seq_mean / thr_mean if thr_mean > 0 else 0

    print(f"Sequential:  {seq_mean:.4f}s +/- {seq_stdev:.4f}s")
    print(f"Threaded:    {thr_mean:.4f}s +/- {thr_stdev:.4f}s")
    print(f"Speedup:     {speedup:.2f}x")
    print()

    if gil_disabled:
        print(f"With {RAIL_COUNT} rails, free-threaded Python achieves ~{speedup:.1f}x speedup")
        if speedup > 1.5:
            print("True parallelism confirmed — threads run on separate cores")
        else:
            print("Warning: speedup lower than expected, check CPU core count")
    else:
        print(f"With GIL, threading speedup is ~{speedup:.1f}x (expected ~1.0x for CPU-bound work)")
        if speedup < 1.3:
            print("Confirmed: GIL prevents parallel execution of CPU-bound threads")

    # Output JSON for CI
    result = {
        "python_version": sys.version,
        "build": build_type,
        "gil_disabled": bool(gil_disabled),
        "iterations": ITERATIONS,
        "rail_count": RAIL_COUNT,
        "sequential_mean_s": round(seq_mean, 4),
        "sequential_stdev_s": round(seq_stdev, 4),
        "threaded_mean_s": round(thr_mean, 4),
        "threaded_stdev_s": round(thr_stdev, 4),
        "speedup": round(speedup, 2),
    }
    print()
    print("JSON:", json.dumps(result))


if __name__ == "__main__":
    main()
