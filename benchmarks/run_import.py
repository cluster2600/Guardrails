#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import / cold-start benchmark.

Measures the time to import nemoguardrails and create an LLMRails
instance.  This is a key metric for serverless/FaaS deployments.

Usage:
    python -m benchmarks.run_import [--output out.json]
"""

import argparse
import json
import statistics
import subprocess
import sys
import time


# We measure import time by spawning fresh Python processes to avoid
# module caching.  Each subprocess imports nemoguardrails and prints
# the elapsed time.

_IMPORT_SCRIPT = """
import time
t0 = time.perf_counter()
try:
    import nemoguardrails
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"OK {elapsed:.2f}")
except Exception as e:
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"ERR {elapsed:.2f} {e}")
"""

_COLD_START_SCRIPT = """
import time
t0 = time.perf_counter()
try:
    from nemoguardrails.rails.llm.config import RailsConfig
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"OK {elapsed:.2f}")
except Exception as e:
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"ERR {elapsed:.2f} {e}")
"""


def measure_subprocess(script: str, rounds: int = 5) -> dict:
    """Run a Python snippet in a fresh process and collect timings."""
    timings = []
    errors = []

    for _ in range(rounds):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        if output.startswith("OK "):
            timings.append(float(output.split()[1]))
        elif output.startswith("ERR "):
            parts = output.split(maxsplit=2)
            timings.append(float(parts[1]))
            errors.append(parts[2] if len(parts) > 2 else "unknown")

    return {
        "timings_ms": timings,
        "errors": errors,
        "mean_ms": statistics.mean(timings) if timings else 0,
        "stdev_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
        "min_ms": min(timings) if timings else 0,
        "max_ms": max(timings) if timings else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Import / cold-start benchmark")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    print(f"Python {sys.version}\n")

    print("Measuring: import nemoguardrails ...")
    import_stats = measure_subprocess(_IMPORT_SCRIPT, rounds=args.rounds)
    print(f"  mean={import_stats['mean_ms']:.0f}ms  min={import_stats['min_ms']:.0f}ms  max={import_stats['max_ms']:.0f}ms")
    if import_stats["errors"]:
        print(f"  errors: {import_stats['errors'][:3]}")

    print("\nMeasuring: import RailsConfig ...")
    config_stats = measure_subprocess(_COLD_START_SCRIPT, rounds=args.rounds)
    print(f"  mean={config_stats['mean_ms']:.0f}ms  min={config_stats['min_ms']:.0f}ms  max={config_stats['max_ms']:.0f}ms")
    if config_stats["errors"]:
        print(f"  errors: {config_stats['errors'][:3]}")

    result = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "import_nemoguardrails": import_stats,
        "import_railsconfig": config_stats,
    }

    blob = json.dumps(result, indent=2)
    print(f"\n{blob}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(blob + "\n")


if __name__ == "__main__":
    main()
