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

"""Benchmark regression checker.

Compares current benchmark results against a baseline and fails if
any metric exceeds the allowed regression threshold.

Usage:
    python -m benchmarks.compare \\
        --baseline perf_baselines/linux-py314.json \\
        --current out.json \\
        --max-p95-regression 5 \\
        --max-first-token-regression-ms 20

Exit codes:
    0 — all checks passed
    1 — one or more regressions exceeded thresholds
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict] | dict:
    """Load benchmark results from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Normalise: single result → list
    if isinstance(data, dict):
        return [data]
    return data


def index_by_scenario(results: list[dict]) -> dict[str, dict]:
    """Index results by scenario name for easy lookup."""
    return {r["scenario"]: r for r in results if "scenario" in r}


def check_regression(
    baseline: dict,
    current: dict,
    max_p95_pct: float,
    max_ftl_ms: float,
    max_mean_pct: float,
) -> list[str]:
    """Check a single scenario for regressions.

    Returns a list of failure messages (empty = passed).
    """
    failures = []
    scenario = current.get("scenario", "unknown")

    # p95 latency regression
    b_p95 = baseline.get("p95_ms", 0)
    c_p95 = current.get("p95_ms", 0)
    if b_p95 > 0:
        pct = ((c_p95 - b_p95) / b_p95) * 100
        if pct > max_p95_pct:
            failures.append(
                f"[{scenario}] p95 regression: {b_p95:.1f}ms → {c_p95:.1f}ms ({pct:+.1f}%, threshold: {max_p95_pct}%)"
            )

    # First-token latency regression
    b_ftl = baseline.get("first_token_ms", 0)
    c_ftl = current.get("first_token_ms", 0)
    if b_ftl > 0:
        delta = c_ftl - b_ftl
        if delta > max_ftl_ms:
            failures.append(
                f"[{scenario}] first-token regression: {b_ftl:.1f}ms → {c_ftl:.1f}ms "
                f"({delta:+.1f}ms, threshold: {max_ftl_ms}ms)"
            )

    # Mean latency regression
    b_mean = baseline.get("mean_ms", 0)
    c_mean = current.get("mean_ms", 0)
    if b_mean > 0:
        pct = ((c_mean - b_mean) / b_mean) * 100
        if pct > max_mean_pct:
            failures.append(
                f"[{scenario}] mean regression: {b_mean:.1f}ms → {c_mean:.1f}ms "
                f"({pct:+.1f}%, threshold: {max_mean_pct}%)"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Benchmark regression checker")
    parser.add_argument("--baseline", required=True, help="Baseline results JSON")
    parser.add_argument("--current", required=True, help="Current results JSON")
    parser.add_argument(
        "--max-p95-regression", type=float, default=5.0, help="Max allowed p95 regression in %% (default: 5)"
    )
    parser.add_argument(
        "--max-first-token-regression-ms",
        type=float,
        default=20.0,
        help="Max allowed first-token regression in ms (default: 20)",
    )
    parser.add_argument(
        "--max-mean-regression", type=float, default=10.0, help="Max allowed mean regression in %% (default: 10)"
    )
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"Baseline file not found: {args.baseline}")
        print("Skipping regression check (no baseline to compare against)")
        sys.exit(0)

    baseline_results = index_by_scenario(load_results(args.baseline))
    current_results = index_by_scenario(load_results(args.current))

    all_failures: list[str] = []
    checked = 0
    skipped = 0

    for scenario_name, current in current_results.items():
        if scenario_name not in baseline_results:
            skipped += 1
            continue

        baseline = baseline_results[scenario_name]
        failures = check_regression(
            baseline,
            current,
            max_p95_pct=args.max_p95_regression,
            max_ftl_ms=args.max_first_token_regression_ms,
            max_mean_pct=args.max_mean_regression,
        )
        all_failures.extend(failures)
        checked += 1

    print(f"Checked {checked} scenarios, skipped {skipped} (no baseline)")
    print()

    if all_failures:
        print("REGRESSION FAILURES:")
        for f in all_failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("All checks passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
