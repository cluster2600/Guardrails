#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance dashboard — compare benchmark results against baselines.

Reads JSON results from benchmark runs and baseline files, compares each
scenario's p95_ms against its baseline, and outputs a formatted markdown
table showing regressions and improvements.

Usage:
    python -m benchmarks.dashboard \\
        --results /tmp/perf-latency.json \\
        --baseline perf_baselines/linux-py312.json

    python -m benchmarks.dashboard \\
        --results /tmp/perf-latency.json \\
        --baseline perf_baselines/linux-py314.json \\
        --output report.md

Exit codes:
    0 — all scenarios within threshold
    1 — one or more scenarios exceeded their regression threshold
"""

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> list[dict]:
    """Load and normalise a JSON results file (single object or array)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def index_by_scenario(records: list[dict]) -> dict[str, dict]:
    """Index a list of result dicts by their 'scenario' key."""
    return {r["scenario"]: r for r in records if "scenario" in r}


def compute_delta_pct(baseline_val: float, actual_val: float) -> float:
    """Compute percentage change from baseline to actual.

    Positive = regression (slower), negative = improvement (faster).
    """
    if baseline_val <= 0:
        return 0.0
    return ((actual_val - baseline_val) / baseline_val) * 100.0


def format_status(delta_pct: float, threshold_pct: float) -> str:
    """Return a pass/fail status indicator."""
    if delta_pct > threshold_pct:
        return "FAIL"
    if delta_pct > 0:
        return "WARN"
    return "PASS"


def build_comparison_rows(
    baseline_idx: dict[str, dict],
    results_idx: dict[str, dict],
) -> list[dict]:
    """Build comparison rows for every scenario found in results.

    Returns a list of row dicts with: scenario, baseline_p95, actual_p95,
    delta_pct, threshold_pct, status, and optional memory fields.
    """
    rows: list[dict] = []

    for scenario_name, result in sorted(results_idx.items()):
        baseline = baseline_idx.get(scenario_name)
        if baseline is None:
            rows.append({
                "scenario": scenario_name,
                "baseline_p95": None,
                "actual_p95": result.get("p95_ms", 0.0),
                "delta_pct": 0.0,
                "threshold_pct": 20,
                "status": "SKIP",
                "rss_delta": result.get("rss_mb_delta"),
                "baseline_rss_delta": None,
            })
            continue

        baseline_p95 = baseline.get("p95_ms", 0.0)
        actual_p95 = result.get("p95_ms", 0.0)
        threshold = baseline.get("threshold_pct", 20)
        delta = compute_delta_pct(baseline_p95, actual_p95)
        status = format_status(delta, threshold)

        rows.append({
            "scenario": scenario_name,
            "baseline_p95": baseline_p95,
            "actual_p95": actual_p95,
            "delta_pct": delta,
            "threshold_pct": threshold,
            "status": status,
            "rss_delta": result.get("rss_mb_delta"),
            "baseline_rss_delta": baseline.get("rss_mb_delta"),
        })

    return rows


def render_markdown(rows: list[dict], results_path: str, baseline_path: str) -> str:
    """Render comparison rows as a markdown report."""
    lines: list[str] = []

    lines.append("# Performance Dashboard")
    lines.append("")
    lines.append(f"**Results:** `{results_path}`")
    lines.append(f"**Baseline:** `{baseline_path}`")
    lines.append("")

    # Summary counts
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_warn = sum(1 for r in rows if r["status"] == "WARN")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    lines.append(f"**Summary:** {n_pass} passed, {n_warn} warnings, {n_fail} failed, {n_skip} skipped")
    lines.append("")

    # Main latency table
    lines.append("## Latency Comparison (p95)")
    lines.append("")
    lines.append("| Scenario | Baseline p95 (ms) | Actual p95 (ms) | Delta % | Threshold % | Status |")
    lines.append("|----------|-------------------:|----------------:|--------:|------------:|--------|")

    for row in rows:
        scenario = row["scenario"]
        if row["status"] == "SKIP":
            bl_str = "N/A"
            delta_str = "N/A"
        else:
            bl_str = f"{row['baseline_p95']:.1f}"
            delta_str = f"{row['delta_pct']:+.1f}"

        actual_str = f"{row['actual_p95']:.1f}"
        threshold_str = f"{row['threshold_pct']}"

        status = row["status"]

        lines.append(
            f"| {scenario} | {bl_str} | {actual_str} | {delta_str} | {threshold_str} | {status} |"
        )

    # If any rows have meaningful memory data, add a memory section
    memory_rows = [r for r in rows if r.get("rss_delta") is not None and r["rss_delta"] != 0.0]
    if memory_rows:
        lines.append("")
        lines.append("## Memory (RSS Delta)")
        lines.append("")
        lines.append("| Scenario | Baseline RSS Delta (MiB) | Actual RSS Delta (MiB) | Status |")
        lines.append("|----------|-------------------------:|-----------------------:|--------|")
        for row in memory_rows:
            scenario = row["scenario"]
            bl_rss = row.get("baseline_rss_delta")
            act_rss = row.get("rss_delta", 0.0)
            bl_str = f"{bl_rss:.1f}" if bl_rss is not None else "N/A"
            act_str = f"{act_rss:.1f}" if act_rss is not None else "N/A"
            # Memory regression check: fail if >threshold_pct above baseline
            if bl_rss and bl_rss > 0 and act_rss is not None:
                mem_delta_pct = compute_delta_pct(bl_rss, act_rss)
                mem_status = format_status(mem_delta_pct, row["threshold_pct"])
            else:
                mem_status = "SKIP"
            lines.append(f"| {scenario} | {bl_str} | {act_str} | {mem_status} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Performance dashboard: compare results against baselines"
    )
    parser.add_argument(
        "--results", "-r", required=True,
        help="Path to benchmark results JSON file (single or array)"
    )
    parser.add_argument(
        "--baseline", "-b", required=True,
        help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write markdown report to this file (default: stdout only)"
    )
    parser.add_argument(
        "--threshold-override", type=float, default=None,
        help="Override per-scenario threshold_pct with a global value"
    )
    args = parser.parse_args()

    # --- Load data ---
    if not Path(args.baseline).exists():
        print(f"Baseline file not found: {args.baseline}", file=sys.stderr)
        print("Skipping dashboard (no baseline to compare against)")
        return 0

    if not Path(args.results).exists():
        print(f"Results file not found: {args.results}", file=sys.stderr)
        return 1

    baseline_records = load_json(args.baseline)
    results_records = load_json(args.results)

    # Apply global threshold override if requested
    if args.threshold_override is not None:
        for rec in baseline_records:
            rec["threshold_pct"] = args.threshold_override

    baseline_idx = index_by_scenario(baseline_records)
    results_idx = index_by_scenario(results_records)

    # --- Compare ---
    rows = build_comparison_rows(baseline_idx, results_idx)

    if not rows:
        print("No scenarios found in results file.")
        return 0

    # --- Render ---
    report = render_markdown(rows, args.results, args.baseline)
    print(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n")
        print(f"\nReport written to {args.output}", file=sys.stderr)

    # --- Exit code ---
    has_failure = any(r["status"] == "FAIL" for r in rows)
    if has_failure:
        print("\nOne or more scenarios exceeded their regression threshold.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
