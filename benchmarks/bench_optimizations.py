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

"""Micro-benchmarks for the P1/P2 performance optimisations.

Run:  python benchmarks/bench_optimizations.py
"""

import statistics
import time
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bench(fn, iterations=10_000, warmup=100, label=""):
    """Run *fn* for *iterations* and print timing stats."""
    # warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)

    mean_ns = statistics.mean(times)
    median_ns = statistics.median(times)
    p99_ns = sorted(times)[int(len(times) * 0.99)]
    total_ms = sum(times) / 1e6

    print(f"  {label}")
    print(f"    iterations : {iterations:,}")
    print(f"    total      : {total_ms:,.2f} ms")
    print(f"    mean       : {mean_ns:,.0f} ns")
    print(f"    median     : {median_ns:,.0f} ns")
    print(f"    p99        : {p99_ns:,.0f} ns")
    print()
    return mean_ns


# ---------------------------------------------------------------------------
# 1. _LRUDict vs plain dict vs OrderedDict
# ---------------------------------------------------------------------------


def bench_lru_dict():
    from nemoguardrails.rails.llm.llmrails import _LRUDict

    print("=" * 60)
    print("1. _LRUDict vs plain dict vs OrderedDict  (10k inserts)")
    print("=" * 60)

    SIZE = 256

    # --- _LRUDict ---
    lru = _LRUDict(maxsize=SIZE)

    def lru_insert():
        for i in range(1000):
            lru[f"key_{i}"] = i

    t_lru = bench(lru_insert, iterations=100, warmup=5, label="_LRUDict (maxsize=256, 1k inserts/iter)")

    # --- plain dict (unbounded) ---
    plain = {}

    def plain_insert():
        for i in range(1000):
            plain[f"key_{i}"] = i

    t_plain = bench(plain_insert, iterations=100, warmup=5, label="plain dict (unbounded, 1k inserts/iter)")

    # --- OrderedDict with manual eviction ---
    od = OrderedDict()

    def od_insert():
        for i in range(1000):
            od[f"key_{i}"] = i
            if len(od) > SIZE:
                od.popitem(last=False)

    t_od = bench(od_insert, iterations=100, warmup=5, label="OrderedDict + popitem (maxsize=256, 1k inserts/iter)")

    print(f"  Summary: _LRUDict is {t_plain / t_lru:.1f}x vs plain dict, {t_od / t_lru:.1f}x vs OrderedDict")
    print()


# ---------------------------------------------------------------------------
# 2. Action name normalisation: cached vs uncached
# ---------------------------------------------------------------------------


def bench_action_name_cache():
    from nemoguardrails.actions.action_dispatcher import ActionDispatcher

    print("=" * 60)
    print("2. Action name normalisation: cached vs uncached")
    print("=" * 60)

    d = ActionDispatcher(load_all_actions=False)
    for i in range(50):
        d.register_action(lambda **kw: None, name=f"action_{i}")

    names = [f"action_{i}" for i in range(50)] + [f"Action{i}" for i in range(50)]

    # --- Cached (normal path) ---
    def cached_lookup():
        for n in names:
            d._normalize_action_name(n)

    t_cached = bench(
        cached_lookup, iterations=5_000, warmup=100, label="cached _normalize_action_name (100 names/iter)"
    )

    # --- Uncached (clear cache each time) ---
    def uncached_lookup():
        d._normalised_names.clear()
        for n in names:
            d._normalize_action_name(n)

    t_uncached = bench(
        uncached_lookup, iterations=5_000, warmup=100, label="uncached _normalize_action_name (100 names/iter)"
    )

    print(f"  Summary: cache speedup = {t_uncached / t_cached:.1f}x")
    print()


# ---------------------------------------------------------------------------
# 3. Jinja2 template caching: cached vs uncached
# ---------------------------------------------------------------------------


def bench_template_cache():
    from nemoguardrails.llm.taskmanager import LLMTaskManager
    from nemoguardrails.rails.llm.config import RailsConfig

    print("=" * 60)
    print("3. Jinja2 template compilation: cached vs uncached")
    print("=" * 60)

    config = RailsConfig.from_content(yaml_content="models: []")
    tm = LLMTaskManager(config)

    templates = [
        "Hello {{ name }}, welcome to {{ place }}!",
        "The {{ animal }} jumped over the {{ object }}.",
        "{% for item in items %}{{ item }}{% endfor %}",
        "{{ greeting | upper }}, {{ name }}!",
        "{% if condition %}yes{% else %}no{% endif %}",
    ]

    # --- Cached (normal path) ---
    def cached_compile():
        for t in templates:
            tm._get_compiled_template(t)

    t_cached = bench(
        cached_compile, iterations=10_000, warmup=100, label="cached _get_compiled_template (5 templates/iter)"
    )

    # --- Uncached (clear cache each time) ---
    def uncached_compile():
        tm._template_cache.clear()
        for t in templates:
            tm._get_compiled_template(t)

    t_uncached = bench(
        uncached_compile, iterations=10_000, warmup=100, label="uncached _get_compiled_template (5 templates/iter)"
    )

    print(f"  Summary: cache speedup = {t_uncached / t_cached:.1f}x")
    print()

    # --- Variable extraction ---
    print("-" * 60)
    print("3b. Template variable extraction: cached vs uncached")
    print("-" * 60)

    def cached_vars():
        for t in templates:
            tm._get_template_variables(t)

    t_cv = bench(cached_vars, iterations=10_000, warmup=100, label="cached _get_template_variables (5 templates/iter)")

    def uncached_vars():
        tm._variables_cache.clear()
        for t in templates:
            tm._get_template_variables(t)

    t_uv = bench(
        uncached_vars, iterations=10_000, warmup=100, label="uncached _get_template_variables (5 templates/iter)"
    )

    print(f"  Summary: cache speedup = {t_uv / t_cv:.1f}x")
    print()


# ---------------------------------------------------------------------------
# 4. Copy-on-write event snapshot: tuple vs list.copy
# ---------------------------------------------------------------------------


def bench_event_snapshot():
    print("=" * 60)
    print("4. Event list snapshot: tuple() vs list.copy()")
    print("=" * 60)

    # Simulate a realistic event list size
    events = [{"type": f"event_{i}", "data": {"key": f"value_{i}"}} for i in range(200)]

    def tuple_snapshot():
        base = tuple(events)
        for _ in range(10):
            _ = list(base) + [{"type": "start_flow"}]

    t_tuple = bench(
        tuple_snapshot, iterations=5_000, warmup=100, label="tuple snapshot + 10 list(base) expansions (200 events)"
    )

    def list_copy_snapshot():
        for _ in range(10):
            _ = events.copy()
            _.append({"type": "start_flow"})

    t_copy = bench(list_copy_snapshot, iterations=5_000, warmup=100, label="10x list.copy() + append (200 events)")

    print(f"  Summary: tuple approach is {t_copy / t_tuple:.1f}x vs list.copy()")
    print()


# ---------------------------------------------------------------------------
# 5. _render_string end-to-end (cached vs cold)
# ---------------------------------------------------------------------------


def bench_render_string():
    from nemoguardrails.llm.taskmanager import LLMTaskManager
    from nemoguardrails.rails.llm.config import RailsConfig

    print("=" * 60)
    print("5. _render_string end-to-end: warm cache vs cold cache")
    print("=" * 60)

    config = RailsConfig.from_content(yaml_content="models: []")
    tm = LLMTaskManager(config)

    template = "Hello {{ name }}, you have {{ count }} messages from {{ sender }}."
    context = {"name": "Alice", "count": 42, "sender": "Bob"}

    # Warm up the cache
    tm._render_string(template, context=context)

    # --- Cached ---
    def cached_render():
        tm._render_string(template, context=context)

    t_cached = bench(cached_render, iterations=10_000, warmup=100, label="cached _render_string")

    # --- Uncached ---
    def uncached_render():
        tm._template_cache.clear()
        tm._variables_cache.clear()
        tm._render_string(template, context=context)

    t_uncached = bench(
        uncached_render, iterations=10_000, warmup=100, label="uncached _render_string (caches cleared each call)"
    )

    print(f"  Summary: cache speedup = {t_uncached / t_cached:.1f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("NeMo Guardrails Performance Optimisation Benchmarks")
    print("=" * 60)
    print()

    bench_lru_dict()
    bench_action_name_cache()
    bench_template_cache()
    bench_event_snapshot()
    bench_render_string()

    print("=" * 60)
    print("All benchmarks complete.")
    print("=" * 60)
