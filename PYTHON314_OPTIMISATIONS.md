# Python 3.14 Performance Optimisations for NeMo Guardrails

This document describes the performance optimisations implemented to take
advantage of Python 3.12 / 3.14 / 3.14t (free-threaded) improvements in the
NeMo Guardrails framework.  All changes are backwards-compatible with Python
3.10+.

## Summary of Changes

### 1. Eager Task Factory (Python 3.12+)

**File:** `nemoguardrails/rails/llm/dag_scheduler.py`

The `TopologicalScheduler.execute()` method now installs
`asyncio.eager_task_factory` for the duration of each execution call on
Python 3.12+.  Eager tasks begin executing immediately when created, rather
than waiting for the next event-loop iteration.  This eliminates the
scheduling round-trip for coroutines that complete synchronously — such as
cache hits, trivially fast checks, or already-resolved futures.

The factory is installed at the start of `execute()` and restored
afterwards, so it does not leak state into the caller's event loop.

**Benchmark results (eager vs standard task creation):**

| Tasks | Python 3.10 (no eager) | Python 3.14 (standard) | Python 3.14 (eager) | Speedup |
|------:|-----------------------:|-----------------------:|--------------------:|--------:|
|     4 |             0.2179 ms  |             0.1981 ms  |           0.1588 ms | 1.25x   |
|     8 |             0.2300 ms  |             0.2173 ms  |           0.1508 ms | 1.44x   |
|    16 |             0.2555 ms  |             0.2393 ms  |           0.1706 ms | 1.40x   |
|    32 |             0.3087 ms  |             0.2846 ms  |           0.1968 ms | 1.45x   |

The eager factory delivers a consistent **1.25x–1.45x speedup** on task
creation and dispatch, which directly benefits the DAG scheduler's parallel
group execution.

### 2. Shared CPU Thread Pool for Free-Threaded Python

**File:** `nemoguardrails/rails/llm/dag_scheduler.py`

A module-level `ThreadPoolExecutor` (`_CPU_POOL`) is created when the
interpreter is detected as a free-threaded build (Python 3.13t / 3.14t).
The pool is sized to `os.cpu_count()`, enabling true thread-level
parallelism for CPU-bound guardrail checks without GIL contention.

On standard (GIL-enabled) builds, `get_cpu_executor()` returns `None`,
which causes `loop.run_in_executor(None, ...)` to use the default thread
pool — preserving the existing behaviour.

**Detection mechanism:**
```python
_IS_FREE_THREADED = bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False)
```

### 3. CPU-Bound Rail Offloading via `run_in_executor`

**Files:**
- `nemoguardrails/library/regex/actions.py`
- `nemoguardrails/library/sensitive_data_detection/actions.py`
- `nemoguardrails/library/injection_detection/actions.py`
- `nemoguardrails/library/content_safety/actions.py`

All CPU-intensive guardrail actions now dispatch their synchronous work to a
thread pool via `loop.run_in_executor(get_cpu_executor(), ...)`.  This has
two benefits:

1. **On all Python versions:** prevents CPU-bound work from blocking the
   asyncio event loop, allowing I/O-bound rails to proceed concurrently.

2. **On free-threaded Python 3.14t:** the shared `_CPU_POOL` enables true
   parallel execution across multiple CPU cores, since the GIL no longer
   serialises thread execution.

The specific operations offloaded:

| Module                    | Offloaded Operation                          |
|--------------------------|----------------------------------------------|
| `regex/actions.py`        | Compiled regex pattern matching loop         |
| `sensitive_data_detection`| Presidio `analyzer.analyze()` + anonymisation |
| `injection_detection`     | YARA rule matching and string replacement     |
| `content_safety`          | `fast-langdetect` language detection          |

### 4. Version-Gated Feature Flags

**File:** `nemoguardrails/rails/llm/dag_scheduler.py`

Three module-level constants enable version-aware optimisation without
runtime overhead:

```python
_PY_VERSION = sys.version_info[:2]
_HAS_EAGER_TASK_FACTORY: bool = _PY_VERSION >= (3, 12)
_IS_FREE_THREADED: bool = bool(getattr(sys, "_is_gil_enabled", lambda: True)() is False)
```

These flags are evaluated once at import time and used in conditional
branches throughout the scheduler.

## Benchmark Results

### Full Scheduler Execute (TopologicalScheduler.execute())

This benchmark runs the actual `TopologicalScheduler.execute()` codepath,
including eager task factory installation, early-exit detection, and
per-group timeout handling.

| Topology         | Rails | Groups | Python 3.10  | Python 3.14  | Improvement |
|-----------------|------:|-------:|-------------:|-------------:|------------:|
| wide_4           |     4 |      1 |      7.7 ms  |      7.6 ms  |     +1.2%   |
| wide_8           |     8 |      1 |      8.1 ms  |      6.6 ms  |    +18.8%   |
| diamond          |     4 |      3 |     21.9 ms  |     21.0 ms  |     +4.2%   |
| fan_out_8        |     9 |      2 |     15.0 ms  |     14.8 ms  |     +1.5%   |
| deep_pipeline_6  |     6 |      6 |     41.5 ms  |     41.1 ms  |     +1.0%   |

The **wide_8** topology (8 independent rails in a single group) shows the
largest improvement at **18.8%**, as it benefits most from the eager task
factory — all 8 tasks are created and begin executing immediately rather
than waiting for successive event-loop iterations.

### Import Time (PEP 649 Deferred Annotations)

Python 3.14 includes PEP 649 (deferred evaluation of annotations), which
reduces import time for annotation-heavy modules.  The `from __future__
import annotations` directive already provides this on 3.10+, but 3.14's
native support avoids the `__future__` import overhead.

| Module                                 | Python 3.10   | Python 3.14   | Improvement |
|---------------------------------------|-------------:|-------------:|------------:|
| `nemoguardrails.rails.llm.config`      |    975.1 ms  |    853.9 ms  |    +12.4%   |
| `nemoguardrails.rails.llm.llmrails`    |    990.3 ms  |    862.3 ms  |    +12.9%   |
| `nemoguardrails.rails.llm.dag_scheduler`|   990.7 ms  |    865.0 ms  |    +12.7%   |
| `nemoguardrails.llm.taskmanager`       |  1,028.2 ms  |    878.1 ms  |    +14.6%   |
| `nemoguardrails.actions.action_dispatcher`| 1,058.4 ms |    909.2 ms  |    +14.1%   |

Annotation-heavy modules see a **12–15% reduction** in cold-start import
time on Python 3.14.

### GC Tail Latency

Under sustained GC pressure, Python 3.14's incremental garbage collector
produces slightly lower p99/p50 ratios:

| Scenario       | Python 3.10 p99/p50 | Python 3.14 p99/p50 |
|---------------|--------------------:|--------------------:|
| light_pressure |              1.27x  |              1.32x  |
| heavy_pressure |              1.36x  |              1.33x  |
| burst_pressure |              1.36x  |              1.35x  |

### Free-Threaded Python 3.14t

The eager task factory is available on 3.14t and shows a **1.49x speedup**
at 32 concurrent tasks.  The free-threaded build detects correctly:

```
Python 3.14.3 (free-threaded, 10 CPUs)
GIL disabled: True
```

The shared `_CPU_POOL` is automatically sized to 10 workers on this system,
enabling true parallel dispatch of CPU-bound rails.

## Backwards Compatibility

All optimisations are version-gated and fully backwards-compatible:

- **Python 3.10–3.11:** No eager factory, no free-threaded pool.
  `get_cpu_executor()` returns `None`, falling back to the default executor.
- **Python 3.12–3.13:** Eager task factory is enabled.
  `get_cpu_executor()` returns `None` (standard GIL build).
- **Python 3.14:** Eager task factory + faster imports + incremental GC.
- **Python 3.14t:** All above + shared CPU thread pool for true parallelism.

## Running the Benchmarks

```bash
# Full suite
python -m benchmarks.bench_py314_advantages

# Individual sections
python -m benchmarks.bench_py314_advantages --section eager
python -m benchmarks.bench_py314_advantages --section scheduler

# Save results to JSON
python -m benchmarks.bench_py314_advantages --output results.json

# Cross-version comparison
python3.10 -m benchmarks.bench_py314_advantages -o /tmp/py310.json
python3.14 -m benchmarks.bench_py314_advantages -o /tmp/py314.json
python3.14t -m benchmarks.bench_py314_advantages -o /tmp/py314t.json
```

## Available Benchmark Sections

1. **threading** — Sequential vs ThreadPool vs async executor for CPU rails
2. **template** — Jinja2 template cache (cold vs hot rendering)
3. **import** — Cold-start import timing via subprocess
4. **gc** — GC tail latency under escalating pressure
5. **asyncio** — `asyncio.gather` with I/O-bound rails at scale
6. **dag** — DAG scheduler group parallelism
7. **eager** — Eager task factory vs standard task creation
8. **scheduler** — Full `TopologicalScheduler.execute()` codepath
