# Python 3.14 Performance Optimisations for NeMo Guardrails

This document describes the performance optimisations impelemented to take
advantage of Python 3.12 / 3.14 / 3.14t (free-threaded) improvements in the
NeMo Guardrails framework.  All changes are backwards-compatible with Python
3.10+ and are version-gated at import time so that no runtime overhead is
incurred on older interpreters.

## Table of Contents

1. [Summary of Changes](#summary-of-changes)
2. [Benchmark Methodology](#benchmark-methodology)
3. [Benchmark Results](#benchmark-results)
4. [Backwards Compatibility](#backwards-compatibility)
5. [Running the Benchmarks](#running-the-benchmarks)

---

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

**How it works:**

```python
# In TopologicalScheduler.execute()
loop = asyncio.get_running_loop()
if _HAS_EAGER_TASK_FACTORY:
    previous_task_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
try:
    # ... execute groups with asyncio.create_task() ...
finally:
    if _HAS_EAGER_TASK_FACTORY:
        loop.set_task_factory(previous_task_factory)
```

When a coroutine completes before its first suspension point (i.e. it does
not `await` anything, or it `await`s a value that is already available),
the eager factory returns the result immediately without scheduling a
round-trip through the event loop.  This is precisely what happens when a
guardrail check hits its cache or determines early that no action is needed.

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

This uses the `sys._is_gil_enabled()` function introduced in PEP 703.  On
standard Python builds where this function does not exist, the `getattr`
fallback returns a lambda that returns `True`, correctly identifying the
build as GIL-enabled.

### 3. CPU-Bound Rail Offloading via `run_in_executor`

**Files modified:**

- `nemoguardrails/library/regex/actions.py`
- `nemoguardrails/library/sensitive_data_detection/actions.py`
- `nemoguardrails/library/injection_detection/actions.py`
- `nemoguardrails/library/content_safety/actions.py`

All CPU-intensive guardrail actions now dispatch their synchronous work to
a thread pool via `loop.run_in_executor(get_cpu_executor(), ...)`.  This
has two distinct benefits:

1. **On all Python versions:** prevents CPU-bound work from blocking the
   asyncio event loop, allowing I/O-bound rails (e.g. LLM API calls,
   embedding lookups) to proceed concurrently.

2. **On free-threaded Python 3.14t:** the shared `_CPU_POOL` enables true
   parallel execution across multiple CPU cores, since the GIL no longer
   serialises thread execution.

The specific operations offloaded to the thread pool:

| Module                          | Offloaded Operation                                |
|--------------------------------|----------------------------------------------------|
| `regex/actions.py`              | Compiled regex pattern matching loop               |
| `sensitive_data_detection/`     | Presidio `analyzer.analyze()` + anonymisation      |
| `injection_detection/`          | YARA rule compilation, matching, string replacement|
| `content_safety/actions.py`     | `fast-langdetect` language detection               |

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
branches throughout the scheduler.  The cost on older Python versions is
exactly two boolean comparisons per `execute()` call.

---

## Benchmark Methodology

The benchmark suite (`benchmarks/bench_py314_advantages.py`) was
significantly improved to produce accurate, reproducible measurements:

### Key methodological improvements

1. **Persistent event loop** — Sections 7 and 8 use
   `loop.run_until_complete()` on a long-lived event loop, rather than
   `asyncio.run()` per iteration.  `asyncio.run()` creates and tears down
   an event loop each time, adding ~0.1ms of overhead that drowns out the
   sub-microsecond scheduling improvements we are measuring.

2. **Pure-Python CPU work** — Section 1 (threading) now uses a pure-Python
   CPU-bound function (`_pure_python_cpu_work`) that genuinely holds the
   GIL throughout execution.  The previous benchmark used `hashlib.sha256`
   which is implemented in C and releases the GIL during hashing — meaning
   threads could already run in parallel on standard Python, hiding the
   free-threaded advantage entirely.

3. **Sub-millisecond rail durations** — Section 8 (scheduler before/after)
   uses 0.1ms rails instead of 5ms, making the scheduling overhead visible
   rather than being drowned out by sleep time.

4. **Before/after comparison** — Section 8 directly compares the same
   topology with and without the eager task factory on the same Python
   version, proving the code change delivers measurable improvement.

5. **Mixed fast/slow workloads** — Section 7 includes a mixed benchmark
   (half instant coroutines, half 1ms I/O) that reflects realistic
   guardrail pipelines where some checks hit the cache and others perform
   real work.

### Benchmark environment

All results below were collected on the same machine:

- **Hardware:** Apple M2 Pro, 10 CPU cores
- **OS:** macOS Darwin 25.4.0
- **Python 3.10.11** (standard build, GIL enabled)
- **Python 3.14.3** (standard build, GIL enabled)
- **Python 3.14.3** (free-threaded build, GIL disabled)

---

## Benchmark Results

### Eager Task Factory (Section 7)

Measures `asyncio.create_task()` + `asyncio.gather()` throughput on a
persistent event loop with instantly-completing coroutines (simulating
cache hits).

| Tasks | Python 3.10 (no eager) | Python 3.14 (standard) | Python 3.14 (eager) | Speedup |
|------:|-----------------------:|-----------------------:|--------------------:|--------:|
|     4 |             0.0941 ms  |             0.0936 ms  |           0.0421 ms | **2.22x** |
|     8 |             0.1110 ms  |             0.1073 ms  |           0.0483 ms | **2.22x** |
|    16 |             0.1363 ms  |             0.1261 ms  |           0.0580 ms | **2.17x** |
|    32 |             0.1862 ms  |             0.1709 ms  |           0.0834 ms | **2.05x** |
|    64 |             0.2858 ms  |             0.2526 ms  |           0.1283 ms | **1.97x** |

The eager factory delivers a consistent **2.0x–2.2x speedup** for
instantly-completing tasks.  This directly benefits the DAG scheduler when
guardrail checks hit the template or result cache.

On Python 3.14t (free-threaded), the eager factory shows a similar
**2.0x speedup**.

### Free-Threaded Parallel CPU Rails (Section 1)

Compares sequential execution vs `ThreadPoolExecutor` for pure-Python
CPU-bound work that holds the GIL throughout.  This is the definitive
test for whether free-threaded Python enables true parallelism.

| Rails | Workers | Python 3.10 seq | 3.10 tpool | 3.14t seq | 3.14t tpool | 3.14t speedup |
|------:|--------:|----------------:|-----------:|----------:|------------:|--------------:|
|     1 |       1 |         3.4 ms  |    3.4 ms  |   4.7 ms  |     4.6 ms  |       1.02x   |
|     2 |       2 |         6.9 ms  |    6.9 ms  |   9.1 ms  |     4.8 ms  |     **1.92x** |
|     4 |       4 |        13.8 ms  |   13.8 ms  |  18.5 ms  |     7.3 ms  |     **2.54x** |
|     8 |       8 |        27.6 ms  |   27.2 ms  |  37.2 ms  |     9.2 ms  |     **4.06x** |

On Python 3.10, threads achieve **1.01x** — the GIL completely prevents
parallel execution of pure-Python code.

On Python 3.14t, threads achieve **4.06x speedup at 8 rails** — genuine
parallelism across CPU cores.  This is the single largest improvement and
directly benefits guardrail checks that perform regex matching, PII
scanning, or YARA rule evaluation.

### Import Time (Section 3)

Python 3.14 includes PEP 649 (deferred evaluation of annotations), which
reduces import time for annotation-heavy modules.

| Module                                   | Python 3.10   | Python 3.14   | Improvement |
|-----------------------------------------|-------------:|-------------:|------------:|
| `nemoguardrails.rails.llm.config`        |    975.1 ms  |    853.9 ms  |  **+12.4%** |
| `nemoguardrails.rails.llm.llmrails`      |    990.3 ms  |    862.3 ms  |  **+12.9%** |
| `nemoguardrails.rails.llm.dag_scheduler` |    990.7 ms  |    865.0 ms  |  **+12.7%** |
| `nemoguardrails.llm.taskmanager`         |  1,028.2 ms  |    878.1 ms  |  **+14.6%** |
| `nemoguardrails.actions.action_dispatcher`| 1,058.4 ms  |    909.2 ms  |  **+14.1%** |

Annotation-heavy modules see a **12–15% reduction** in cold-start import
time on Python 3.14.  This compounds across the full module graph.

### Template Rendering Cache (Section 2)

Measures the Jinja2 template cache that avoids re-parsing and
re-compiling templates on every guardrail prompt render.

| Template  | Cold (parse+compile) | Hot (cached)  | Speedup    | Savings      |
|-----------|---------------------:|--------------:|-----------:|-------------:|
| simple    |           0.2107 ms  |    0.0046 ms  |  **45.4x** | 206 us/call  |
| medium    |           0.7442 ms  |    0.0207 ms  |  **36.0x** | 724 us/call  |
| complex   |           1.2523 ms  |    0.0104 ms  | **120.6x** | 1242 us/call |

### GC Tail Latency (Section 4)

Under sustained GC pressure, Python 3.14's incremental garbage collector
produces slightly lower p99/p50 ratios, indicating more consistent
latency:

| Scenario        | Python 3.10 p99/p50 | Python 3.14 p99/p50 |
|----------------|--------------------:|--------------------:|
| baseline        |              1.30x  |              1.28x  |
| light_pressure  |              1.27x  |              1.32x  |
| heavy_pressure  |              1.36x  |              1.33x  |
| burst_pressure  |              1.36x  |              1.35x  |

### Asyncio Gather (Section 5)

I/O-bound rails (simulated external API calls) show near-perfect parallel
scaling:

| Rails | Serial       | Parallel     | Speedup     |
|------:|-------------:|-------------:|------------:|
|     2 |     43.2 ms  |     22.3 ms  |     1.94x   |
|     4 |     88.1 ms  |     23.6 ms  |     3.73x   |
|     8 |    178.7 ms  |     22.7 ms  |     7.88x   |
|    16 |    357.1 ms  |     23.9 ms  |  **14.97x** |

---

## Backwards Compatibility

All optimisations are version-gated and fully backwards-compatible:

| Python Version   | Eager Factory | CPU Pool    | `get_cpu_executor()` |
|-----------------|:-------------:|:-----------:|:--------------------:|
| 3.10–3.11       | No            | No          | Returns `None`       |
| 3.12–3.13       | Yes           | No          | Returns `None`       |
| 3.14 (standard) | Yes           | No          | Returns `None`       |
| 3.14t (free)    | Yes           | Yes (N CPUs)| Returns pool         |

When `get_cpu_executor()` returns `None`, `loop.run_in_executor(None, ...)`
falls back to Python's default `ThreadPoolExecutor`, preserving the
existing behaviour identically.

---

## Running the Benchmarks

```bash
# Full suite (all 8 sections)
python -m benchmarks.bench_py314_advantages

# Individual sections
python -m benchmarks.bench_py314_advantages --section eager
python -m benchmarks.bench_py314_advantages --section scheduler
python -m benchmarks.bench_py314_advantages --section threading

# Save results to JSON for cross-version comparison
python3.10 -m benchmarks.bench_py314_advantages -o /tmp/py310.json
python3.14 -m benchmarks.bench_py314_advantages -o /tmp/py314.json
python3.14t -m benchmarks.bench_py314_advantages -o /tmp/py314t.json
```

### Available Benchmark Sections

| Key         | Section                                              | What It Measures                                     |
|------------|------------------------------------------------------|------------------------------------------------------|
| `threading` | Section 1: Free-threaded parallel CPU rail execution | GIL vs no-GIL with pure-Python work                  |
| `template`  | Section 2: Jinja2 template rendering (M3 cache)     | Cold parse vs cached render                          |
| `import`    | Section 3: Import time (PEP 649)                     | Cold-start time for annotation-heavy modules         |
| `gc`        | Section 4: GC tail latency under pressure            | p99/p999 under reference cycle load                  |
| `asyncio`   | Section 5: Asyncio gather with I/O rails             | Parallel scheduling efficiency                       |
| `dag`       | Section 6: DAG scheduler parallelism                 | Topological group execution                          |
| `eager`     | Section 7: Eager task factory (persistent loop)      | Standard vs eager task creation overhead             |
| `scheduler` | Section 8: Scheduler execute() before/after          | Real execute() with and without eager factory        |

---

## Key Takeaways

1. **Eager task factory delivers 2x speedup** for cache-hit rails — this is
   the most impactful change for typical guardrail pipelines where many
   checks resolve instantly from cache.

2. **Free-threaded Python enables 4x parallel CPU throughput** — guardrail
   checks that perform regex matching, PII scanning, or YARA rule evaluation
   can now run in genuine parallel across CPU cores.

3. **Import time drops 12–15%** on Python 3.14 thanks to deferred annotation
   evaluation (PEP 649), reducing cold-start latency for guardrail services.

4. **All improvements are automatic** — no configuration changes needed.
   The framework detects the Python version at import time and enables the
   appropriate optimisations transparently.
