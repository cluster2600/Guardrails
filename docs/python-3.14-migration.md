# Python 3.14 Migration Guide

This guide covers everything you need to know about running NeMo Guardrails on
Python 3.14, including the langchain compatibility shim, asyncio changes,
dependency considerations, and free-threaded (no-GIL) support.

---

## Overview

Python 3.14 introduces several changes that affect NeMo Guardrails:

| Change | Impact | Status |
|--------|--------|--------|
| **PEP 649** — deferred evaluation of annotations | Breaks `langchain 0.3.x` (`Chain.dict()` shadows `builtins.dict`) | Shimmed via `_langchain_compat.py` |
| **asyncio lifecycle tightening** | `get_event_loop()` may return a closed loop | Fixed in `utils.get_or_create_event_loop()` |
| **Native extension wheels** | Some optional deps lack 3.14 wheels | Documented below |
| **Free-threaded builds (3.14t)** | No GIL — true parallel rail execution | Supported (experimental) |

## Requirements

| Dependency | Minimum version | Notes |
|------------|----------------|-------|
| Python | 3.14.0a1+ | Standard or free-threaded build |
| langchain | 0.3.x *or* 1.x | 0.3.x requires the compat shim; 1.x works natively |
| langchain-core | 0.3.x *or* 1.x | Must match langchain version |
| pydantic | >= 2.0 | Required for PEP 649 annotation handling |

If you are starting a new project, we recommend **langchain >= 1.0.0** (also
called `langchain-classic`) which includes the upstream fix for the `dict()`
shadow issue ([langchain-ai/langchain#33575](https://github.com/langchain-ai/langchain/pull/33575)).

## What Changed

### The langchain compatibility shim (`_langchain_compat.py`)

Python 3.14's PEP 649 introduces deferred evaluation of annotations.  Under
this scheme, pydantic resolves type annotations using `vars(cls)`, which means
a class method named `dict()` shadows the builtin `dict` type.  langchain
0.3.x defines `Chain.dict()` and similar methods, so an annotation like
`dict[str, Any]` resolves to the *method* rather than the builtin, raising:

```
TypeError: 'function' object is not subscriptable
```

The shim in `nemoguardrails/_langchain_compat.py` addresses this by:

1. **Detecting Python >= 3.14** at import time (`sys.version_info >= (3, 14)`).
2. **Injecting `builtins.dict`** into the module-level namespace of affected
   langchain modules *before* pydantic triggers annotation evaluation.
3. **Providing safe import wrappers** (`safe_import_langchain()`,
   `import_init_chat_model()`, `import_chat_models_base()`) that handle both
   langchain 0.3.x and 1.x import paths.

The patched modules are:

- `langchain.chains.base`
- `langchain.chains`
- `langchain.schema`
- `langchain.schema.runnable.base`

This patching is idempotent — each module is marked with a
`__dict_patched__` sentinel to prevent double-patching.

**If you are already on langchain >= 1.0.0**, the shim detects that no
patching is needed and becomes a no-op.

### Event loop fix (`utils.get_or_create_event_loop()`)

Python 3.14 tightened asyncio lifecycle rules.  `asyncio.get_event_loop()` may
now return a **closed** loop in contexts where it previously returned a usable
one.  The updated helper:

```python
def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
```

This ensures NeMo Guardrails always has a valid, open event loop available,
regardless of the Python version.

## Known Limitations

### Native dependencies without 3.14 wheels

Some **optional** NeMo Guardrails dependencies do not yet publish wheels for
Python 3.14.  These are not required for core functionality but affect specific
features:

| Package | Feature affected | Workaround |
|---------|-----------------|------------|
| `annoy` | Nearest-neighbour retrieval for knowledge bases | Build from source (requires C++ compiler) or use `faiss-cpu` / `chromadb` instead |
| `yara-python` | YARA-based content scanning | Build from source with `pip install --no-binary yara-python yara-python` |

> **Important:** In our triage testing, ~77% of initial test failures on
> Python 3.14 were caused by a single missing native dependency (`annoy`)
> cascading through the runtime.  If you see a large number of failures when
> first running on 3.14, check whether annoy is installed before
> investigating further.

To build `annoy` from source on Python 3.14, ensure a C++ compiler is
available in your environment:

```bash
# Debian / Ubuntu (or Docker images based on them)
apt-get install -y build-essential

# Then install annoy from source
pip install --no-binary annoy annoy
```

In Docker, both `Dockerfile.py314` and `Dockerfile.test-py314` already include
`build-essential` for this purpose.

Check the respective upstream issue trackers for wheel availability updates.

### Dependencies not auto-resolved on 3.14

Some packages that are normally pulled in as transitive dependencies may not
be automatically resolved by pip on Python 3.14.  If you encounter import
errors, install these explicitly:

```bash
pip install opentelemetry-api opentelemetry-sdk tqdm
```

### OpenTelemetry

The OpenTelemetry SDK (`opentelemetry-api`, `opentelemetry-sdk`) has known
issues on Python 3.14 related to PEP 649 annotation changes in some of its
internal modules.  Additionally, these packages may not be auto-resolved as
transitive dependencies on 3.14.  If you rely on OTel tracing:

1. Install explicitly: `pip install opentelemetry-api opentelemetry-sdk`
2. Pin to a version that supports 3.14, or disable tracing until a compatible
   release is available.

## Running on Python 3.14 with Docker

Docker is the easiest way to test NeMo Guardrails on Python 3.14 without
modifying your local Python installation.  We provide two Dockerfiles:

### Quick test run

`Dockerfile.test-py314` is a single-stage image that installs all dependencies
and runs the test suite:

```bash
docker build -t guardrails-test-py314 -f Dockerfile.test-py314 .
docker run guardrails-test-py314
```

Tests that depend on native-only packages (annoy, yara) are automatically
skipped via `-k` filters and `pytest.importorskip()` guards.  Both Dockerfiles
include `build-essential` so that native extensions can be compiled from source
if needed.

### Multi-stage image (test + benchmark)

`Dockerfile.py314` is a multi-stage build with separate `test` and `bench`
targets:

```bash
# Build and run tests
docker build --target test  -t guardrails-test-py314  -f Dockerfile.py314 .
docker run guardrails-test-py314

# Build and run benchmarks
docker build --target bench -t guardrails-bench-py314 -f Dockerfile.py314 .
docker run guardrails-bench-py314
```

The bench stage includes the full benchmark harness and performance baselines,
so you can run latency benchmarks inside a reproducible environment.

Both images use `python:3.14` as the base and install dependencies via pip
(poetry does not yet fully support Python 3.14).  The `PYTHONPATH` is set to
`/app` so that `nemoguardrails` is importable without a wheel install.

### CI workflow

The reusable workflow `.github/workflows/_test-py314.yml` runs the test suite
on Python 3.14 in GitHub Actions using `setup-python` with
`allow-prereleases: true`.  It skips annoy/yara-dependent tests and the eval
suite.  This workflow is marked `continue-on-error: true` whilst Python 3.14
support is still experimental.

## Free-Threaded Python (3.14t)

### What is it?

Python 3.14 offers an experimental **free-threaded** build (sometimes written
as `3.14t` or referred to as "no-GIL Python").  This build disables the Global
Interpreter Lock, allowing Python threads to execute truly in parallel on
multiple CPU cores.

### Why it matters for NeMo Guardrails

Guardrail evaluation often involves multiple independent checks (input rails,
output rails, content-safety scans) that are traditionally executed serially or
via `asyncio.gather()`.  With the GIL, CPU-bound rails competing for the same
interpreter lock see no benefit from threading.  With 3.14t, CPU-bound rails
dispatched to a thread pool can run in genuine parallel, delivering near-linear
speedups proportional to the number of available cores.

### How to use it

1. **Install the free-threaded build:**

   ```bash
   # Using pyenv
   pyenv install 3.14.0a1-freethreading
   pyenv shell 3.14.0a1-freethreading

   # Or via the official installer with the --free-threaded flag
   ```

2. **Verify the build:**

   ```python
   import sys, sysconfig
   print(f"Python {sys.version}")
   print(f"GIL disabled: {bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}")
   ```

3. **Run the threading benchmark** to confirm parallel speedup:

   ```bash
   python benchmarks/bench_threading.py
   ```

   On a 4-core machine with `BENCH_RAIL_COUNT=4`, you should see:
   - **Standard Python:** ~1.0x speedup (GIL prevents parallel CPU work)
   - **Free-threaded:** ~3.0-3.8x speedup (true parallelism)

### Expected speedups

The `bench_threading.py` benchmark simulates CPU-bound guardrail work (pattern
matching, token counting, hash-based deduplication) across multiple rails:

| Configuration | Standard (GIL) | Free-threaded (no-GIL) |
|--------------|----------------|----------------------|
| 1 rail | Baseline | Baseline |
| 4 rails, sequential | ~4.0x wall time | ~4.0x wall time |
| 4 rails, threaded | ~4.0x wall time (GIL) | ~1.0-1.3x wall time |

The speedup scales with the number of CPU cores and the ratio of CPU-bound to
I/O-bound work in your rail configuration.

### Caveats

- Free-threaded Python is **experimental** in 3.14.  Expect rough edges.
- Not all C extensions are thread-safe without the GIL.  Check that your
  dependencies are compatible before deploying to production.
- The NeMo Guardrails core is safe for free-threaded use, but third-party
  action providers may need auditing.

## Performance

### Python 3.14 vs 3.12 comparison

The CI performance gate runs the same benchmark suite on both Python 3.12 and
3.14.  Based on the baseline files in `perf_baselines/`, the key observations
are:

| Metric | Python 3.12 | Python 3.14 | Notes |
|--------|------------|------------|-------|
| Latency (p95) | Baseline | Within 5% | Regression gate enforces max 5% p95 increase |
| Mean latency | Baseline | Within 10% | Regression gate enforces max 10% mean increase |
| First-token latency | Baseline | Within 20ms | Regression gate enforces max 20ms increase |
| Import / cold-start | Baseline | Slightly higher | PEP 649 deferred annotations add minor startup cost |
| Free-threaded (3.14t) | N/A | ~3-4x speedup on CPU-bound rails | Requires free-threaded build |

Python 3.14 is expected to perform comparably to 3.12 for standard
(GIL-enabled) builds.  The compatibility shim adds negligible overhead — it
runs once at import time and is a no-op on subsequent calls.

The primary performance benefit of 3.14 comes from the **free-threaded build**,
which enables true parallel execution of CPU-bound guardrail checks.  See the
[Free-Threaded Python](#free-threaded-python-314t) section for details.

To run a head-to-head comparison locally:

```bash
# Run on Python 3.12
python3.12 -m benchmarks.run_latency --output /tmp/perf-312.json

# Run on Python 3.14
python3.14 -m benchmarks.run_latency --output /tmp/perf-314.json

# Compare
python -m benchmarks.compare \
  --baseline /tmp/perf-312.json \
  --current /tmp/perf-314.json \
  --max-p95-regression 5 \
  --max-mean-regression 10
```

### Benchmark harness

This release includes a comprehensive benchmark harness under `benchmarks/`:

| Runner | What it measures |
|--------|-----------------|
| `run_latency.py` | Per-request latency (p50/p95/p99) across serial/parallel/tracing configurations |
| `run_throughput.py` | Requests per second under concurrent load |
| `run_streaming.py` | First-token latency and tokens-per-second for streaming responses |
| `run_import.py` | Cold-start / import time (critical for serverless deployments) |
| `bench_threading.py` | GIL vs free-threaded parallel speedup |

### Running benchmarks locally

```bash
# Latency smoke suite (recommended starting point)
python -m benchmarks.run_latency --output /tmp/perf-latency.json

# Import / cold-start
python -m benchmarks.run_import --output /tmp/perf-import.json

# Compare against baselines
python -m benchmarks.compare \
  --baseline perf_baselines/linux-py314.json \
  --current /tmp/perf-latency.json \
  --max-p95-regression 5 \
  --max-first-token-regression-ms 20 \
  --max-mean-regression 10
```

### CI integration

The GitHub Actions workflow `.github/workflows/perf.yml` runs:

- **On every PR:** latency smoke + import benchmarks (Python 3.12 and 3.14)
- **Nightly:** full suite including throughput, streaming, and free-threaded
  benchmarks (3.12, 3.14, 3.14t)

Results are uploaded as CI artefacts and compared against baselines in
`perf_baselines/`.  See `perf_baselines/README.md` for instructions on
updating baselines after deliberate performance changes.

## Troubleshooting

### `TypeError: 'function' object is not subscriptable`

**Cause:** You are running langchain 0.3.x on Python 3.14 and the compatibility
shim did not activate early enough.

**Fix:** Ensure NeMo Guardrails is imported *before* directly importing
langchain chain classes.  The shim patches langchain modules at import time:

```python
# Correct order
import nemoguardrails  # activates the compat shim
from langchain.chains import LLMChain

# Incorrect — may fail before the shim runs
from langchain.chains import LLMChain
import nemoguardrails
```

Alternatively, upgrade to **langchain >= 1.0.0** which does not have this issue.

### `RuntimeError: There is no current event loop in thread`

**Cause:** Python 3.14 no longer implicitly creates an event loop in
non-main threads.

**Fix:** Use `nemoguardrails.utils.get_or_create_event_loop()` instead of
`asyncio.get_event_loop()` directly.  This is handled automatically within
NeMo Guardrails, but if you call asyncio APIs from custom action code, use
the helper.

### `asyncio.get_event_loop()` returns a closed loop

**Cause:** Python 3.14 tightened asyncio lifecycle management.  A loop that
was previously usable may now be returned in a closed state.

**Fix:** Same as above — `get_or_create_event_loop()` detects closed loops
and creates a fresh one.

### `ImportError: No module named 'annoy'` (or `yara`)

**Cause:** These optional native dependencies do not yet have pre-built wheels
for Python 3.14.

**Fix:** Either build from source (`pip install --no-binary :all: annoy`) or
switch to an alternative backend (e.g. `faiss-cpu` for nearest-neighbour
retrieval).  See the [Known Limitations](#known-limitations) section above.

### Free-threaded build shows no speedup

**Cause:** Your Python build may not actually be free-threaded.

**Fix:** Verify with:

```python
import sysconfig
print(sysconfig.get_config_var("Py_GIL_DISABLED"))
# Should print 1 for free-threaded builds
```

If it prints `0` or `None`, you are running a standard build.  Install the
free-threaded variant explicitly.

### `ModuleNotFoundError: No module named 'langchain_classic'`

**Cause:** The fallback import path for langchain 1.x uses
`langchain_classic`, which is the rebranded package name.

**Fix:** Install the correct package:

```bash
pip install langchain-classic  # langchain 1.x
# OR
pip install langchain           # langchain 0.3.x (requires compat shim on 3.14)
```
