# Parallel Rail Execution with Dependency Scheduling

This guide explains how to configure NeMo Guardrails to run rails concurrently
while respecting dependency ordering between them.  It covers the basic
parallel mode, the new DAG-based dependency scheduler, the `@cpu_bound`
decorator for CPU-intensive rails, and thread-pool configuration for
free-threaded Python.

---

## Background

By default rails execute sequentially in the order declared in `config.yml`.
Sequential execution is simple and deterministic, but total latency is the sum
of every rail's latency.

Setting `parallel: true` runs all rails in a section concurrently, reducing
latency to approximately the slowest rail.  However this requires that every
rail is fully independent -- no rail may depend on the output of another.

The DAG-based scheduler fills the gap: you declare explicit dependencies
between rails and the scheduler automatically determines which rails can run
concurrently and which must wait.

---

## Quick Start: All-Parallel Mode (No Dependencies)

If your rails are fully independent, enable parallel mode with a single flag:

```yaml
rails:
  input:
    parallel: true
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control
  output:
    parallel: true
    flows:
      - content safety check output $model=content_safety
      - self check output
```

Both input rails run concurrently.  Both output rails run concurrently.  If
any rail signals a block, the remaining rails in that section are cancelled.

---

## Declaring Dependencies with `depends_on`

When one rail needs the result of another, declare the relationship with the
`depends_on` field.  The scheduler builds a directed acyclic graph (DAG) and
executes rails in topological order: rails with no unmet dependencies run
concurrently; a rail waits until all of its dependencies have completed.

### Basic example

```yaml
rails:
  input:
    parallel: true
    flows:
      - name: pii scrub input
        action: pii_scrub
        depends_on: []

      - name: content safety check input
        action: content_safety_check
        depends_on: []

      - name: jailbreak detection
        action: jailbreak_detect
        depends_on:
          - pii scrub input
```

Execution proceeds in two groups:

| Group | Rails                                         | Runs |
|-------|-----------------------------------------------|------|
| 0     | `pii scrub input`, `content safety check input` | concurrently |
| 1     | `jailbreak detection`                          | after group 0 |

`pii scrub input` and `content safety check input` have no dependencies so
they start immediately in parallel.  `jailbreak detection` depends on
`pii scrub input`, so it waits for group 0 to finish before starting.

### Diamond dependency

```yaml
rails:
  input:
    parallel: true
    flows:
      - name: embedding lookup
        action: embed_lookup
        depends_on: []

      - name: pii scrub input
        action: pii_scrub
        depends_on: []

      - name: classify intent
        action: classify
        depends_on:
          - embedding lookup

      - name: jailbreak detection
        action: jailbreak_detect
        depends_on:
          - pii scrub input

      - name: policy enforcement
        action: enforce_policy
        depends_on:
          - classify intent
          - jailbreak detection
```

Execution groups:

| Group | Rails                                   |
|-------|-----------------------------------------|
| 0     | `embedding lookup`, `pii scrub input`   |
| 1     | `classify intent`, `jailbreak detection` |
| 2     | `policy enforcement`                     |

### Cycle detection

The scheduler validates the dependency graph at configuration load time.  If a
cycle exists, startup fails with an error:

```
InvalidRailsConfigurationError: Cycle detected in input rail dependencies:
  jailbreak detection -> pii scrub input -> jailbreak detection
```

Fix cycles by restructuring the dependencies so that the graph is acyclic.

---

## Fallback Behavior

| Configuration                          | Behavior                              |
|----------------------------------------|---------------------------------------|
| `parallel: false` (default)            | Sequential execution, declaration order |
| `parallel: true`, no `depends_on`      | All rails run concurrently (same as today) |
| `parallel: true`, with `depends_on`    | DAG-scheduled grouped execution       |
| `parallel: false`, with `depends_on`   | `depends_on` is ignored; rails run sequentially |

Existing configurations that use `parallel: true` without any `depends_on`
fields behave identically to the current release.  No migration is required.

---

## CPU-Bound Rails and the `@cpu_bound` Decorator

Rails that call external APIs (LLM endpoints, moderation services) are
I/O-bound and benefit directly from `asyncio` concurrency.  Rails that perform
CPU-intensive work (regex scanning, hashing, PII pattern matching) block the
event loop and prevent other rails in the same group from making progress.

The `@cpu_bound` decorator dispatches a synchronous function to a
`ThreadPoolExecutor` via `loop.run_in_executor()`, unblocking the event loop:

```python
import asyncio
import hashlib
import re

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),   # SSN
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # Email
]


def cpu_bound(fn):
    """Decorator that dispatches a sync function to a thread-pool executor."""
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper._cpu_bound = True
    return wrapper


@cpu_bound
def scan_for_pii(text: str) -> dict:
    """Scan text for PII patterns -- CPU-intensive."""
    found = []
    for pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(pattern.pattern)

    # Simulate heavier compute (token hashing, classification, etc.)
    digest = text.encode("utf-8")
    for _ in range(500):
        digest = hashlib.sha256(digest).digest()

    return {"pii_found": bool(found), "patterns": found}
```

Register the action so that the runtime can call it from a Colang flow:

```python
from nemoguardrails import LLMRails

rails = LLMRails(config)
rails.register_action(scan_for_pii, name="scan_for_pii")
```

When `scan_for_pii` is invoked as part of a parallel group, it runs in a
thread and does not block I/O-bound rails in the same group.

---

## Thread Pool Configuration for Free-Threaded Python

Standard CPython uses a Global Interpreter Lock (GIL) that prevents true
parallel execution of Python bytecode across threads.  Dispatching CPU-bound
work to threads still helps because it unblocks the `asyncio` event loop, but
the CPU work itself runs serially.

Python 3.14t (free-threaded / no-GIL builds) removes this limitation.
CPU-bound threads run on separate cores and achieve near-linear speedup.

### Verifying your Python build

```bash
python -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
# 1 = free-threaded, 0 or None = standard GIL build
```

### Setting the thread-pool size

The default `ThreadPoolExecutor` that `asyncio` uses is sized to
`min(32, os.cpu_count() + 4)`.  For guardrail workloads you may want explicit
control.  Set a custom executor on the event loop before creating `LLMRails`:

```python
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Size the pool to the number of CPU-bound rails you expect to run in parallel.
# A reasonable default is the number of CPU cores.
pool = ThreadPoolExecutor(max_workers=os.cpu_count())
loop = asyncio.get_event_loop()
loop.set_default_executor(pool)
```

### Running the threading benchmark

The repository includes a benchmark that measures sequential vs. threaded
execution of CPU-bound rail work:

```bash
python benchmarks/bench_threading.py
```

On a free-threaded build with 4 rails you should see near-4x speedup.  On a
standard GIL build the speedup is approximately 1.0x for CPU-bound work.

---

## Performance Tips

### Which rails benefit from parallelism?

| Rail type                         | Bound   | Parallel benefit                |
|-----------------------------------|---------|---------------------------------|
| LLM-based safety check            | I/O     | High -- network wait overlaps   |
| External moderation API call       | I/O     | High                            |
| Regex / PII scanning               | CPU     | Medium (needs `@cpu_bound`)     |
| Hash-based deduplication           | CPU     | Medium (needs `@cpu_bound`)     |
| Embedding similarity lookup        | I/O+CPU | High for the I/O portion        |
| Context variable read (cheap)      | Neither | Low -- overhead may exceed cost |

### General guidelines

1. **Start sequential, then optimize.**  Get your rails working correctly in
   sequential mode first.  Add `parallel: true` only after confirming that
   the rails are independent or properly declaring dependencies.

2. **Identify I/O-bound rails first.**  Rails that call LLM endpoints or
   external services provide the biggest latency wins from parallelism because
   the event loop can overlap network waits.

3. **Use `@cpu_bound` for heavy compute.**  If a rail does more than trivial
   string operations, wrap its sync function with `@cpu_bound` so it runs in a
   thread and does not starve the event loop.

4. **Avoid shared mutable state.**  Rails in the same execution group must not
   write to the same context variable.  If two rails both need to write to
   `$user_message`, they must be in different groups (declare a dependency).

5. **Keep dependency chains short.**  Every sequential dependency adds a
   group boundary.  A chain of four dependent rails still executes four groups
   sequentially.  Restructure rails to reduce chain depth when possible.

6. **Monitor with tracing.**  Enable tracing in `config.yml` to see which
   rails ran in which group and how long each took:

   ```yaml
   tracing:
     enabled: true
   ```

---

## Migration from Sequential to Parallel Execution

### Step 1: Audit rail independence

For each pair of rails in a section, ask:

* Does rail B read a context variable that rail A writes?
* Does rail B read the (potentially mutated) user message that rail A
  modified?

If the answer is no for all pairs, the rails are independent and can use
simple `parallel: true`.

### Step 2: Enable parallel mode

```yaml
# Before
rails:
  input:
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control

# After
rails:
  input:
    parallel: true
    flows:
      - content safety check input $model=content_safety
      - topic safety check input $model=topic_control
```

### Step 3: Add dependencies where needed

If some rails have ordering requirements, switch to the named-flow form with
`depends_on`:

```yaml
rails:
  input:
    parallel: true
    flows:
      - name: pii scrub input
        action: pii_scrub
        depends_on: []

      - name: content safety check input
        action: content_safety_check
        depends_on:
          - pii scrub input

      - name: topic safety check input
        action: topic_safety_check
        depends_on: []
```

Here `content safety check input` waits for `pii scrub input` to finish
(because it needs the scrubbed text), while `topic safety check input` runs in
parallel with `pii scrub input` since they are independent.

### Step 4: Wrap CPU-bound actions

If any rail action performs significant CPU work, apply the `@cpu_bound`
decorator as shown above.

### Step 5: Test and benchmark

1. Run your existing test suite to confirm output parity with sequential mode.
2. Use the built-in benchmarks to measure latency improvement:

   ```bash
   python benchmarks/run_latency.py
   python benchmarks/run_throughput.py
   ```

3. Check logs for unexpected race conditions or context variable conflicts.

---

## Troubleshooting

### "Cycle detected in rail dependencies"

The dependency graph contains a cycle.  Review the `depends_on` fields and
remove or restructure the circular reference.

### Rail results differ between sequential and parallel mode

A rail is likely reading a context variable that another concurrent rail
writes.  Add a `depends_on` edge from the reader to the writer so the reader
waits for the writer to finish.

### CPU-bound rails block other rails

The rail function is running synchronously on the event loop.  Apply the
`@cpu_bound` decorator to dispatch it to the thread pool.

### Low speedup on free-threaded Python

Check that the thread pool has enough workers (`ThreadPoolExecutor(max_workers=...)`)
and that you are using a free-threaded build (`Py_GIL_DISABLED=1`).
