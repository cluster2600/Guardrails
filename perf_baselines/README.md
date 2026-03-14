# Performance Baselines

This directory holds known-good benchmark results used by CI to detect
regressions.  Files are named by platform and Python version:

    linux-py312.json     — CPython 3.12 on Linux (PR gate)
    linux-py314.json     — CPython 3.14 on Linux (PR gate)
    linux-py314t.json    — CPython 3.14 free-threaded on Linux (nightly)

## Updating baselines

After a deliberate, reviewed performance change:

```bash
python -m benchmarks.run_latency --output perf_baselines/linux-py314.json
git add perf_baselines/
git commit -s -m "perf: update baselines after <change>"
```

Never update baselines to hide a regression — investigate first.
