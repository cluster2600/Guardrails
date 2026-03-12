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

"""Tests for nemoguardrails._thread_safety module.

Covers:
  - is_free_threaded() detection
  - ThreadSafeDict correctness under concurrent access
  - ThreadSafeCache LRU eviction, stats, and concurrent correctness
  - atomic_init single-execution guarantee under concurrent access
"""

import concurrent.futures
import threading
import time

import pytest

from nemoguardrails._thread_safety import (
    ThreadSafeCache,
    ThreadSafeDict,
    atomic_init,
    is_free_threaded,
)


# ---------------------------------------------------------------------------
# is_free_threaded
# ---------------------------------------------------------------------------


class TestIsFreeThreaded:
    """Tests for the is_free_threaded() runtime detection."""

    def test_returns_bool(self):
        result = is_free_threaded()
        assert isinstance(result, bool)

    def test_consistent_across_calls(self):
        """The function caches its result -- repeated calls must agree."""
        assert is_free_threaded() == is_free_threaded()


# ---------------------------------------------------------------------------
# ThreadSafeDict -- basic dict behaviour
# ---------------------------------------------------------------------------


class TestThreadSafeDictBasic:
    """ThreadSafeDict must behave like a plain dict for single-threaded use."""

    def test_setitem_getitem(self):
        d: ThreadSafeDict = ThreadSafeDict()
        d["a"] = 1
        assert d["a"] == 1

    def test_delitem(self):
        d: ThreadSafeDict = ThreadSafeDict({"x": 10})
        del d["x"]
        assert "x" not in d

    def test_pop(self):
        d: ThreadSafeDict = ThreadSafeDict({"k": 42})
        assert d.pop("k") == 42
        assert d.pop("k", "default") == "default"

    def test_update(self):
        d: ThreadSafeDict = ThreadSafeDict()
        d.update({"a": 1, "b": 2})
        assert dict(d) == {"a": 1, "b": 2}

    def test_setdefault(self):
        d: ThreadSafeDict = ThreadSafeDict()
        d.setdefault("x", 99)
        assert d["x"] == 99
        d.setdefault("x", 0)
        assert d["x"] == 99  # not overwritten

    def test_clear(self):
        d: ThreadSafeDict = ThreadSafeDict({"a": 1, "b": 2})
        d.clear()
        assert len(d) == 0

    def test_contains(self):
        d: ThreadSafeDict = ThreadSafeDict({"hello": "world"})
        assert "hello" in d
        assert "missing" not in d

    def test_get(self):
        d: ThreadSafeDict = ThreadSafeDict({"k": "v"})
        assert d.get("k") == "v"
        assert d.get("missing", "fallback") == "fallback"

    def test_keys_values_items(self):
        d: ThreadSafeDict = ThreadSafeDict({"a": 1, "b": 2})
        assert set(d.keys()) == {"a", "b"}
        assert set(d.values()) == {1, 2}
        assert set(d.items()) == {("a", 1), ("b", 2)}

    def test_len(self):
        d: ThreadSafeDict = ThreadSafeDict({"x": 1, "y": 2})
        assert len(d) == 2

    def test_iter(self):
        d: ThreadSafeDict = ThreadSafeDict({"a": 1, "b": 2})
        assert set(d) == {"a", "b"}

    def test_copy(self):
        d: ThreadSafeDict = ThreadSafeDict({"a": 1})
        c = d.copy()
        assert isinstance(c, ThreadSafeDict)
        assert c == d

    def test_bool(self):
        assert not ThreadSafeDict()
        assert ThreadSafeDict({"a": 1})

    def test_repr(self):
        d: ThreadSafeDict = ThreadSafeDict({"k": "v"})
        r = repr(d)
        assert "ThreadSafeDict" in r
        assert "'k'" in r

    def test_isinstance_dict(self):
        """ThreadSafeDict must pass isinstance checks for dict."""
        d: ThreadSafeDict = ThreadSafeDict()
        assert isinstance(d, dict)

    def test_popitem(self):
        d: ThreadSafeDict = ThreadSafeDict({"only": 1})
        k, v = d.popitem()
        assert k == "only" and v == 1
        assert len(d) == 0


# ---------------------------------------------------------------------------
# ThreadSafeDict -- concurrent correctness
# ---------------------------------------------------------------------------


class TestThreadSafeDictConcurrent:
    """Stress-test ThreadSafeDict under real multi-threaded contention."""

    def test_concurrent_writes_and_reads(self):
        """Multiple threads writing disjoint keys must not lose data."""
        d: ThreadSafeDict = ThreadSafeDict()
        num_threads = 8
        items_per_thread = 500

        def writer(thread_id: int):
            for i in range(items_per_thread):
                key = f"t{thread_id}_k{i}"
                d[key] = i

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(writer, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)

        assert len(d) == num_threads * items_per_thread

    def test_concurrent_update_same_key(self):
        """All threads writing the same key -- final value must be from one of them."""
        d: ThreadSafeDict = ThreadSafeDict()
        num_threads = 16
        iterations = 200

        def writer(thread_id: int):
            for i in range(iterations):
                d["shared"] = (thread_id, i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(writer, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)

        tid, idx = d["shared"]
        assert 0 <= tid < num_threads
        assert 0 <= idx < iterations

    def test_concurrent_pop(self):
        """Pop from multiple threads -- each key should only be popped once."""
        keys = [f"k{i}" for i in range(500)]
        d: ThreadSafeDict = ThreadSafeDict({k: True for k in keys})
        popped: list = []
        lock = threading.Lock()

        def popper():
            local_popped = []
            for k in keys:
                val = d.pop(k, None)
                if val is not None:
                    local_popped.append(k)
            with lock:
                popped.extend(local_popped)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(popper) for _ in range(8)]
            concurrent.futures.wait(futures)

        # Every key must appear exactly once across all threads.
        assert sorted(popped) == sorted(keys)


# ---------------------------------------------------------------------------
# ThreadSafeCache -- basic LRU behaviour
# ---------------------------------------------------------------------------


class TestThreadSafeCacheBasic:
    """ThreadSafeCache must act as a bounded LRU cache."""

    def test_put_and_get(self):
        cache = ThreadSafeCache(maxsize=10)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_get_missing_returns_default(self):
        cache = ThreadSafeCache(maxsize=10)
        assert cache.get("no_such_key") is None
        assert cache.get("no_such_key", "fallback") == "fallback"

    def test_lru_eviction(self):
        cache = ThreadSafeCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Access "a" so it becomes MRU.
        cache.get("a")
        # Insert "d" -- should evict "b" (LRU).
        cache.put("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key(self):
        cache = ThreadSafeCache(maxsize=5)
        cache.put("k", "old")
        cache.put("k", "new")
        assert cache.get("k") == "new"
        assert len(cache) == 1

    def test_contains(self):
        cache = ThreadSafeCache(maxsize=5)
        cache.put("x", 42)
        assert "x" in cache
        assert "y" not in cache

    def test_invalidate(self):
        cache = ThreadSafeCache(maxsize=5)
        cache.put("k", "v")
        cache.invalidate("k")
        assert cache.get("k") is None
        # Invalidating a missing key is a no-op.
        cache.invalidate("missing")

    def test_clear(self):
        cache = ThreadSafeCache(maxsize=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_stats(self):
        cache = ThreadSafeCache(maxsize=100)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        s = cache.stats()
        assert s["size"] == 1
        assert s["maxsize"] == 100
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_unlimited_maxsize(self):
        """maxsize=0 means unlimited."""
        cache = ThreadSafeCache(maxsize=0)
        for i in range(1000):
            cache.put(i, i)
        assert len(cache) == 1000


# ---------------------------------------------------------------------------
# ThreadSafeCache -- concurrent correctness
# ---------------------------------------------------------------------------


class TestThreadSafeCacheConcurrent:
    """Stress-test ThreadSafeCache under multi-threaded contention."""

    def test_concurrent_put_get(self):
        cache = ThreadSafeCache(maxsize=500)
        num_threads = 8
        items_per_thread = 200
        barrier = threading.Barrier(num_threads)

        def worker(tid: int):
            barrier.wait()
            for i in range(items_per_thread):
                cache.put(f"t{tid}_k{i}", i)
            for i in range(items_per_thread):
                cache.get(f"t{tid}_k{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)
            # Check for exceptions
            for f in futures:
                f.result()

        # We can't assert exact size because eviction may have occurred,
        # but ensure no crash and stats are consistent.
        s = cache.stats()
        assert s["size"] <= 500
        assert s["hits"] + s["misses"] > 0


# ---------------------------------------------------------------------------
# atomic_init
# ---------------------------------------------------------------------------


class TestAtomicInit:
    """atomic_init must guarantee single execution under concurrent calls."""

    def test_single_execution(self):
        call_count = 0

        @atomic_init
        def expensive():
            nonlocal call_count
            call_count += 1
            return "result"

        assert expensive() == "result"
        assert expensive() == "result"
        assert call_count == 1

    def test_concurrent_single_execution(self):
        """Race N threads to call the wrapped function -- exactly one must win."""
        call_count = 0
        lock = threading.Lock()
        barrier = threading.Barrier(16)

        @atomic_init
        def init_resource():
            nonlocal call_count
            with lock:
                call_count += 1
            time.sleep(0.01)  # simulate slow init
            return 42

        results = []
        result_lock = threading.Lock()

        def racer():
            barrier.wait()
            val = init_resource()
            with result_lock:
                results.append(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(racer) for _ in range(16)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        assert call_count == 1
        assert all(r == 42 for r in results)

    def test_exception_cached(self):
        """If the init function raises, the exception is re-raised on every call."""

        @atomic_init
        def failing_init():
            raise ValueError("init failed")

        with pytest.raises(ValueError, match="init failed"):
            failing_init()

        # Second call also raises the same error.
        with pytest.raises(ValueError, match="init failed"):
            failing_init()

    def test_reset(self):
        """After reset(), the function can be called again."""
        call_count = 0

        @atomic_init
        def resettable():
            nonlocal call_count
            call_count += 1
            return call_count

        assert resettable() == 1
        assert resettable() == 1
        resettable.reset()
        assert resettable() == 2

    def test_preserves_function_name(self):
        @atomic_init
        def my_special_func():
            return True

        assert my_special_func.__name__ == "my_special_func"


# ---------------------------------------------------------------------------
# Integration: ThreadSafeDict used in ActionDispatcher pattern
# ---------------------------------------------------------------------------


class TestActionDispatcherPattern:
    """Simulate the ActionDispatcher lazy-init pattern with ThreadSafeDict."""

    def test_class_to_instance_promotion(self):
        """Simulate the lazy class instantiation pattern from action_dispatcher."""

        class FakeAction:
            instance_count = 0

            def __init__(self):
                FakeAction.instance_count += 1

            def run(self):
                return "ok"

        actions: ThreadSafeDict = ThreadSafeDict()
        actions["my_action"] = FakeAction
        init_lock = threading.Lock()

        def lazy_get(name: str):
            fn = actions.get(name)
            if fn is not None and isinstance(fn, type):
                with init_lock:
                    # Double-check
                    fn = actions.get(name)
                    if isinstance(fn, type):
                        instance = fn()
                        actions[name] = instance
                        return instance
                    return fn
            return fn

        # Simulate 8 threads racing to instantiate the same action class.
        barrier = threading.Barrier(8)
        results = []
        result_lock = threading.Lock()

        def racer():
            barrier.wait()
            instance = lazy_get("my_action")
            with result_lock:
                results.append(instance)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(racer) for _ in range(8)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        # The class should have been instantiated exactly once.
        assert FakeAction.instance_count == 1
        # All threads should have received the same instance.
        assert all(r is results[0] for r in results)
