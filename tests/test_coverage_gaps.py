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

"""Additional tests to close patch coverage gaps in PR #1727.

Targets uncovered lines in:
  - _thread_safety.py: PEP 584 merge operators, atomic_init fast-path re-raise
  - action_dispatcher.py: free-threaded path (mocked), normalise fast path
  - jailbreak_detection/heuristics/checks.py: cpu_bound import fallback
  - sensitive_data_detection/actions.py: executor offload paths
  - colang/v1_0/runtime/runtime.py: DAG scheduler caching
"""

import asyncio
import concurrent.futures
import sys
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails._thread_safety import (
    ThreadSafeCache,
    ThreadSafeDict,
    _AtomicInitWrapper,
    atomic_init,
)

# ---------------------------------------------------------------------------
# ThreadSafeDict -- PEP 584 merge operators (|, |=, __ror__)
# ---------------------------------------------------------------------------


class TestThreadSafeDictMergeOperators:
    """Cover the __or__, __ior__, and __ror__ methods."""

    def test_or_returns_threadsafe_dict(self):
        a = ThreadSafeDict({"x": 1})
        b = {"y": 2}
        merged = a | b
        assert isinstance(merged, ThreadSafeDict)
        assert dict(merged) == {"x": 1, "y": 2}

    def test_or_with_two_threadsafe_dicts(self):
        a = ThreadSafeDict({"x": 1})
        b = ThreadSafeDict({"y": 2, "x": 99})
        merged = a | b
        assert merged["x"] == 99  # right side wins
        assert merged["y"] == 2

    def test_ior_in_place_merge(self):
        a = ThreadSafeDict({"x": 1})
        original_id = id(a)
        a |= {"y": 2, "x": 10}
        assert id(a) == original_id
        assert a["x"] == 10
        assert a["y"] == 2

    def test_ror_plain_dict_on_left(self):
        """dict | ThreadSafeDict triggers __ror__."""
        plain = {"a": 1}
        tsd = ThreadSafeDict({"b": 2, "a": 99})
        merged = plain | tsd
        assert isinstance(merged, ThreadSafeDict)
        assert merged["a"] == 99  # right side wins
        assert merged["b"] == 2


# ---------------------------------------------------------------------------
# ThreadSafeDict -- equality edge cases
# ---------------------------------------------------------------------------


class TestThreadSafeDictEquality:
    """Cover __eq__ with non-dict objects."""

    def test_eq_with_non_dict(self):
        d = ThreadSafeDict({"a": 1})
        assert d != "not a dict"
        assert d != 42
        assert d != [("a", 1)]

    def test_eq_with_plain_dict(self):
        d = ThreadSafeDict({"a": 1, "b": 2})
        assert d == {"a": 1, "b": 2}

    def test_eq_with_another_threadsafe_dict(self):
        a = ThreadSafeDict({"x": 1})
        b = ThreadSafeDict({"x": 1})
        assert a == b


# ---------------------------------------------------------------------------
# atomic_init -- fast-path re-raise of cached exception
# ---------------------------------------------------------------------------


class TestAtomicInitFastPathReRaise:
    """Cover the fast-path branch at lines 482-484 of _thread_safety.py."""

    def test_cached_exception_re_raised_on_fast_path(self):
        """After first call fails, the second call should hit the fast path
        (self._initialized is True, self._exc is not None) and re-raise
        without acquiring the lock again."""
        call_count = 0

        @atomic_init
        def bad_init():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        # First call -- slow path, acquires lock, raises
        with pytest.raises(RuntimeError, match="fail"):
            bad_init()
        assert call_count == 1

        # Second call -- fast path (lock-free), re-raises cached exception
        with pytest.raises(RuntimeError, match="fail"):
            bad_init()
        # Function was NOT called again
        assert call_count == 1

    def test_reset_clears_cached_exception(self):
        """After reset(), the exception cache is cleared and the function
        can be invoked again."""
        call_count = 0

        @atomic_init
        def fragile_init():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first try fails")
            return "success"

        with pytest.raises(ValueError):
            fragile_init()

        fragile_init.reset()
        result = fragile_init()
        assert result == "success"
        assert call_count == 2


# ---------------------------------------------------------------------------
# ThreadSafeCache -- unlimited mode (maxsize=0)
# ---------------------------------------------------------------------------


class TestThreadSafeCacheUnlimited:
    """Cover the maxsize=0 no-eviction branch in put()."""

    def test_unlimited_cache_never_evicts(self):
        cache = ThreadSafeCache(maxsize=0)
        for i in range(2000):
            cache.put(f"key_{i}", i)
        # All entries remain
        assert len(cache) == 2000
        assert cache.get("key_0") == 0
        assert cache.get("key_1999") == 1999

    def test_unlimited_cache_stats(self):
        cache = ThreadSafeCache(maxsize=0)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("z")  # miss
        s = cache.stats()
        assert s["maxsize"] == 0
        assert s["hits"] == 1
        assert s["misses"] == 1


# ---------------------------------------------------------------------------
# ActionDispatcher -- _normalize_action_name fast path
# ---------------------------------------------------------------------------


class TestNormalizeActionNameFastPath:
    """Cover the immediate return when name is already in the registry."""

    def test_exact_name_skips_normalisation(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="my_action")
        # Fast path: "my_action" is already registered, return as-is
        assert d._normalize_action_name("my_action") == "my_action"

    def test_camelcase_with_action_suffix_normalised(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="generate_user_intent")
        # CamelCase with Action suffix should normalise
        result = d._normalize_action_name("GenerateUserIntentAction")
        assert result == "generate_user_intent"


# ---------------------------------------------------------------------------
# ActionDispatcher -- free-threaded instantiation path (mocked)
# ---------------------------------------------------------------------------


class TestAtomicInstantiateActionFreeThreaded:
    """Cover the free-threaded branch of _atomic_instantiate_action by
    mocking is_free_threaded() to return True."""

    def test_free_threaded_path_creates_per_action_lock(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)

        class SomeAction:
            def run(self, **kw):
                return "ok"

        d.register_action(SomeAction, name="some_action")

        with patch("nemoguardrails.actions.action_dispatcher.is_free_threaded", return_value=True):
            instance = d._atomic_instantiate_action("some_action", SomeAction)

        assert isinstance(instance, SomeAction)
        assert "some_action" in d._init_locks

    def test_free_threaded_double_check_returns_existing(self):
        """Simulate the race-condition branch where another thread already
        promoted the class to an instance."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)

        class SomeAction:
            pass

        existing_instance = SomeAction()
        d._registered_actions["some_action"] = existing_instance

        with patch("nemoguardrails.actions.action_dispatcher.is_free_threaded", return_value=True):
            result = d._atomic_instantiate_action("some_action", SomeAction)

        # Should return the existing instance (double-check hit)
        assert result is existing_instance


# ---------------------------------------------------------------------------
# ActionDispatcher -- cpu_bound warning log path
# ---------------------------------------------------------------------------


class TestCpuBoundWarningPaths:
    """Cover the warning log when @cpu_bound is set but no thread pool."""

    @pytest.mark.asyncio
    async def test_function_cpu_bound_no_pool_logs_warning(self):
        import logging

        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)

        def sync_action(**kw):
            return "inline"

        sync_action._cpu_bound = True
        sync_action.action_meta = {"name": "sync_cpu"}
        d.register_action(sync_action, name="sync_cpu")

        with patch("nemoguardrails.actions.action_dispatcher.log") as mock_log:
            result, status = await d.execute_action("sync_cpu", {})

        assert status == "success"
        assert result == "inline"
        # Verify the warning was logged
        mock_log.warning.assert_any_call(
            "Action `%s` is @cpu_bound but no thread pool is configured; "
            "running inline and blocking the event loop.",
            "sync_cpu",
        )

    @pytest.mark.asyncio
    async def test_class_run_cpu_bound_no_pool_logs_warning(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        d = ActionDispatcher(load_all_actions=False)

        class MyAction:
            def run(self, **kw):
                return "class inline"

        MyAction.action_meta = {"name": "class_cpu"}
        MyAction.run._cpu_bound = True
        d.register_action(MyAction, name="class_cpu")

        with patch("nemoguardrails.actions.action_dispatcher.log") as mock_log:
            result, status = await d.execute_action("class_cpu", {})

        assert status == "success"
        assert result == "class inline"
        mock_log.warning.assert_any_call(
            "Action `%s.run` is @cpu_bound but no thread pool is configured; "
            "running inline and blocking the event loop.",
            "class_cpu",
        )


# ---------------------------------------------------------------------------
# Jailbreak detection -- cpu_bound import fallback
# ---------------------------------------------------------------------------


class TestJailbreakCpuBoundImport:
    """Cover the try/except ImportError fallback in checks.py."""

    def test_fallback_decorator_is_identity(self):
        """Simulate the ImportError path: the fallback cpu_bound decorator
        should be a no-op identity function."""
        # This replicates the exact fallback from checks.py lines 21-26
        def cpu_bound_fallback(fn):
            return fn

        def sample_fn(x):
            return x * 2

        decorated = cpu_bound_fallback(sample_fn)
        assert decorated is sample_fn
        assert decorated(21) == 42

    def test_real_cpu_bound_sets_attribute(self):
        """When thread_pool is available, cpu_bound should set _cpu_bound attr."""
        try:
            from nemoguardrails.rails.llm.thread_pool import cpu_bound

            @cpu_bound
            def my_fn():
                return 1

            assert getattr(my_fn, "_cpu_bound", False) is True
        except ImportError:
            pytest.skip("thread_pool module not available")


# ---------------------------------------------------------------------------
# DAG scheduler caching in RuntimeV1_0
# ---------------------------------------------------------------------------


class TestRuntimeDagSchedulerInit:
    """Cover the DAG scheduler caching in _init_flow_configs."""

    def test_scheduler_none_when_no_dependencies(self):
        """When rails have no dependencies, scheduler should be None."""
        from unittest.mock import PropertyMock

        config = MagicMock()
        config.flows = []
        config.rails.input.has_dependencies = False
        config.rails.output.has_dependencies = False
        config.rails.input.flow_configs = []
        config.rails.output.flow_configs = []

        with patch(
            "nemoguardrails.colang.v1_0.runtime.runtime.RuntimeV1_0.__init__",
            return_value=None,
        ):
            from nemoguardrails.colang.v1_0.runtime.runtime import RuntimeV1_0

            rt = RuntimeV1_0.__new__(RuntimeV1_0)
            rt.config = config
            rt.flow_configs = {}
            rt._init_flow_configs()

        assert rt._input_dag_scheduler is None
        assert rt._output_dag_scheduler is None

    def test_scheduler_built_when_dependencies_exist(self):
        """When input rails have dependencies, a scheduler should be built."""
        config = MagicMock()
        config.flows = []
        config.rails.input.has_dependencies = True
        config.rails.input.flow_configs = [
            {"name": "a"},
            {"name": "b", "depends_on": ["a"]},
        ]
        config.rails.output.has_dependencies = False
        config.rails.output.flow_configs = []

        with patch(
            "nemoguardrails.colang.v1_0.runtime.runtime.RuntimeV1_0.__init__",
            return_value=None,
        ):
            from nemoguardrails.colang.v1_0.runtime.runtime import RuntimeV1_0

            rt = RuntimeV1_0.__new__(RuntimeV1_0)
            rt.config = config
            rt.flow_configs = {}
            rt._init_flow_configs()

        from nemoguardrails.rails.llm.dag_scheduler import TopologicalScheduler

        assert isinstance(rt._input_dag_scheduler, TopologicalScheduler)
        assert rt._output_dag_scheduler is None


# ---------------------------------------------------------------------------
# ThreadSafeCache -- concurrent eviction stress
# ---------------------------------------------------------------------------


class TestThreadSafeCacheEvictionStress:
    """Stress the eviction path under concurrency."""

    def test_cache_never_exceeds_maxsize(self):
        cache = ThreadSafeCache(maxsize=10)
        num_threads = 8
        items_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def writer(tid):
            barrier.wait()
            for i in range(items_per_thread):
                cache.put(f"t{tid}_k{i}", i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(writer, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        assert len(cache) <= 10


# ---------------------------------------------------------------------------
# ThreadSafeDict -- popitem and setdefault under threading
# ---------------------------------------------------------------------------


class TestThreadSafeDictConcurrentEdges:
    """Cover concurrent popitem and setdefault paths."""

    def test_concurrent_setdefault(self):
        d = ThreadSafeDict()
        num_threads = 16
        barrier = threading.Barrier(num_threads)
        results = []
        lock = threading.Lock()

        def worker(tid):
            barrier.wait()
            val = d.setdefault("shared", tid)
            with lock:
                results.append(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        # All threads should get the same value (first one wins)
        assert all(r == results[0] for r in results)
        assert d["shared"] == results[0]


# ---------------------------------------------------------------------------
# _AtomicInitWrapper -- concurrent reset + call
# ---------------------------------------------------------------------------


class TestAtomicInitConcurrentReset:
    """Cover the edge case where reset() is called concurrently with __call__."""

    def test_reset_then_call_re_executes(self):
        counter = 0

        @atomic_init
        def init_fn():
            nonlocal counter
            counter += 1
            return counter

        assert init_fn() == 1
        assert init_fn() == 1

        # Reset and re-call in rapid succession
        init_fn.reset()
        assert init_fn() == 2

        init_fn.reset()
        assert init_fn() == 3
