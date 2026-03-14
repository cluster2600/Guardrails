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

"""Tests for the performance optimisations introduced in this PR.

Each test class exercises one of the optimisations:

  1. ``_LRUDict`` — bounded LRU eviction using ``OrderedDict``, including
     the CPython 3.10 ``popitem`` edge case, ``maxsize < 1`` guard, and
     thread-safety under concurrent access.
  2. Per-instance ``process_events`` semaphore — verifies that the
     module-level global semaphore has been removed.
  3. Action name normalisation cache — checks population, hits,
     invalidation on new registrations, and CamelCase handling.
  4. Jinja2 template and variable caching — ensures that
     ``ThreadSafeCache`` stores compiled templates and ``frozenset``
     variable sets, returning the same cached object on repeated lookups.
  5. ``ThreadSafeCache`` — thread-safe bounded LRU from ``_thread_safety``.
  6. Eager task factory — verifies installation on Python 3.12+.
"""

import asyncio
import sys
import threading

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.rails.llm.llmrails import _LRUDict

# ---------------------------------------------------------------------------
# _LRUDict tests
# ---------------------------------------------------------------------------


class TestLRUDict:
    """Tests for the bounded LRU dict."""

    def test_basic_operations(self):
        d = _LRUDict(maxsize=5)
        d["a"] = 1
        d["b"] = 2
        assert d["a"] == 1
        assert d["b"] == 2
        assert len(d) == 2

    def test_eviction_on_overflow(self):
        d = _LRUDict(maxsize=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["d"] = 4  # should evict "a"
        assert "a" not in d
        assert len(d) == 3
        assert d["d"] == 4

    def test_eviction_order(self):
        d = _LRUDict(maxsize=2)
        d["x"] = 10
        d["y"] = 20
        d["z"] = 30  # evicts "x"
        assert "x" not in d
        assert "y" in d
        assert "z" in d

    def test_update_existing_key_no_eviction(self):
        d = _LRUDict(maxsize=2)
        d["a"] = 1
        d["b"] = 2
        d["a"] = 10  # update, should NOT evict
        assert len(d) == 2
        assert d["a"] == 10
        assert d["b"] == 2

    def test_delete(self):
        d = _LRUDict(maxsize=5)
        d["a"] = 1
        del d["a"]
        assert "a" not in d
        assert len(d) == 0

    def test_maxsize_one(self):
        d = _LRUDict(maxsize=1)
        d["a"] = 1
        d["b"] = 2
        assert "a" not in d
        assert d["b"] == 2

    def test_maxsize_zero_raises(self):
        with pytest.raises(ValueError, match="maxsize must be at least 1"):
            _LRUDict(maxsize=0)

    def test_is_dict_subclass(self):
        d = _LRUDict(maxsize=10)
        assert isinstance(d, dict)

    def test_access_promotes_to_mru(self):
        """Reading a key should move it to the MRU position, preventing eviction."""
        d = _LRUDict(maxsize=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        # Access "a" to promote it
        _ = d["a"]
        # Insert "d" — should evict "b" (LRU), not "a"
        d["d"] = 4
        assert "a" in d
        assert "b" not in d
        assert "c" in d
        assert "d" in d

    def test_stress(self):
        d = _LRUDict(maxsize=100)
        for i in range(1000):
            d[f"key_{i}"] = i
        assert len(d) == 100
        # Only the last 100 should remain
        assert "key_999" in d
        assert "key_0" not in d

    def test_has_lock(self):
        """_LRUDict should have a threading lock for free-threaded safety."""
        d = _LRUDict(maxsize=10)
        assert hasattr(d, "_lock")
        assert isinstance(d._lock, type(threading.RLock()))

    def test_concurrent_access(self):
        """Concurrent reads and writes should not corrupt the dict."""
        d = _LRUDict(maxsize=50)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 200):
                    d[f"key_{i}"] = i
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    for key in list(d.keys()):
                        try:
                            _ = d[key]
                        except KeyError:
                            pass  # evicted between keys() and __getitem__
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1000,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(d) <= 50


# ---------------------------------------------------------------------------
# Per-instance semaphore tests
# ---------------------------------------------------------------------------


class TestPerInstanceSemaphore:
    """Tests for per-instance process_events semaphore."""

    def test_no_global_semaphore(self):
        """The module-level global semaphore should no longer exist."""
        import nemoguardrails.rails.llm.llmrails as mod

        assert not hasattr(mod, "process_events_semaphore")


# ---------------------------------------------------------------------------
# Action name normalisation cache tests
# ---------------------------------------------------------------------------


class TestActionNameCache:
    """Tests for the action name normalisation cache."""

    def test_cache_populated_on_first_call(self):
        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="my_action")
        result = d._normalize_action_name("my_action")
        assert result == "my_action"
        assert "my_action" in d._normalised_names

    def test_cache_hit_on_second_call(self):
        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="my_action")
        d._normalize_action_name("my_action")
        # Verify it's cached
        assert d._normalised_names["my_action"] == "my_action"
        # Second call should use cache
        result = d._normalize_action_name("my_action")
        assert result == "my_action"

    def test_cache_cleared_on_register(self):
        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="act_a")
        d._normalize_action_name("act_a")
        assert len(d._normalised_names) > 0
        # Register a new action — cache should be cleared
        d.register_action(lambda **kw: None, name="act_b")
        assert len(d._normalised_names) == 0

    def test_camelcase_normalisation_cached(self):
        d = ActionDispatcher(load_all_actions=False)
        d.register_action(lambda **kw: None, name="my_action")
        result = d._normalize_action_name("MyAction")
        # "MyAction" -> remove "Action" -> "My" -> "my"
        assert "MyAction" in d._normalised_names

    def test_has_lock(self):
        """The normalisation cache should be protected by a lock."""
        d = ActionDispatcher(load_all_actions=False)
        assert hasattr(d, "_normalised_names_lock")

    def test_concurrent_normalisation(self):
        """Concurrent normalisations should not corrupt the cache."""
        d = ActionDispatcher(load_all_actions=False)
        for i in range(100):
            d.register_action(lambda **kw: None, name=f"action_{i}")

        errors = []

        def normalise_many():
            try:
                for i in range(100):
                    d._normalize_action_name(f"action_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=normalise_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent normalisation errors: {errors}"


# ---------------------------------------------------------------------------
# Template caching tests
# ---------------------------------------------------------------------------


class TestTemplateCaching:
    """Tests for Jinja2 template and variable caching."""

    def test_template_cache_populated(self):
        from nemoguardrails.llm.taskmanager import LLMTaskManager
        from nemoguardrails.rails.llm.config import RailsConfig

        config = RailsConfig.from_content(yaml_content="models: []")
        tm = LLMTaskManager(config)

        template_str = "Hello {{ name }}"
        t1 = tm._get_compiled_template(template_str)
        t2 = tm._get_compiled_template(template_str)
        assert t1 is t2  # same object from cache

    def test_variables_cache_populated(self):
        from nemoguardrails.llm.taskmanager import LLMTaskManager
        from nemoguardrails.rails.llm.config import RailsConfig

        config = RailsConfig.from_content(yaml_content="models: []")
        tm = LLMTaskManager(config)

        template_str = "{{ greeting }}, {{ name }}!"
        v1 = tm._get_template_variables(template_str)
        v2 = tm._get_template_variables(template_str)
        assert v1 is v2  # same frozenset object from cache
        assert "greeting" in v1
        assert "name" in v1

    def test_different_templates_cached_separately(self):
        from nemoguardrails.llm.taskmanager import LLMTaskManager
        from nemoguardrails.rails.llm.config import RailsConfig

        config = RailsConfig.from_content(yaml_content="models: []")
        tm = LLMTaskManager(config)

        t1 = tm._get_compiled_template("{{ a }}")
        t2 = tm._get_compiled_template("{{ b }}")
        assert t1 is not t2
        assert len(tm._template_cache) == 2

    def test_template_cache_is_thread_safe(self):
        """Template cache should use ThreadSafeCache from _thread_safety."""
        from nemoguardrails._thread_safety import ThreadSafeCache
        from nemoguardrails.llm.taskmanager import LLMTaskManager
        from nemoguardrails.rails.llm.config import RailsConfig

        config = RailsConfig.from_content(yaml_content="models: []")
        tm = LLMTaskManager(config)

        assert isinstance(tm._template_cache, ThreadSafeCache)
        assert isinstance(tm._variables_cache, ThreadSafeCache)


# ---------------------------------------------------------------------------
# ThreadSafeCache tests
# ---------------------------------------------------------------------------


class TestThreadSafeCache:
    """Tests for ThreadSafeCache from _thread_safety module."""

    def test_basic_put_get(self):
        from nemoguardrails._thread_safety import ThreadSafeCache

        cache = ThreadSafeCache(maxsize=10)
        cache.put("a", 1)
        assert cache.get("a") == 1

    def test_lru_eviction(self):
        from nemoguardrails._thread_safety import ThreadSafeCache

        cache = ThreadSafeCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_access_promotes_to_mru(self):
        from nemoguardrails._thread_safety import ThreadSafeCache

        cache = ThreadSafeCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # promote "a"
        cache.put("d", 4)  # should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_concurrent_access(self):
        from nemoguardrails._thread_safety import ThreadSafeCache

        cache = ThreadSafeCache(maxsize=50)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 200):
                    cache.put(f"key_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(400):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(200,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(cache) <= 50

    def test_stats(self):
        from nemoguardrails._thread_safety import ThreadSafeCache

        cache = ThreadSafeCache(maxsize=10)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


# ---------------------------------------------------------------------------
# Eager task factory tests
# ---------------------------------------------------------------------------


class TestEagerTaskFactory:
    """Tests for the eager task factory installation."""

    @pytest.mark.asyncio
    async def test_eager_factory_installed_on_312_plus(self):
        """On Python 3.12+, the eager task factory should be available."""
        from nemoguardrails.rails.llm.llmrails import _HAS_EAGER_TASK_FACTORY, _ensure_eager_task_factory

        _ensure_eager_task_factory()

        if _HAS_EAGER_TASK_FACTORY:
            loop = asyncio.get_running_loop()
            if hasattr(loop, "get_task_factory"):
                assert loop.get_task_factory() is asyncio.eager_task_factory

    def test_version_flag(self):
        from nemoguardrails.rails.llm.llmrails import _HAS_EAGER_TASK_FACTORY, _PY_VERSION

        assert _PY_VERSION == sys.version_info[:2]
        expected = sys.version_info[:2] >= (3, 12)
        assert _HAS_EAGER_TASK_FACTORY is expected


# ---------------------------------------------------------------------------
# ThreadSafeDict tests
# ---------------------------------------------------------------------------


class TestThreadSafeDict:
    """Tests for ThreadSafeDict from _thread_safety module."""

    def test_basic_operations(self):
        from nemoguardrails._thread_safety import ThreadSafeDict

        d = ThreadSafeDict()
        d["a"] = 1
        assert d["a"] == 1
        assert "a" in d
        assert len(d) == 1

    def test_concurrent_mutations(self):
        from nemoguardrails._thread_safety import ThreadSafeDict

        d = ThreadSafeDict()
        errors = []

        def writer(prefix):
            try:
                for i in range(500):
                    d[f"{prefix}_{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("a",)),
            threading.Thread(target=writer, args=("b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(d) == 1000


# ---------------------------------------------------------------------------
# atomic_init tests
# ---------------------------------------------------------------------------


class TestAtomicInit:
    """Tests for the atomic_init decorator from _thread_safety."""

    def test_called_once(self):
        from nemoguardrails._thread_safety import atomic_init

        call_count = 0

        @atomic_init
        def initialise():
            nonlocal call_count
            call_count += 1
            return 42

        result1 = initialise()
        result2 = initialise()
        assert result1 == 42
        assert result2 == 42
        assert call_count == 1

    def test_concurrent_init(self):
        from nemoguardrails._thread_safety import atomic_init

        call_count = 0

        @atomic_init
        def initialise():
            nonlocal call_count
            call_count += 1
            import time

            time.sleep(0.01)
            return call_count

        results = []
        errors = []

        def run():
            try:
                results.append(initialise())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert call_count == 1
        assert all(r == 1 for r in results)
