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

"""Tests for the CPU-bound thread-pool dispatch (GARDR-20, GARDR-21)."""

import os
import threading

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher
from nemoguardrails.rails.llm.thread_pool import (
    RailThreadPool,
    cpu_bound,
    is_free_threaded,
)

# ---------------------------------------------------------------------------
# @cpu_bound decorator tests
# ---------------------------------------------------------------------------


class TestCpuBoundDecorator:
    """Tests for the @cpu_bound decorator."""

    def test_marks_sync_function(self):
        """@cpu_bound sets _cpu_bound = True on a regular function."""

        @cpu_bound
        def my_func():
            return 42

        assert getattr(my_func, "_cpu_bound", False) is True

    def test_preserves_function_identity(self):
        """The decorator should return the same function object (not a wrapper)."""

        @cpu_bound
        def my_func():
            return 42

        # It should be the exact same function, just with an extra attribute.
        assert my_func() == 42

    def test_rejects_async_function(self):
        """@cpu_bound must raise TypeError for coroutine functions."""

        with pytest.raises(TypeError, match="cannot decorate async function"):

            @cpu_bound
            async def my_async_func():
                return 42

    def test_works_with_arguments(self):
        """Decorated functions should still accept args/kwargs normally."""

        @cpu_bound
        def add(a, b, extra=0):
            return a + b + extra

        assert add(1, 2) == 3
        assert add(1, 2, extra=10) == 13
        assert getattr(add, "_cpu_bound", False) is True


# ---------------------------------------------------------------------------
# RailThreadPool tests
# ---------------------------------------------------------------------------


class TestRailThreadPool:
    """Tests for the RailThreadPool class."""

    def test_default_max_workers(self):
        """Default max_workers should be min(4, cpu_count)."""
        pool = RailThreadPool()
        expected = min(4, os.cpu_count() or 1)
        assert pool.max_workers == expected
        pool.shutdown(wait=True)

    def test_custom_max_workers(self):
        pool = RailThreadPool(max_workers=2)
        assert pool.max_workers == 2
        pool.shutdown(wait=True)

    def test_enabled_by_default(self):
        pool = RailThreadPool()
        assert pool.enabled is True
        pool.shutdown(wait=True)

    def test_disabled_pool(self):
        pool = RailThreadPool(enabled=False)
        assert pool.enabled is False
        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_dispatch_runs_in_thread(self):
        """When enabled, dispatch should run the callable in a worker thread."""
        pool = RailThreadPool(max_workers=1, thread_name_prefix="test-pool")
        try:
            thread_names = []

            def capture_thread():
                thread_names.append(threading.current_thread().name)
                return 99

            result = await pool.dispatch(capture_thread)
            assert result == 99
            assert len(thread_names) == 1
            assert thread_names[0].startswith("test-pool")
        finally:
            pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_dispatch_with_kwargs(self):
        """dispatch() must forward both positional and keyword arguments."""
        pool = RailThreadPool(max_workers=1)
        try:

            def multiply(a, b, factor=1):
                return a * b * factor

            result = await pool.dispatch(multiply, 3, 4, factor=2)
            assert result == 24
        finally:
            pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_dispatch_disabled_runs_inline(self):
        """When disabled, dispatch should call the function inline."""
        pool = RailThreadPool(enabled=False)

        main_thread = threading.current_thread().name
        caller_threads = []

        def note_thread():
            caller_threads.append(threading.current_thread().name)
            return "inline"

        result = await pool.dispatch(note_thread)
        assert result == "inline"
        assert caller_threads[0] == main_thread

    @pytest.mark.asyncio
    async def test_dispatch_after_shutdown_runs_inline(self):
        """After shutdown, dispatch falls back to inline execution."""
        pool = RailThreadPool(max_workers=1)
        pool.shutdown(wait=True)

        result = await pool.dispatch(lambda: 7)
        assert result == 7

    def test_repr(self):
        pool = RailThreadPool(max_workers=2, enabled=True)
        r = repr(pool)
        assert "enabled" in r
        assert "workers=2" in r
        pool.shutdown(wait=True)

    def test_repr_disabled(self):
        pool = RailThreadPool(enabled=False)
        r = repr(pool)
        assert "disabled" in r


# ---------------------------------------------------------------------------
# is_free_threaded() utility
# ---------------------------------------------------------------------------


class TestIsFreethreaded:
    def test_returns_bool(self):
        assert isinstance(is_free_threaded(), bool)


# ---------------------------------------------------------------------------
# ActionDispatcher integration tests
# ---------------------------------------------------------------------------


class TestActionDispatcherThreadPoolIntegration:
    """Test that ActionDispatcher correctly routes @cpu_bound functions."""

    @pytest.mark.asyncio
    async def test_cpu_bound_action_dispatched_to_pool(self):
        """A @cpu_bound action should run in the thread pool."""
        pool = RailThreadPool(max_workers=1, thread_name_prefix="integ-test")
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=pool)

        thread_names = []

        @cpu_bound
        def my_cpu_action(**kwargs):
            thread_names.append(threading.current_thread().name)
            return "done"

        dispatcher.register_action(my_cpu_action, name="my_cpu_action")

        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        assert result == "done"
        assert len(thread_names) == 1
        assert thread_names[0].startswith("integ-test")

        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_cpu_bound_action_no_pool_runs_inline(self):
        """Without a pool, @cpu_bound actions run inline (backward compat)."""
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=None)

        @cpu_bound
        def my_cpu_action(**kwargs):
            return "inline"

        dispatcher.register_action(my_cpu_action, name="my_cpu_action")

        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        assert result == "inline"

    @pytest.mark.asyncio
    async def test_regular_sync_action_not_dispatched(self):
        """A regular (non-cpu_bound) sync action should NOT go through the pool."""
        pool = RailThreadPool(max_workers=1, thread_name_prefix="should-not-see")
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=pool)

        main_thread = threading.current_thread().name
        seen_threads = []

        def regular_action(**kwargs):
            seen_threads.append(threading.current_thread().name)
            return "regular"

        dispatcher.register_action(regular_action, name="regular_action")

        result, status = await dispatcher.execute_action("regular_action", {})
        assert status == "success"
        assert result == "regular"
        # Should have run on the main/event-loop thread, not in the pool.
        assert seen_threads[0] == main_thread

        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_async_action_unaffected(self):
        """Async actions should still work normally, even with a pool configured."""
        pool = RailThreadPool(max_workers=1)
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=pool)

        async def my_async_action(**kwargs):
            return "async_result"

        dispatcher.register_action(my_async_action, name="my_async_action")

        result, status = await dispatcher.execute_action("my_async_action", {})
        assert status == "success"
        assert result == "async_result"

        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_thread_pool_setter(self):
        """The thread_pool can be set after dispatcher construction."""
        dispatcher = ActionDispatcher(load_all_actions=False)
        assert dispatcher.thread_pool is None

        pool = RailThreadPool(max_workers=1, thread_name_prefix="late-attach")
        dispatcher.thread_pool = pool
        assert dispatcher.thread_pool is pool

        thread_names = []

        @cpu_bound
        def my_cpu_action(**kwargs):
            thread_names.append(threading.current_thread().name)
            return "late"

        dispatcher.register_action(my_cpu_action, name="my_cpu_action")
        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert result == "late"
        assert thread_names[0].startswith("late-attach")

        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_cpu_bound_action_with_params(self):
        """@cpu_bound actions receive keyword params correctly."""
        pool = RailThreadPool(max_workers=1)
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=pool)

        @cpu_bound
        def tokenize(text: str = "", **kwargs):
            return len(text.split())

        dispatcher.register_action(tokenize, name="tokenize")

        result, status = await dispatcher.execute_action("tokenize", {"text": "hello world foo"})
        assert status == "success"
        assert result == 3

        pool.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_cpu_bound_action_exception_propagates(self):
        """Exceptions in @cpu_bound actions should propagate and result in 'failed'."""
        pool = RailThreadPool(max_workers=1)
        dispatcher = ActionDispatcher(load_all_actions=False, thread_pool=pool)

        @cpu_bound
        def failing_action(**kwargs):
            raise ValueError("boom")

        dispatcher.register_action(failing_action, name="failing_action")

        result, status = await dispatcher.execute_action("failing_action", {})
        assert status == "failed"
        assert result is None

        pool.shutdown(wait=True)
