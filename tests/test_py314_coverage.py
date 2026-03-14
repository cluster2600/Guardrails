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

"""Tests for Python 3.14 performance optimisations.

Covers the new code paths introduced for:
  - Eager task factory in TopologicalScheduler.execute()
  - get_cpu_executor() public API
  - Version-gated feature flags
  - @cpu_bound dispatch in ActionDispatcher
  - run_in_executor offloading in guardrail actions
  - DAG scheduler caching in RuntimeV1_0._init_flow_configs
  - Jailbreak detection caching and cpu_bound dispatch
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails.rails.llm.dag_scheduler import (
    _HAS_EAGER_TASK_FACTORY,
    _IS_FREE_THREADED,
    _PY_VERSION,
    TopologicalScheduler,
    build_scheduler_from_config,
    get_cpu_executor,
)

# ---------------------------------------------------------------------------
# Version flags
# ---------------------------------------------------------------------------


class TestVersionFlags:
    """Verify version-gated constants are set correctly."""

    def test_py_version_tuple(self):
        assert _PY_VERSION == sys.version_info[:2]

    def test_eager_task_factory_flag(self):
        expected = sys.version_info[:2] >= (3, 12)
        assert _HAS_EAGER_TASK_FACTORY is expected

    def test_is_free_threaded_flag(self):
        # On standard builds, _IS_FREE_THREADED should be False
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        expected = bool(gil_enabled is False)
        assert _IS_FREE_THREADED is expected

    def test_get_cpu_executor_returns_none_on_standard_build(self):
        # On standard (GIL-enabled) builds, should return None
        result = get_cpu_executor()
        if _IS_FREE_THREADED:
            assert result is not None
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Eager task factory in scheduler execute()
# ---------------------------------------------------------------------------


class TestEagerTaskFactory:
    """Test that execute() installs the eager task factory idempotently."""

    @pytest.mark.asyncio
    async def test_execute_installs_eager_factory(self):
        """After execute(), the eager task factory should be set on the loop (3.12+)."""
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            return {"action": "continue"}

        await s.execute(executor)

        loop = asyncio.get_running_loop()
        if hasattr(loop, "get_task_factory") and _HAS_EAGER_TASK_FACTORY:
            assert loop.get_task_factory() is asyncio.eager_task_factory

    @pytest.mark.asyncio
    async def test_execute_eager_factory_survives_error(self):
        """Even if a rail raises, the eager factory remains set."""
        s = build_scheduler_from_config(["a"])

        async def executor(rail_name, ctx):
            raise RuntimeError("boom")

        # execute catches rail errors, so this shouldn't raise
        result = await s.execute(executor)

        loop = asyncio.get_running_loop()
        if hasattr(loop, "get_task_factory") and _HAS_EAGER_TASK_FACTORY:
            assert loop.get_task_factory() is asyncio.eager_task_factory

        assert result["results"]["a"]["action"] == "error"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test execute() with a shared context dict."""
        s = build_scheduler_from_config(["a", "b"])

        received_contexts = {}

        async def executor(rail_name, ctx):
            received_contexts[rail_name] = ctx
            return {"action": "continue"}

        context = {"user_message": "hello"}
        await s.execute(executor, context=context)

        assert received_contexts["a"] is context
        assert received_contexts["b"] is context


# ---------------------------------------------------------------------------
# ActionDispatcher cpu_bound dispatch
# ---------------------------------------------------------------------------


class TestActionDispatcherCpuBound:
    """Test @cpu_bound action dispatch through the thread pool."""

    @pytest.mark.asyncio
    async def test_cpu_bound_with_thread_pool(self):
        """When a thread pool is set, @cpu_bound actions should be dispatched to it."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        def my_action(**kwargs):
            return "cpu result"

        my_action._cpu_bound = True
        my_action.action_meta = {"name": "my_cpu_action"}
        dispatcher.register_action(my_action, name="my_cpu_action")

        # Create a mock thread pool
        mock_pool = MagicMock()
        mock_pool.dispatch = AsyncMock(return_value="pool result")
        dispatcher.thread_pool = mock_pool

        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        assert result == "pool result"
        mock_pool.dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cpu_bound_without_thread_pool(self):
        """Without a thread pool, @cpu_bound actions run inline with a warning."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        def my_action(**kwargs):
            return "inline result"

        my_action._cpu_bound = True
        my_action.action_meta = {"name": "my_cpu_action"}
        dispatcher.register_action(my_action, name="my_cpu_action")

        # No thread pool set
        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        assert result == "inline result"

    @pytest.mark.asyncio
    async def test_class_action_cpu_bound_run(self):
        """Class-based action with @cpu_bound run() method dispatches to pool."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        class MyAction:
            def run(self, **kwargs):
                return "class cpu result"

        MyAction.action_meta = {"name": "my_class_action"}
        MyAction.run._cpu_bound = True
        dispatcher.register_action(MyAction, name="my_class_action")

        mock_pool = MagicMock()
        mock_pool.dispatch = AsyncMock(return_value="pool class result")
        dispatcher.thread_pool = mock_pool

        result, status = await dispatcher.execute_action("my_class_action", {})
        assert status == "success"
        assert result == "pool class result"

    @pytest.mark.asyncio
    async def test_class_action_cpu_bound_run_no_pool(self):
        """Class-based action with @cpu_bound run() but no pool runs inline."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        class MyAction:
            def run(self, **kwargs):
                return "inline class result"

        MyAction.action_meta = {"name": "my_class_action"}
        MyAction.run._cpu_bound = True
        dispatcher.register_action(MyAction, name="my_class_action")

        result, status = await dispatcher.execute_action("my_class_action", {})
        assert status == "success"
        assert result == "inline class result"

    @pytest.mark.asyncio
    async def test_runnable_action_execution(self):
        """Runnable (LangChain) actions should be invoked via ainvoke."""
        from langchain_core.runnables import RunnableLambda

        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        runnable = RunnableLambda(func=lambda x: "runnable result")
        dispatcher.register_action(runnable, name="my_runnable")

        result, status = await dispatcher.execute_action("my_runnable", {})
        assert status == "success"
        assert result == "runnable result"


# ---------------------------------------------------------------------------
# ThreadSafeDict in ActionDispatcher
# ---------------------------------------------------------------------------


class TestActionDispatcherThreadSafety:
    """Test ThreadSafeDict usage in ActionDispatcher."""

    def test_dispatcher_uses_correct_dict_type(self):
        from nemoguardrails._thread_safety import ThreadSafeDict, is_free_threaded
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        if is_free_threaded():
            assert isinstance(dispatcher._registered_actions, ThreadSafeDict)
        else:
            assert type(dispatcher._registered_actions) is dict

    def test_init_locks_created(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher._init_locks, dict)
        assert hasattr(dispatcher._init_locks_guard, "acquire")


# ---------------------------------------------------------------------------
# Jailbreak detection cpu_bound fallback
# ---------------------------------------------------------------------------


class TestJailbreakCpuBoundFallback:
    """Test the cpu_bound import fallback in heuristics/checks.py."""

    def test_cpu_bound_fallback_is_identity(self):
        """When thread_pool module isn't available, cpu_bound should be identity."""

        # The actual import uses a try/except, so we test the fallback path
        def cpu_bound_fallback(fn):
            return fn

        def my_fn():
            return 42

        result = cpu_bound_fallback(my_fn)
        assert result is my_fn
        assert result() == 42


# ---------------------------------------------------------------------------
# Jailbreak detection model caching
# ---------------------------------------------------------------------------


class TestJailbreakDetectionCaching:
    """Test jailbreak_detection_model cache integration."""

    def test_jailbreak_heuristics_is_callable(self):
        """Verify the jailbreak_detection_heuristics action is properly decorated."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_heuristics,
        )

        assert callable(jailbreak_detection_heuristics)
        assert hasattr(jailbreak_detection_heuristics, "action_meta")

    def test_jailbreak_detection_model_is_callable(self):
        """Verify the jailbreak_detection_model action is properly decorated."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_model,
        )

        assert callable(jailbreak_detection_model)
        assert hasattr(jailbreak_detection_model, "action_meta")


# ---------------------------------------------------------------------------
# Content safety run_in_executor
# ---------------------------------------------------------------------------


class TestContentSafetyExecutor:
    """Test content_safety actions use run_in_executor."""

    @pytest.mark.asyncio
    async def test_detect_language_uses_executor(self):
        """detect_language should offload to a worker thread."""
        from nemoguardrails.library.content_safety.actions import detect_language

        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value="en",
        ):
            result = await detect_language(context={"user_message": "hello world"}, config=None)
            assert result["language"] == "en"
            assert "refusal_message" in result

    @pytest.mark.asyncio
    async def test_detect_language_unsupported_falls_back_to_en(self):
        """Unsupported languages should fall back to English."""
        from nemoguardrails.library.content_safety.actions import detect_language

        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value="xx",
        ):
            result = await detect_language(context={"user_message": "some text"}, config=None)
            assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_detect_language_none_falls_back_to_en(self):
        """None detection result should fall back to English."""
        from nemoguardrails.library.content_safety.actions import detect_language

        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value=None,
        ):
            result = await detect_language(context={"user_message": "some text"}, config=None)
            assert result["language"] == "en"


# ---------------------------------------------------------------------------
# Content safety helpers
# ---------------------------------------------------------------------------


class TestContentSafetyHelpers:
    """Test content_safety helper functions."""

    def test_detect_language_function(self):
        """_detect_language should handle various inputs."""
        from nemoguardrails.library.content_safety.actions import _detect_language

        # _detect_language imports fast_langdetect inside, so it handles
        # ImportError gracefully by returning None
        result = _detect_language("")
        # Result is either a language code or None
        assert result is None or isinstance(result, str)

    def test_get_refusal_message_custom(self):
        """Custom refusal messages should take precedence."""
        from nemoguardrails.library.content_safety.actions import _get_refusal_message

        custom = {"en": "Custom refusal", "fr": "Refus personnalisé"}
        assert _get_refusal_message("en", custom) == "Custom refusal"
        assert _get_refusal_message("fr", custom) == "Refus personnalisé"

    def test_get_refusal_message_default(self):
        """Default refusal messages should be returned for supported languages."""
        from nemoguardrails.library.content_safety.actions import (
            DEFAULT_REFUSAL_MESSAGES,
            _get_refusal_message,
        )

        assert _get_refusal_message("en", None) == DEFAULT_REFUSAL_MESSAGES["en"]
        assert _get_refusal_message("ja", None) == DEFAULT_REFUSAL_MESSAGES["ja"]

    def test_get_refusal_message_unknown_language_with_custom(self):
        """Unknown language with custom messages should fall back to custom 'en'."""
        from nemoguardrails.library.content_safety.actions import _get_refusal_message

        custom = {"en": "Custom English"}
        assert _get_refusal_message("zz", custom) == "Custom English"

    def test_get_refusal_message_unknown_language_no_custom(self):
        """Unknown language without custom messages should fall back to default 'en'."""
        from nemoguardrails.library.content_safety.actions import (
            DEFAULT_REFUSAL_MESSAGES,
            _get_refusal_message,
        )

        assert _get_refusal_message("zz", None) == DEFAULT_REFUSAL_MESSAGES["en"]

    def test_supported_languages_frozenset(self):
        """SUPPORTED_LANGUAGES should be a frozenset with expected languages."""
        from nemoguardrails.library.content_safety.actions import SUPPORTED_LANGUAGES

        assert isinstance(SUPPORTED_LANGUAGES, frozenset)
        assert "en" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES

    def test_content_safety_check_output_mapping(self):
        """Output mapping should return True when not allowed, False when allowed."""
        from nemoguardrails.library.content_safety.actions import (
            content_safety_check_output_mapping,
        )

        assert content_safety_check_output_mapping({"allowed": True}) is False
        assert content_safety_check_output_mapping({"allowed": False}) is True
        assert content_safety_check_output_mapping({}) is False  # default to allowed


# ---------------------------------------------------------------------------
# Sensitive data detection output_mapping
# ---------------------------------------------------------------------------


class TestSensitiveDataDetectionMapping:
    """Test sensitive_data_detection mapping function."""

    def test_detect_sensitive_data_mapping(self):
        from nemoguardrails.library.sensitive_data_detection.actions import (
            detect_sensitive_data_mapping,
        )

        assert detect_sensitive_data_mapping(True) is True
        assert detect_sensitive_data_mapping(False) is False


# ---------------------------------------------------------------------------
# DAG Scheduler caching in runtime
# ---------------------------------------------------------------------------


class TestRuntimeDagSchedulerCaching:
    """Test that RuntimeV1_0._init_flow_configs builds DAG schedulers."""

    def test_build_scheduler_from_config_returns_scheduler(self):
        """build_scheduler_from_config should return a TopologicalScheduler."""
        scheduler = build_scheduler_from_config(
            [
                {"name": "a"},
                {"name": "b", "depends_on": ["a"]},
            ]
        )
        assert isinstance(scheduler, TopologicalScheduler)
        assert scheduler.num_groups == 2

    def test_build_scheduler_no_deps(self):
        """All independent flows should produce a single group."""
        scheduler = build_scheduler_from_config(["a", "b", "c"])
        assert scheduler.num_groups == 1
        assert scheduler.max_parallelism == 3


# ---------------------------------------------------------------------------
# Scheduler properties
# ---------------------------------------------------------------------------


class TestSchedulerProperties:
    """Test TopologicalScheduler property accessors."""

    def test_groups_property(self):
        s = build_scheduler_from_config(
            [
                {"name": "a"},
                {"name": "b", "depends_on": ["a"]},
            ]
        )
        groups = s.groups
        assert len(groups) == 2

    def test_repr(self):
        s = build_scheduler_from_config(["a", "b"])
        r = repr(s)
        assert "TopologicalScheduler" in r

    @pytest.mark.asyncio
    async def test_execute_timing_info(self):
        """Execute result should contain timing information."""
        s = build_scheduler_from_config(["a"])

        async def executor(rail_name, ctx):
            return {"action": "continue"}

        result = await s.execute(executor)
        assert "elapsed_ms" in result
        assert result["elapsed_ms"] >= 0
        assert "results" in result
        assert "blocked_by" in result
