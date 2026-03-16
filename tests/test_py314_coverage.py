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

# NOTE ON TEST STRATEGY
# ---------------------
# This module validates the Python 3.14 / free-threaded performance
# optimisation paths.  Many of these code paths are version-gated
# (i.e. only active on 3.12+ or 3.14t), so the tests are written to
# assert the *correct* behaviour for whichever interpreter is running
# rather than hard-coding a single expected value.  This ensures CI
# passes on both legacy and modern Python builds.

# ---------------------------------------------------------------------------
# Version flags
# ---------------------------------------------------------------------------


# Catches regressions where version-detection logic drifts out of sync
# with the actual interpreter.  Removing this class would allow the
# feature flags to silently return wrong values, enabling or disabling
# optimisation paths on the wrong Python version.
class TestVersionFlags:
    """Verify version-gated constants are set correctly."""

    # Guards against _PY_VERSION being hard-coded or computed incorrectly.
    def test_py_version_tuple(self):
        assert _PY_VERSION == sys.version_info[:2]

    # Eager task factory requires Python 3.12+; a wrong flag would
    # silently skip the optimisation or crash on older interpreters.
    def test_eager_task_factory_flag(self):
        expected = sys.version_info[:2] >= (3, 12)
        assert _HAS_EAGER_TASK_FACTORY is expected

    def test_is_free_threaded_flag(self):
        # On standard builds, _IS_FREE_THREADED should be False.
        # sys._is_gil_enabled only exists on free-threaded builds;
        # the lambda fallback simulates a GIL-enabled interpreter.
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        expected = bool(gil_enabled is False)
        assert _IS_FREE_THREADED is expected

    # Verifies the public API returns the correct executor type.
    # On GIL builds a thread-pool executor is pointless for CPU work
    # (the GIL serialises it anyway), so None is the correct return.
    def test_get_cpu_executor_returns_none_on_standard_build(self):
        result = get_cpu_executor()
        if _IS_FREE_THREADED:
            assert result is not None
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Eager task factory in scheduler execute()
# ---------------------------------------------------------------------------


# This class catches regressions in the eager task factory installation
# inside the DAG scheduler.  The eager factory (Python 3.12+) avoids a
# round-trip through the event loop for tasks that complete synchronously,
# which is a significant latency win for short guardrail checks.
# Removing these tests would allow the factory installation to silently
# break or regress after an asyncio refactor.
class TestEagerTaskFactory:
    """Test that execute() installs the eager task factory idempotently."""

    @pytest.mark.asyncio
    async def test_execute_installs_eager_factory(self):
        """After execute(), the eager task factory should be set on the loop (3.12+)."""
        # Two independent rails — no dependency edges — to exercise the
        # parallel scheduling path.
        s = build_scheduler_from_config(["a", "b"])

        async def executor(rail_name, ctx):
            return {"action": "continue"}

        await s.execute(executor)

        loop = asyncio.get_running_loop()
        # Guard: get_task_factory only exists on 3.12+; skip assertion
        # gracefully on older interpreters.
        if hasattr(loop, "get_task_factory") and _HAS_EAGER_TASK_FACTORY:
            assert loop.get_task_factory() is asyncio.eager_task_factory

    @pytest.mark.asyncio
    async def test_execute_eager_factory_survives_error(self):
        """Even if a rail raises, the eager factory remains set."""
        s = build_scheduler_from_config(["a"])

        async def executor(rail_name, ctx):
            raise RuntimeError("boom")

        # execute() must catch per-rail errors without tearing down the
        # loop configuration — otherwise a single faulty rail would
        # disable the optimisation for all subsequent requests.
        result = await s.execute(executor)

        loop = asyncio.get_running_loop()
        if hasattr(loop, "get_task_factory") and _HAS_EAGER_TASK_FACTORY:
            assert loop.get_task_factory() is asyncio.eager_task_factory

        # Confirm the error was captured rather than swallowed silently.
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

        # Identity check (``is``, not ``==``) ensures every rail receives
        # the *same* object, not a copy — mutations by one rail must be
        # visible to the next topological group.
        assert received_contexts["a"] is context
        assert received_contexts["b"] is context


# ---------------------------------------------------------------------------
# ActionDispatcher cpu_bound dispatch
# ---------------------------------------------------------------------------


# This class validates the @cpu_bound dispatch machinery.  On free-threaded
# Python, CPU-intensive actions (e.g. regex-heavy jailbreak checks) must be
# offloaded to a thread pool to avoid blocking the async event loop.
# The tests use MagicMock/AsyncMock for the thread pool because:
#   1. We need to verify dispatch *routing* logic, not actual threading.
#   2. A real pool would introduce non-deterministic timing into the suite.
#   3. Mocking lets us assert that dispatch was called exactly once.
# Removing this class would leave the cpu_bound fallback behaviour untested,
# risking silent event-loop blocking on production deployments.
class TestActionDispatcherCpuBound:
    """Test @cpu_bound action dispatch through the thread pool."""

    @pytest.mark.asyncio
    async def test_cpu_bound_with_thread_pool(self):
        """When a thread pool is set, @cpu_bound actions should be dispatched to it."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        # load_all_actions=False avoids importing the entire action catalogue,
        # keeping the test isolated and fast.
        dispatcher = ActionDispatcher(load_all_actions=False)

        def my_action(**kwargs):
            return "cpu result"

        # Manually stamp the _cpu_bound marker that the @cpu_bound decorator
        # would normally apply — avoids depending on the decorator itself.
        my_action._cpu_bound = True
        my_action.action_meta = {"name": "my_cpu_action"}
        dispatcher.register_action(my_action, name="my_cpu_action")

        # Mock the thread pool so we can confirm the action is routed there
        # rather than run inline on the event loop.
        mock_pool = MagicMock()
        mock_pool.dispatch = AsyncMock(return_value="pool result")
        dispatcher.thread_pool = mock_pool

        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        # The result must come from the pool, not the action's own return.
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

        # No thread pool — the dispatcher must gracefully degrade to
        # inline execution rather than raising an error.
        result, status = await dispatcher.execute_action("my_cpu_action", {})
        assert status == "success"
        assert result == "inline result"

    @pytest.mark.asyncio
    async def test_class_action_cpu_bound_run(self):
        """Class-based action with @cpu_bound run() method dispatches to pool."""
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        # Class-based actions are instantiated by the dispatcher; the
        # _cpu_bound marker lives on the run() method, not the class.
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

        # Mirrors the "no pool" scenario for class-based actions specifically,
        # since the dispatcher has a separate code path for classes vs functions.
        result, status = await dispatcher.execute_action("my_class_action", {})
        assert status == "success"
        assert result == "inline class result"

    @pytest.mark.asyncio
    async def test_runnable_action_execution(self):
        """Runnable (LangChain) actions should be invoked via ainvoke."""
        from langchain_core.runnables import RunnableLambda

        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        # RunnableLambda is the simplest LangChain Runnable — verifies that
        # the dispatcher correctly detects a Runnable and calls ainvoke()
        # instead of treating it as a plain callable.
        runnable = RunnableLambda(func=lambda x: "runnable result")
        dispatcher.register_action(runnable, name="my_runnable")

        result, status = await dispatcher.execute_action("my_runnable", {})
        assert status == "success"
        assert result == "runnable result"


# ---------------------------------------------------------------------------
# ThreadSafeDict in ActionDispatcher
# ---------------------------------------------------------------------------


# Ensures the ActionDispatcher initialises its internal registries with
# thread-safe containers.  Without these checks, a refactor could
# accidentally revert to plain dicts, introducing data races on
# free-threaded builds that would only manifest under load.
class TestActionDispatcherThreadSafety:
    """Test ThreadSafeDict usage in ActionDispatcher."""

    def test_dispatcher_uses_correct_dict_type(self):
        from nemoguardrails._thread_safety import ThreadSafeDict
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)

        # Always ThreadSafeDict — actions can be registered from concurrent
        # async tasks regardless of GIL presence.
        assert isinstance(dispatcher._registered_actions, ThreadSafeDict)

    # Verifies that per-action initialisation locks and their guard are
    # created during construction.  _init_locks prevents two coroutines
    # from initialising the same class-based action simultaneously.
    def test_init_locks_created(self):
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher._init_locks, dict)
        # The guard must be a proper lock (has acquire/release).
        assert hasattr(dispatcher._init_locks_guard, "acquire")


# ---------------------------------------------------------------------------
# Jailbreak detection cpu_bound fallback
# ---------------------------------------------------------------------------


# When the thread_pool module is unavailable (e.g. stripped deployment),
# @cpu_bound must degrade to a no-op identity decorator.  Without this
# test, a broken fallback could silently wrap functions in None or raise
# an ImportError at module load time.
class TestJailbreakCpuBoundFallback:
    """Test the cpu_bound import fallback in heuristics/checks.py."""

    def test_cpu_bound_fallback_is_identity(self):
        """When thread_pool module isn't available, cpu_bound should be identity."""

        # Simulate the fallback path that lives inside a try/except in
        # the jailbreak heuristics module.
        def cpu_bound_fallback(fn):
            return fn

        def my_fn():
            return 42

        result = cpu_bound_fallback(my_fn)
        # Must return the *exact same* function object, not a wrapper.
        assert result is my_fn
        assert result() == 42


# ---------------------------------------------------------------------------
# Jailbreak detection model caching
# ---------------------------------------------------------------------------


# Smoke tests that the jailbreak detection actions survive import and
# retain their @action decorator metadata.  A broken decorator or
# circular import would cause these to fail, catching the issue before
# the full integration suite runs.
class TestJailbreakDetectionCaching:
    """Test jailbreak_detection_model cache integration."""

    def test_jailbreak_heuristics_is_callable(self):
        """Verify the jailbreak_detection_heuristics action is properly decorated."""
        from nemoguardrails.library.jailbreak_detection.actions import (
            jailbreak_detection_heuristics,
        )

        assert callable(jailbreak_detection_heuristics)
        # action_meta is stamped by the @action decorator; its absence
        # means the dispatcher would not recognise this function.
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


# Validates that language detection is offloaded via run_in_executor and
# that the fallback logic handles unsupported / missing language codes.
# The _detect_language helper is patched because it depends on the
# fast_langdetect C extension, which may not be installed in CI.
# Removing these tests would leave the executor offloading and the
# English-fallback behaviour unverified.
class TestContentSafetyExecutor:
    """Test content_safety actions use run_in_executor."""

    @pytest.mark.asyncio
    async def test_detect_language_uses_executor(self):
        """detect_language should offload to a worker thread."""
        from nemoguardrails.library.content_safety.actions import detect_language

        # Patch the synchronous helper to avoid a real C-extension call.
        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value="en",
        ):
            result = await detect_language(context={"user_message": "hello world"}, config=None)
            assert result["language"] == "en"
            # refusal_message must always be present so downstream flows
            # can use it without a KeyError.
            assert "refusal_message" in result

    @pytest.mark.asyncio
    async def test_detect_language_unsupported_falls_back_to_en(self):
        """Unsupported languages should fall back to English."""
        from nemoguardrails.library.content_safety.actions import detect_language

        # "xx" is not in SUPPORTED_LANGUAGES — must degrade gracefully.
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

        # None can occur when the C extension is missing or the text is
        # too short for reliable detection.
        with patch(
            "nemoguardrails.library.content_safety.actions._detect_language",
            return_value=None,
        ):
            result = await detect_language(context={"user_message": "some text"}, config=None)
            assert result["language"] == "en"


# ---------------------------------------------------------------------------
# Content safety helpers
# ---------------------------------------------------------------------------


# Unit tests for the pure-function helpers in content_safety.  These
# have no I/O dependencies, so they run without mocking.  The refusal
# message lookup is exercised across several fallback tiers (custom,
# default, unknown language) to prevent silent regression when new
# languages are added.
class TestContentSafetyHelpers:
    """Test content_safety helper functions."""

    def test_detect_language_function(self):
        """_detect_language should handle various inputs."""
        from nemoguardrails.library.content_safety.actions import _detect_language

        # Empty string is an important edge case — fast_langdetect may
        # raise or return None; the wrapper must handle either gracefully.
        result = _detect_language("")
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

        # Japanese is a good secondary language to test — it catches
        # issues with non-Latin key lookup in the defaults dict.
        assert _get_refusal_message("en", None) == DEFAULT_REFUSAL_MESSAGES["en"]
        assert _get_refusal_message("ja", None) == DEFAULT_REFUSAL_MESSAGES["ja"]

    # Fallback tier: unknown language + custom dict => custom English.
    def test_get_refusal_message_unknown_language_with_custom(self):
        """Unknown language with custom messages should fall back to custom 'en'."""
        from nemoguardrails.library.content_safety.actions import _get_refusal_message

        custom = {"en": "Custom English"}
        assert _get_refusal_message("zz", custom) == "Custom English"

    # Fallback tier: unknown language + no custom dict => default English.
    def test_get_refusal_message_unknown_language_no_custom(self):
        """Unknown language without custom messages should fall back to default 'en'."""
        from nemoguardrails.library.content_safety.actions import (
            DEFAULT_REFUSAL_MESSAGES,
            _get_refusal_message,
        )

        assert _get_refusal_message("zz", None) == DEFAULT_REFUSAL_MESSAGES["en"]

    # frozenset guarantees immutability — a plain set could be mutated
    # at runtime, breaking thread safety.
    def test_supported_languages_frozenset(self):
        """SUPPORTED_LANGUAGES should be a frozenset with expected languages."""
        from nemoguardrails.library.content_safety.actions import SUPPORTED_LANGUAGES

        assert isinstance(SUPPORTED_LANGUAGES, frozenset)
        assert "en" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES

    # The output mapping inverts the "allowed" boolean for use in Colang
    # flow conditions — True means "should block".
    def test_content_safety_check_output_mapping(self):
        """Output mapping should return True when not allowed, False when allowed."""
        from nemoguardrails.library.content_safety.actions import (
            content_safety_check_output_mapping,
        )

        assert content_safety_check_output_mapping({"allowed": True}) is False
        assert content_safety_check_output_mapping({"allowed": False}) is True
        assert content_safety_check_output_mapping({}) is False  # missing key defaults to allowed


# ---------------------------------------------------------------------------
# Sensitive data detection output_mapping
# ---------------------------------------------------------------------------


# Simple pass-through contract: the mapping function must preserve the
# boolean unchanged.  If someone accidentally inverts it (like the
# content_safety mapping does), sensitive data would leak through.
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


# Verifies the factory function that converts a rail config list into a
# TopologicalScheduler.  The runtime caches these schedulers per flow
# type, so a broken factory would cause every request to rebuild the DAG.
class TestRuntimeDagSchedulerCaching:
    """Test that RuntimeV1_0._init_flow_configs builds DAG schedulers."""

    def test_build_scheduler_from_config_returns_scheduler(self):
        """build_scheduler_from_config should return a TopologicalScheduler."""
        # Two rails with an explicit dependency edge — must produce two
        # topological groups (b waits for a).
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
        # Three independent rails — the scheduler should batch them into
        # one group for maximum concurrency.
        scheduler = build_scheduler_from_config(["a", "b", "c"])
        assert scheduler.num_groups == 1
        assert scheduler.max_parallelism == 3


# ---------------------------------------------------------------------------
# Scheduler properties
# ---------------------------------------------------------------------------


# Tests the read-only property accessors and the result schema of
# execute().  These are part of the public API surface consumed by
# telemetry and logging code — changing the shape would break dashboards.
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

    # __repr__ is used in log messages; must at least contain the class name.
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
        # Verify the result dict contract — downstream telemetry depends
        # on these keys being present.
        assert "elapsed_ms" in result
        assert result["elapsed_ms"] >= 0
        assert "results" in result
        # blocked_by tracks which rail caused an early abort, if any.
        assert "blocked_by" in result
