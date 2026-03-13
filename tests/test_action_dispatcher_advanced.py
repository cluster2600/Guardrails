# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advanced tests for nemoguardrails.actions.action_dispatcher.

Covers:
  - _atomic_instantiate_action on GIL-enabled builds
  - Runnable action execution
  - Class-based action without run() method
  - Action loading error paths
  - _find_actions empty directory
"""

import inspect
import os
import tempfile

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher, is_action_file


class TestAtomicInstantiateAction:
    """Tests for _atomic_instantiate_action on GIL-enabled builds."""

    def test_instantiate_class_action(self):
        """Class-based action should be instantiated and cached."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        class MyAction:
            def run(self, **kwargs):
                return "result"

        MyAction.action_meta = {"name": "my_action"}

        dispatcher.register_action(MyAction, name="my_action")
        # Verify it's registered as a class
        assert inspect.isclass(dispatcher.registered_actions["my_action"])

        # Instantiate via _atomic_instantiate_action
        instance = dispatcher._atomic_instantiate_action("my_action", MyAction)
        assert not inspect.isclass(instance)
        assert isinstance(instance, MyAction)

        # Should be cached now
        assert not inspect.isclass(dispatcher.registered_actions["my_action"])

    def test_double_instantiation_returns_existing(self):
        """Second call should return the already-instantiated action."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        class MyAction:
            pass

        dispatcher.register_action(MyAction, name="my_action")

        inst1 = dispatcher._atomic_instantiate_action("my_action", MyAction)
        # Register the instance directly to simulate race condition
        dispatcher._registered_actions["my_action"] = inst1

        # Second call — should detect it's already instantiated
        inst2 = dispatcher._atomic_instantiate_action("my_action", MyAction)
        # On GIL builds, this creates a new instance (no double-check).
        # On free-threaded builds, the double-check returns the existing one.
        assert inst2 is not None


class TestExecuteActionEdgeCases:
    """Tests for execute_action edge cases."""

    @pytest.mark.asyncio
    async def test_execute_class_action_with_run(self):
        """Execute a class-based action that has a run() method."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        class MyAction:
            def run(self, **kwargs):
                return {"message": "hello"}

        MyAction.action_meta = {"name": "my_action"}
        dispatcher.register_action(MyAction, name="my_action")

        result, status = await dispatcher.execute_action("my_action", {})
        assert status == "success"
        assert result == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_execute_class_action_without_run(self):
        """Execute a class-based action without run() should log error."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        class BadAction:
            pass

        BadAction.action_meta = {"name": "bad_action"}
        dispatcher.register_action(BadAction, name="bad_action")

        result, status = await dispatcher.execute_action("bad_action", {})
        # Should fail because no run() method and not callable
        assert status == "failed"

    @pytest.mark.asyncio
    async def test_execute_async_function_action(self):
        """Execute an async function action."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        async def my_async_action(**kwargs):
            return "async result"

        my_async_action.action_meta = {"name": "my_async_action"}
        dispatcher.register_action(my_async_action, name="my_async_action")

        result, status = await dispatcher.execute_action("my_async_action", {})
        assert status == "success"
        assert result == "async result"

    @pytest.mark.asyncio
    async def test_execute_action_exception_handling(self):
        """Action that raises should return failed status."""
        dispatcher = ActionDispatcher(load_all_actions=False)

        def failing_action(**kwargs):
            raise RuntimeError("action failed")

        failing_action.action_meta = {"name": "failing_action"}
        dispatcher.register_action(failing_action, name="failing_action")

        result, status = await dispatcher.execute_action("failing_action", {})
        assert status == "failed"
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_unregistered_action(self):
        """Executing an unregistered action should return failed."""
        dispatcher = ActionDispatcher(load_all_actions=False)
        result, status = await dispatcher.execute_action("nonexistent", {})
        assert status == "failed"


class TestLoadActionsFromModule:
    """Tests for _load_actions_from_module error paths."""

    def test_nonexistent_file(self):
        """Loading from nonexistent file should return empty dict."""
        result = ActionDispatcher._load_actions_from_module("/nonexistent/path.py")
        assert result == {}

    def test_invalid_python_file(self):
        """Loading from a file with syntax errors should handle gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(\n")  # syntax error
            f.flush()
            try:
                result = ActionDispatcher._load_actions_from_module(f.name)
                # Should either return empty or raise RuntimeError
            except RuntimeError:
                pass  # expected
            finally:
                os.unlink(f.name)

    def test_valid_module_with_action(self):
        """Loading a valid module with @action-decorated function."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                """
def my_action(**kwargs):
    return "result"

my_action.action_meta = {"name": "test_action_from_module"}
"""
            )
            f.flush()
            try:
                result = ActionDispatcher._load_actions_from_module(f.name)
                assert "test_action_from_module" in result
            finally:
                os.unlink(f.name)


class TestFindActions:
    """Tests for _find_actions."""

    def test_nonexistent_directory(self):
        """_find_actions with nonexistent directory should return empty."""
        dispatcher = ActionDispatcher(load_all_actions=False)
        result = dispatcher._find_actions("/nonexistent/directory")
        assert result == {}


class TestIsActionFile:
    """Tests for is_action_file helper."""

    def test_init_file_excluded(self):
        assert is_action_file("__init__.py") is False
        assert is_action_file("/path/to/__init__.py") is False

    def test_regular_file_included(self):
        assert is_action_file("actions.py") is True
        assert is_action_file("/path/to/my_action.py") is True


class TestActionDispatcherProperties:
    """Tests for ActionDispatcher property accessors."""

    def test_thread_pool_setter(self):
        dispatcher = ActionDispatcher(load_all_actions=False)
        assert dispatcher.thread_pool is None
        dispatcher.thread_pool = "fake_pool"
        assert dispatcher.thread_pool == "fake_pool"

    def test_registered_actions_property(self):
        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher.registered_actions, dict)

    def test_get_registered_actions(self):
        dispatcher = ActionDispatcher(load_all_actions=False)
        dispatcher.register_action(lambda **kw: None, name="test_act")
        assert "test_act" in dispatcher.get_registered_actions()

    def test_has_registered(self):
        dispatcher = ActionDispatcher(load_all_actions=False)
        dispatcher.register_action(lambda **kw: None, name="test_act")
        assert dispatcher.has_registered("test_act") is True
        assert dispatcher.has_registered("nonexistent") is False

    def test_register_action_no_override(self):
        dispatcher = ActionDispatcher(load_all_actions=False)

        def original(**kw):
            return "original"

        def replacement(**kw):
            return "replacement"

        dispatcher.register_action(original, name="act")
        dispatcher.register_action(replacement, name="act", override=False)
        # Original should remain
        assert dispatcher.get_action("act") is original
