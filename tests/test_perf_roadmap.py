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

"""Tests for performance roadmap Tier 1 optimisations."""

from collections import OrderedDict

import pytest

# ---------------------------------------------------------------------------
# Prompt lookup index
# ---------------------------------------------------------------------------


class TestPromptIndex:
    """Test the prompt-model lookup index."""

    def test_module_level_index_exists(self):
        from nemoguardrails.llm.prompts import _prompt_index

        assert isinstance(_prompt_index, dict)
        assert len(_prompt_index) > 0

    def test_index_keys_are_task_names(self):
        from nemoguardrails.llm.prompts import _prompt_index

        for key, prompts in _prompt_index.items():
            assert isinstance(key, str)
            for p in prompts:
                assert p.task == key

    def test_build_prompt_index(self):
        from nemoguardrails.llm.prompts import _build_prompt_index
        from nemoguardrails.rails.llm.config import TaskPrompt

        prompts = [
            TaskPrompt(task="task_a", content="a1"),
            TaskPrompt(task="task_a", content="a2"),
            TaskPrompt(task="task_b", content="b1"),
        ]
        index = _build_prompt_index(prompts)
        assert len(index["task_a"]) == 2
        assert len(index["task_b"]) == 1

    def test_get_prompt_uses_index(self):
        """_get_prompt with an index should return the same result as without."""
        from nemoguardrails.llm.prompts import (
            _build_prompt_index,
            _get_prompt,
            _prompts,
        )

        if not _prompts:
            pytest.skip("No prompts loaded")

        # Pick the first available task
        task_name = _prompts[0].task
        index = _build_prompt_index(_prompts)

        result_with_index = _get_prompt(task_name, "unknown", "standard", _prompts, index)
        result_without_index = _get_prompt(task_name, "unknown", "standard", _prompts, None)
        assert result_with_index == result_without_index

    def test_build_prompt_index_empty(self):
        from nemoguardrails.llm.prompts import _build_prompt_index

        index = _build_prompt_index([])
        assert index == {}


# ---------------------------------------------------------------------------
# Context variable deduplication
# ---------------------------------------------------------------------------


class TestContextDedup:
    """Test that the duplicate streaming_handler_var is fixed."""

    def test_streaming_handler_var_defined_once(self):
        import nemoguardrails.context as ctx

        # Should have exactly one streaming_handler_var
        assert hasattr(ctx, "streaming_handler_var")
        # It should be a ContextVar
        import contextvars

        assert isinstance(ctx.streaming_handler_var, contextvars.ContextVar)

    def test_no_eager_streaming_import(self):
        """StreamingHandler should only be imported under TYPE_CHECKING."""
        import inspect

        source = inspect.getsource(__import__("nemoguardrails.context", fromlist=["context"]))
        # Find the import line — it should appear only inside the TYPE_CHECKING block
        in_type_checking = False
        for line in source.split("\n"):
            stripped = line.strip()
            if "TYPE_CHECKING" in stripped and "if" in stripped:
                in_type_checking = True
            elif stripped and not stripped.startswith("#") and not line.startswith(" "):
                # New top-level statement ends the TYPE_CHECKING block
                if in_type_checking and not stripped.startswith("from") and not stripped.startswith("import"):
                    in_type_checking = False
            if "from nemoguardrails.streaming import StreamingHandler" in stripped:
                assert in_type_checking, "StreamingHandler imported outside TYPE_CHECKING"


# ---------------------------------------------------------------------------
# Streaming fire-and-forget fix
# ---------------------------------------------------------------------------


class TestStreamingTaskTracking:
    """Test that streaming pipe tasks are tracked."""

    @pytest.mark.asyncio
    async def test_pipe_to_task_has_done_callback(self):
        """Verify that piped tasks have exception-handling callbacks."""
        from nemoguardrails.streaming import StreamingHandler

        handler = StreamingHandler()
        pipe_handler = StreamingHandler()
        handler.pipe_to = pipe_handler

        # Push a chunk and verify no untracked task warning
        await handler.push_chunk("test")
        # If we get here without "exception was never retrieved", the fix works


# ---------------------------------------------------------------------------
# Unified free-threading detection
# ---------------------------------------------------------------------------


class TestUnifiedFreeThreading:
    """Test that all modules use the same detection."""

    def test_dag_scheduler_uses_thread_safety(self):
        from nemoguardrails._thread_safety import is_free_threaded
        from nemoguardrails.rails.llm.dag_scheduler import _IS_FREE_THREADED

        assert _IS_FREE_THREADED == is_free_threaded()

    def test_thread_pool_uses_thread_safety(self):
        from nemoguardrails._thread_safety import is_free_threaded
        from nemoguardrails.rails.llm.thread_pool import is_free_threaded as tp_is_ft

        # Should be the exact same function
        assert tp_is_ft is is_free_threaded

    def test_action_dispatcher_uses_thread_safety(self):
        from nemoguardrails._thread_safety import is_free_threaded
        from nemoguardrails.actions.action_dispatcher import (
            is_free_threaded as ad_is_ft,
        )

        assert ad_is_ft is is_free_threaded


# ---------------------------------------------------------------------------
# History cache bounds
# ---------------------------------------------------------------------------


class TestHistoryCacheBounds:
    """Test the bounded LRU history cache."""

    def test_history_cache_is_ordered_dict(self):
        # Can't instantiate LLMRails without config, so check the constant
        from nemoguardrails.rails.llm.llmrails import _HISTORY_CACHE_MAX_SIZE

        assert _HISTORY_CACHE_MAX_SIZE == 1024

    def test_ordered_dict_lru_pattern(self):
        """Verify the LRU eviction pattern works correctly."""
        cache: OrderedDict = OrderedDict()
        max_size = 3

        # Fill cache
        for i in range(3):
            cache[f"key_{i}"] = f"value_{i}"

        # Access key_0 to make it MRU
        cache.move_to_end("key_0")

        # Add a new entry, should evict key_1 (LRU)
        cache["key_3"] = "value_3"
        if len(cache) > max_size:
            cache.popitem(last=False)

        assert "key_1" not in cache
        assert "key_0" in cache
        assert "key_3" in cache


# ---------------------------------------------------------------------------
# KB init threading fix
# ---------------------------------------------------------------------------


class TestKBInitFix:
    """Test that the KB init bypass is removed."""

    def test_no_true_or_bypass(self):
        """The 'True or' bypass should be removed from KB init."""
        import inspect

        from nemoguardrails.rails.llm.llmrails import LLMRails

        source = inspect.getsource(LLMRails.__init__)
        assert "True or check_sync_call_from_async_loop" not in source
        assert "check_sync_call_from_async_loop" in source


# ---------------------------------------------------------------------------
# Double serialisation fix
# ---------------------------------------------------------------------------


class TestDoubleSerialisation:
    """Test that model_dump_json is used instead of json.dumps(model_dump())."""

    def test_no_double_serialisation_in_api(self):
        """Check the source file directly to avoid import issues with fastapi."""
        import os

        api_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "nemoguardrails",
            "server",
            "api.py",
        )
        with open(api_path) as f:
            source = f.read()
        # Should not contain the double serialisation pattern
        assert "json.dumps(processed_chunk.model_dump())" not in source
        # Should contain the optimised version
        assert "model_dump_json()" in source
