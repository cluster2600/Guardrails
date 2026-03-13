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

"""Tests for performance optimisations.

Covers:
  - _LRUDict bounded eviction
  - Per-instance process_events semaphore
  - Action name normalisation cache
  - Jinja2 template and variable caching in LLMTaskManager
"""

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
        assert v1 == v2
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
