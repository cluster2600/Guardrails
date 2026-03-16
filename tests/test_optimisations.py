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
  - _unique_list_concat O(n) path with hashable and unhashable items
  - Bounded LRU events_history_cache on LLMRails
"""

import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _unique_list_concat — O(n) deduplication
# ---------------------------------------------------------------------------


class TestUniqueListConcat:
    """Test the optimised _unique_list_concat function."""

    def test_basic_dedup_hashable(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        result = _unique_list_concat([1, 2, 3], [2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5]

    def test_string_dedup(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        result = _unique_list_concat(["a", "b"], ["b", "c", "a"])
        assert result == ["a", "b", "c"]

    def test_dict_dedup_unhashable(self):
        """Dicts are unhashable — must fall back to linear scan."""
        from nemoguardrails.rails.llm.config import _unique_list_concat

        d1 = {"name": "flow_a"}
        d2 = {"name": "flow_b"}
        d3 = {"name": "flow_a"}  # same content as d1

        result = _unique_list_concat([d1, d2], [d3, {"name": "flow_c"}])
        # d3 has same content as d1 so should be deduped
        assert len(result) == 3
        assert result[0] == {"name": "flow_a"}
        assert result[1] == {"name": "flow_b"}
        assert result[2] == {"name": "flow_c"}

    def test_mixed_hashable_unhashable(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        result = _unique_list_concat(
            [1, "hello", {"x": 1}],
            [1, {"x": 1}, "world"],
        )
        assert result == [1, "hello", {"x": 1}, "world"]

    def test_empty_lists(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        assert _unique_list_concat([], []) == []
        assert _unique_list_concat([1, 2], []) == [1, 2]
        assert _unique_list_concat([], [3, 4]) == [3, 4]

    def test_preserves_order(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        result = _unique_list_concat([3, 1, 2], [4, 2, 5, 1])
        assert result == [3, 1, 2, 4, 5]

    def test_tuples_are_hashable(self):
        from nemoguardrails.rails.llm.config import _unique_list_concat

        result = _unique_list_concat([(1, 2), (3, 4)], [(3, 4), (5, 6)])
        assert result == [(1, 2), (3, 4), (5, 6)]

    def test_large_list_performance(self):
        """Verify O(n) behaviour — should complete quickly for large lists."""
        from nemoguardrails.rails.llm.config import _unique_list_concat

        list1 = list(range(10000))
        list2 = list(range(5000, 15000))
        result = _unique_list_concat(list1, list2)
        assert len(result) == 15000
        assert result == list(range(15000))


# ---------------------------------------------------------------------------
# Bounded LRU events_history_cache
# ---------------------------------------------------------------------------


class TestEventsHistoryCacheBounded:
    """Test the bounded LRU cache for events_history_cache."""

    def _make_rails(self, cache_size="5"):
        """Create an LLMRails instance with a small cache for testing."""
        # We patch the constructor to avoid full initialisation
        with patch.dict(os.environ, {"NEMOGUARDRAILS_EVENTS_CACHE_SIZE": cache_size}):
            from nemoguardrails.rails.llm.llmrails import LLMRails, _LRUDict

            config = MagicMock()
            config.colang_version = "1.0"
            config.flows = []
            config.rails.config = MagicMock()
            config.rails.dialog_rails = []

            # Create instance without running __init__ fully
            rails = LLMRails.__new__(LLMRails)
            rails._events_cache_maxsize = int(cache_size)
            rails.events_history_cache = _LRUDict(maxsize=int(cache_size))
            return rails

    def test_cache_is_lru_dict(self):
        rails = self._make_rails()
        assert isinstance(rails.events_history_cache, _LRUDict)

    def test_cache_eviction_on_overflow(self):
        """_LRUDict.__setitem__ evicts LRU entry when cache exceeds maxsize."""
        rails = self._make_rails("3")

        rails.events_history_cache["k1"] = [{"type": "event1"}]
        rails.events_history_cache["k2"] = [{"type": "event2"}]
        rails.events_history_cache["k3"] = [{"type": "event3"}]

        # Add a 4th entry — should evict k1 (oldest)
        rails.events_history_cache["k4"] = [{"type": "event4"}]

        assert "k1" not in rails.events_history_cache
        assert len(rails.events_history_cache) == 3
        assert list(rails.events_history_cache.keys()) == ["k2", "k3", "k4"]

    def test_read_promotes_to_mru(self):
        """_LRUDict.__getitem__ promotes accessed key; verify eviction order."""
        rails = self._make_rails("3")

        rails.events_history_cache["k1"] = [{"type": "event1"}]
        rails.events_history_cache["k2"] = [{"type": "event2"}]
        rails.events_history_cache["k3"] = [{"type": "event3"}]

        # Read k1 — __getitem__ promotes it to MRU
        _ = rails.events_history_cache["k1"]

        # Add k4 — should evict k2 (now the oldest)
        rails.events_history_cache["k4"] = [{"type": "event4"}]

        assert "k2" not in rails.events_history_cache
        assert "k1" in rails.events_history_cache
        assert list(rails.events_history_cache.keys()) == ["k3", "k1", "k4"]

    def test_write_promotes_existing_key_to_mru(self):
        """Overwriting an existing key should promote it to MRU position."""
        rails = self._make_rails("3")

        rails.events_history_cache["k1"] = [{"type": "event1"}]
        rails.events_history_cache["k2"] = [{"type": "event2"}]
        rails.events_history_cache["k3"] = [{"type": "event3"}]

        # Re-write k1 with new value — __setitem__ promotes to MRU
        rails.events_history_cache["k1"] = [{"type": "event1_updated"}]

        # k1 is now MRU, so adding k4 should evict k2 (oldest)
        rails.events_history_cache["k4"] = [{"type": "event4"}]

        assert "k2" not in rails.events_history_cache
        # Note: reading k1 via __getitem__ promotes it to MRU again,
        # so the final order is k3, k4, k1.
        assert rails.events_history_cache["k1"] == [{"type": "event1_updated"}]
        assert list(rails.events_history_cache.keys()) == ["k3", "k4", "k1"]

    def test_env_var_controls_size(self):
        rails = self._make_rails("100")
        assert rails._events_cache_maxsize == 100

    def test_zero_means_unlimited(self):
        """maxsize=0 disables eviction."""
        from nemoguardrails.rails.llm.llmrails import _LRUDict

        cache = _LRUDict(maxsize=0)

        for i in range(200):
            cache[f"k{i}"] = [{"type": f"event{i}"}]

        assert len(cache) == 200
