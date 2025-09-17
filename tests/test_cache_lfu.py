# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Comprehensive test suite for LFU Cache implementation.

Tests all functionality including basic operations, eviction policies,
capacity management, edge cases, and persistence functionality.
"""

import json
import os
import tempfile
import time
import unittest
from typing import Any

from nemoguardrails.cache.lfu import LFUCache


class TestLFUCache(unittest.TestCase):
    """Test cases for LFU Cache implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = LFUCache(3)

    def test_initialization(self):
        """Test cache initialization with various capacities."""
        # Normal capacity
        cache = LFUCache(5)
        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

        # Zero capacity
        cache_zero = LFUCache(0)
        self.assertEqual(cache_zero.size(), 0)

        # Negative capacity should raise error
        with self.assertRaises(ValueError):
            LFUCache(-1)

    def test_basic_put_get(self):
        """Test basic put and get operations."""
        # Put and get single item
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.size(), 1)

        # Put and get multiple items
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")

        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.size(), 3)

    def test_get_nonexistent_key(self):
        """Test getting non-existent keys."""
        # Default behavior (returns None)
        self.assertIsNone(self.cache.get("nonexistent"))

        # With custom default
        self.assertEqual(self.cache.get("nonexistent", "default"), "default")

        # After adding some items
        self.cache.put("key1", "value1")
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key2", 42), 42)

    def test_update_existing_key(self):
        """Test updating values for existing keys."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")

        # Update existing key
        self.cache.put("key1", "new_value1")
        self.assertEqual(self.cache.get("key1"), "new_value1")

        # Size should not change
        self.assertEqual(self.cache.size(), 2)

    def test_lfu_eviction_basic(self):
        """Test basic LFU eviction when cache is full."""
        # Fill cache
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Access 'a' and 'b' to increase their frequency
        self.cache.get("a")  # freq: 2
        self.cache.get("b")  # freq: 2
        # 'c' remains at freq: 1

        # Add new item - should evict 'c' (lowest frequency)
        self.cache.put("d", 4)

        self.assertEqual(self.cache.get("a"), 1)
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.get("d"), 4)
        self.assertIsNone(self.cache.get("c"))  # Should be evicted

    def test_lfu_with_lru_tiebreaker(self):
        """Test LRU eviction among items with same frequency."""
        # Fill cache - all items have frequency 1
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Add new item - should evict 'a' (least recently used among freq 1)
        self.cache.put("d", 4)

        self.assertIsNone(self.cache.get("a"))  # Should be evicted
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.get("c"), 3)
        self.assertEqual(self.cache.get("d"), 4)

    def test_frequency_increment(self):
        """Test that frequencies are properly incremented."""
        self.cache.put("a", 1)

        # Access 'a' multiple times
        for _ in range(5):
            self.assertEqual(self.cache.get("a"), 1)

        # Fill the rest of cache
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Add new items - 'a' should not be evicted due to high frequency
        self.cache.put("d", 4)  # Should evict 'b' or 'c'
        self.assertEqual(self.cache.get("a"), 1)  # 'a' should still be there

    def test_complex_eviction_scenario(self):
        """Test complex eviction scenario with multiple frequency levels."""
        # Create a new cache for this test
        cache = LFUCache(4)

        # Add items and create different frequency levels
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)

        # Create frequency pattern:
        # a: freq 3 (accessed 2 more times)
        # b: freq 2 (accessed 1 more time)
        # c: freq 2 (accessed 1 more time)
        # d: freq 1 (not accessed)

        cache.get("a")
        cache.get("a")
        cache.get("b")
        cache.get("c")

        # Add new item - should evict 'd' (freq 1)
        cache.put("e", 5)
        self.assertIsNone(cache.get("d"))

        # Add another item - should evict one of the least frequently used
        cache.put("f", 6)

        # After eviction, we should have:
        # - 'a' (freq 3) - definitely kept
        # - 'b' (freq 2) and 'c' (freq 2) - higher frequency, both kept
        # - 'f' (freq 1) - just added
        # - 'e' (freq 1) was evicted as it was least recently used among freq 1 items

        # Check that we're at capacity
        self.assertEqual(cache.size(), 4)

        # 'a' should definitely still be there (highest frequency)
        self.assertEqual(cache.get("a"), 1)

        # 'b' and 'c' should both be there (freq 2)
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

        # 'f' should be there (just added)
        self.assertEqual(cache.get("f"), 6)

        # 'e' should have been evicted (freq 1, LRU among freq 1 items)
        self.assertIsNone(cache.get("e"))

    def test_zero_capacity_cache(self):
        """Test cache with zero capacity."""
        cache = LFUCache(0)

        # Put should not store anything
        cache.put("key", "value")
        self.assertEqual(cache.size(), 0)
        self.assertIsNone(cache.get("key"))

        # Multiple puts
        for i in range(10):
            cache.put(f"key{i}", f"value{i}")

        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

    def test_clear_method(self):
        """Test clearing the cache."""
        # Add items
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)

        # Verify items exist
        self.assertEqual(self.cache.size(), 3)
        self.assertFalse(self.cache.is_empty())

        # Clear cache
        self.cache.clear()

        # Verify cache is empty
        self.assertEqual(self.cache.size(), 0)
        self.assertTrue(self.cache.is_empty())

        # Verify items are gone
        self.assertIsNone(self.cache.get("a"))
        self.assertIsNone(self.cache.get("b"))
        self.assertIsNone(self.cache.get("c"))

        # Can still use cache after clear
        self.cache.put("new_key", "new_value")
        self.assertEqual(self.cache.get("new_key"), "new_value")

    def test_various_data_types(self):
        """Test cache with various data types as keys and values."""
        # Integer keys
        self.cache.put(1, "one")
        self.cache.put(2, "two")
        self.assertEqual(self.cache.get(1), "one")
        self.assertEqual(self.cache.get(2), "two")

        # Tuple keys
        self.cache.put((1, 2), "tuple_value")
        self.assertEqual(self.cache.get((1, 2)), "tuple_value")

        # Clear for more tests
        self.cache.clear()

        # Complex values
        self.cache.put("list", [1, 2, 3])
        self.cache.put("dict", {"a": 1, "b": 2})
        self.cache.put("set", {1, 2, 3})

        self.assertEqual(self.cache.get("list"), [1, 2, 3])
        self.assertEqual(self.cache.get("dict"), {"a": 1, "b": 2})
        self.assertEqual(self.cache.get("set"), {1, 2, 3})

    def test_none_values(self):
        """Test storing None as a value."""
        self.cache.put("key", None)
        # get should return None for the value, not the default
        self.assertIsNone(self.cache.get("key"))
        self.assertEqual(self.cache.get("key", "default"), None)

        # Verify key exists
        self.assertEqual(self.cache.size(), 1)

    def test_size_and_capacity(self):
        """Test size tracking and capacity limits."""
        # Start empty
        self.assertEqual(self.cache.size(), 0)

        # Add items up to capacity
        for i in range(3):
            self.cache.put(f"key{i}", f"value{i}")
            self.assertEqual(self.cache.size(), i + 1)

        # Add more items - size should stay at capacity
        for i in range(3, 10):
            self.cache.put(f"key{i}", f"value{i}")
            self.assertEqual(self.cache.size(), 3)

    def test_eviction_updates_size(self):
        """Test that eviction properly updates cache size."""
        # Fill cache
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)
        self.assertEqual(self.cache.size(), 3)

        # Cause eviction
        self.cache.put("d", 4)
        self.assertEqual(self.cache.size(), 3)  # Size should remain at capacity

    def test_is_empty(self):
        """Test is_empty method in various states."""
        # Initially empty
        self.assertTrue(self.cache.is_empty())

        # After adding item
        self.cache.put("key", "value")
        self.assertFalse(self.cache.is_empty())

        # After clearing
        self.cache.clear()
        self.assertTrue(self.cache.is_empty())

    def test_repeated_puts_same_key(self):
        """Test repeated puts with the same key don't increase size."""
        self.cache.put("key", "value1")
        self.assertEqual(self.cache.size(), 1)

        # Update same key multiple times
        for i in range(10):
            self.cache.put("key", f"value{i}")
            self.assertEqual(self.cache.size(), 1)

        # Final value should be the last one
        self.assertEqual(self.cache.get("key"), "value9")

    def test_access_pattern_preserves_frequently_used(self):
        """Test that frequently accessed items are preserved during evictions."""
        # Create specific access pattern
        cache = LFUCache(3)

        # Add three items
        cache.put("rarely_used", 1)
        cache.put("sometimes_used", 2)
        cache.put("frequently_used", 3)

        # Create access pattern
        # frequently_used: access 10 times
        for _ in range(10):
            cache.get("frequently_used")

        # sometimes_used: access 3 times
        for _ in range(3):
            cache.get("sometimes_used")

        # rarely_used: no additional access (freq = 1)

        # Add new items to trigger evictions
        cache.put("new1", 4)  # Should evict rarely_used
        cache.put("new2", 5)  # Should evict new1 (freq = 1)

        # frequently_used and sometimes_used should still be there
        self.assertEqual(cache.get("frequently_used"), 3)
        self.assertEqual(cache.get("sometimes_used"), 2)

        # rarely_used and new1 should be evicted
        self.assertIsNone(cache.get("rarely_used"))
        self.assertIsNone(cache.get("new1"))

        # new2 should be there
        self.assertEqual(cache.get("new2"), 5)


class TestLFUCacheInterface(unittest.TestCase):
    """Test that LFUCache properly implements CacheInterface."""

    def test_interface_methods_exist(self):
        """Verify all interface methods are implemented."""
        cache = LFUCache(5)

        # Check all required methods exist and are callable
        self.assertTrue(callable(getattr(cache, "get", None)))
        self.assertTrue(callable(getattr(cache, "put", None)))
        self.assertTrue(callable(getattr(cache, "size", None)))
        self.assertTrue(callable(getattr(cache, "is_empty", None)))
        self.assertTrue(callable(getattr(cache, "clear", None)))

        # Check property
        self.assertEqual(cache.capacity, 5)

    def test_persistence_interface_methods(self):
        """Verify persistence interface methods are implemented."""
        # Cache without persistence
        cache_no_persist = LFUCache(5)
        self.assertTrue(callable(getattr(cache_no_persist, "persist_now", None)))
        self.assertTrue(
            callable(getattr(cache_no_persist, "supports_persistence", None))
        )
        self.assertFalse(cache_no_persist.supports_persistence())

        # Cache with persistence
        temp_file = os.path.join(tempfile.mkdtemp(), "test_interface.json")
        try:
            cache_with_persist = LFUCache(
                5, persistence_interval=10.0, persistence_path=temp_file
            )
            self.assertTrue(cache_with_persist.supports_persistence())

            # persist_now should work without errors
            cache_with_persist.put("key", "value")
            cache_with_persist.persist_now()  # Should not raise any exception
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(os.path.dirname(temp_file)):
                os.rmdir(os.path.dirname(temp_file))


class TestLFUCachePersistence(unittest.TestCase):
    """Test cases for LFU Cache persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_cache.json")

    def tearDown(self):
        """Clean up test files."""
        # Clean up any created files
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_basic_persistence(self):
        """Test basic save and load functionality."""
        # Create cache and add items
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        cache.put("key1", "value1")
        cache.put("key2", {"nested": "value"})
        cache.put("key3", [1, 2, 3])

        # Force persistence
        cache.persist_now()

        # Verify file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Load into new cache
        new_cache = LFUCache(
            5, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Verify data was loaded correctly
        self.assertEqual(new_cache.size(), 3)
        self.assertEqual(new_cache.get("key1"), "value1")
        self.assertEqual(new_cache.get("key2"), {"nested": "value"})
        self.assertEqual(new_cache.get("key3"), [1, 2, 3])

    def test_frequency_preservation(self):
        """Test that frequencies are preserved across persistence."""
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Create different frequency levels
        cache.put("freq1", "value1")
        cache.put("freq3", "value3")
        cache.put("freq5", "value5")

        # Access items to create different frequencies
        cache.get("freq3")  # freq = 2
        cache.get("freq3")  # freq = 3

        cache.get("freq5")  # freq = 2
        cache.get("freq5")  # freq = 3
        cache.get("freq5")  # freq = 4
        cache.get("freq5")  # freq = 5

        # Force persistence
        cache.persist_now()

        # Load into new cache
        new_cache = LFUCache(
            5, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Add new items to test eviction order
        new_cache.put("new1", "newvalue1")
        new_cache.put("new2", "newvalue2")
        new_cache.put("new3", "newvalue3")

        # freq1 should be evicted first (lowest frequency)
        self.assertIsNone(new_cache.get("freq1"))
        # freq3 and freq5 should still be there
        self.assertEqual(new_cache.get("freq3"), "value3")
        self.assertEqual(new_cache.get("freq5"), "value5")

    def test_periodic_persistence(self):
        """Test automatic periodic persistence."""
        # Use short interval for testing
        cache = LFUCache(5, persistence_interval=0.5, persistence_path=self.test_file)

        cache.put("key1", "value1")

        # File shouldn't exist yet
        self.assertFalse(os.path.exists(self.test_file))

        # Wait for interval to pass
        time.sleep(0.6)

        # Access cache to trigger persistence check
        cache.get("key1")

        # File should now exist
        self.assertTrue(os.path.exists(self.test_file))

        # Verify content
        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(data["capacity"], 5)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["key"], "key1")

    def test_persistence_with_empty_cache(self):
        """Test persistence behavior with empty cache."""
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Add and remove items
        cache.put("key1", "value1")
        cache.clear()

        # Force persistence
        cache.persist_now()

        # File should be removed when cache is empty
        self.assertFalse(os.path.exists(self.test_file))

    def test_no_persistence_when_disabled(self):
        """Test that persistence doesn't occur when not configured."""
        # Create cache without persistence
        cache = LFUCache(5)

        cache.put("key1", "value1")
        cache.persist_now()  # Should do nothing

        # No file should be created
        self.assertFalse(os.path.exists("lfu_cache.json"))

    def test_load_from_nonexistent_file(self):
        """Test loading when persistence file doesn't exist."""
        # Create cache with non-existent file
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)

        # Should start empty
        self.assertEqual(cache.size(), 0)
        self.assertTrue(cache.is_empty())

    def test_persistence_with_complex_data(self):
        """Test persistence with various data types."""
        cache = LFUCache(10, persistence_interval=10.0, persistence_path=self.test_file)

        # Add various data types
        test_data = {
            "string": "hello world",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, [3, 4]],
            "dict": {"a": 1, "b": {"c": 2}},
            "tuple_key": "value_for_tuple",  # Will use string key since tuples aren't JSON serializable
        }

        for key, value in test_data.items():
            cache.put(key, value)

        # Force persistence
        cache.persist_now()

        # Load into new cache
        new_cache = LFUCache(
            10, persistence_interval=10.0, persistence_path=self.test_file
        )

        # Verify all data types
        for key, value in test_data.items():
            self.assertEqual(new_cache.get(key), value)

    def test_persistence_file_corruption_handling(self):
        """Test handling of corrupted persistence files."""
        # Create invalid JSON file
        with open(self.test_file, "w") as f:
            f.write("{ invalid json content")

        # Should handle gracefully and start with empty cache
        cache = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache.size(), 0)

        # Cache should still be functional
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_multiple_persistence_cycles(self):
        """Test multiple save/load cycles."""
        # First cycle
        cache1 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        cache1.put("key1", "value1")
        cache1.put("key2", "value2")
        cache1.persist_now()

        # Second cycle - load and modify
        cache2 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache2.size(), 2)
        cache2.put("key3", "value3")
        cache2.persist_now()

        # Third cycle - verify all changes
        cache3 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        self.assertEqual(cache3.size(), 3)
        self.assertEqual(cache3.get("key1"), "value1")
        self.assertEqual(cache3.get("key2"), "value2")
        self.assertEqual(cache3.get("key3"), "value3")

    def test_capacity_change_on_load(self):
        """Test loading cache data into cache with different capacity."""
        # Create cache with capacity 5
        cache1 = LFUCache(5, persistence_interval=10.0, persistence_path=self.test_file)
        for i in range(5):
            cache1.put(f"key{i}", f"value{i}")
        cache1.persist_now()

        # Load into cache with smaller capacity
        cache2 = LFUCache(3, persistence_interval=10.0, persistence_path=self.test_file)

        # Current design: loads all persisted items regardless of new capacity
        # This is a valid design choice - preserve data integrity on load
        self.assertEqual(cache2.size(), 5)

        # The cache continues to operate with loaded items
        # New items can still be added, and the cache will manage its size
        cache2.put("new_key", "new_value")

        # Verify the cache is still functional and contains the new item
        self.assertEqual(cache2.get("new_key"), "new_value")
        self.assertGreaterEqual(
            cache2.size(), 4
        )  # At least has the new item plus some old ones

    def test_persistence_timing(self):
        """Test that persistence doesn't happen too frequently."""
        cache = LFUCache(5, persistence_interval=1.0, persistence_path=self.test_file)

        cache.put("key1", "value1")

        # Multiple operations within interval shouldn't trigger persistence
        for i in range(10):
            cache.get("key1")
            self.assertFalse(os.path.exists(self.test_file))
            time.sleep(0.05)  # Total time still less than interval

        # Wait for interval to pass
        time.sleep(0.6)
        cache.get("key1")

        # Now file should exist
        self.assertTrue(os.path.exists(self.test_file))

    def test_persistence_with_statistics(self):
        """Test persistence doesn't interfere with statistics tracking."""
        cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=0.5,
            persistence_path=self.test_file,
        )

        # Perform operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")
        cache.get("nonexistent")

        # Wait for persistence
        time.sleep(0.6)
        cache.get("key1")  # Trigger persistence

        # Check stats are still correct
        stats = cache.get_stats()
        self.assertEqual(stats["puts"], 2)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)

        # Load into new cache with stats
        new_cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=0.5,
            persistence_path=self.test_file,
        )

        # Stats should be reset in new instance
        new_stats = new_cache.get_stats()
        self.assertEqual(new_stats["puts"], 0)
        self.assertEqual(new_stats["hits"], 0)

        # But data should be loaded
        self.assertEqual(new_cache.size(), 2)


class TestLFUCacheStatsLogging(unittest.TestCase):
    """Test cases for LFU Cache statistics logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_file = tempfile.mktemp()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_stats_logging_disabled_by_default(self):
        """Test that stats logging is disabled when not configured."""
        cache = LFUCache(5, track_stats=True)
        self.assertFalse(cache.supports_stats_logging())

    def test_stats_logging_requires_tracking(self):
        """Test that stats logging requires stats tracking to be enabled."""
        # Logging without tracking
        cache = LFUCache(5, track_stats=False, stats_logging_interval=1.0)
        self.assertFalse(cache.supports_stats_logging())

        # Both enabled
        cache = LFUCache(5, track_stats=True, stats_logging_interval=1.0)
        self.assertTrue(cache.supports_stats_logging())

    def test_log_stats_now(self):
        """Test immediate stats logging."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=60.0)

        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")
        cache.get("nonexistent")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            # Verify log was called
            self.assertEqual(mock_log.call_count, 1)
            log_message = mock_log.call_args[0][0]

            # Check log format
            self.assertIn("LFU Cache Statistics", log_message)
            self.assertIn("Size: 2/5", log_message)
            self.assertIn("Hits: 1", log_message)
            self.assertIn("Misses: 1", log_message)
            self.assertIn("Hit Rate: 50.00%", log_message)
            self.assertIn("Evictions: 0", log_message)
            self.assertIn("Puts: 2", log_message)
            self.assertIn("Updates: 0", log_message)

    def test_periodic_stats_logging(self):
        """Test automatic periodic stats logging."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.5)

        # Add some data
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            # Initial operations shouldn't trigger logging
            cache.get("key1")
            self.assertEqual(mock_log.call_count, 0)

            # Wait for interval to pass
            time.sleep(0.6)

            # Next operation should trigger logging
            cache.get("key1")
            self.assertEqual(mock_log.call_count, 1)

            # Another operation without waiting shouldn't trigger
            cache.get("key2")
            self.assertEqual(mock_log.call_count, 1)

            # Wait again
            time.sleep(0.6)
            cache.put("key3", "value3")
            self.assertEqual(mock_log.call_count, 2)

    def test_stats_logging_with_empty_cache(self):
        """Test stats logging with empty cache."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        # Generate a miss first
        cache.get("nonexistent")

        # Wait for interval to pass
        time.sleep(0.2)

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            # This will trigger stats logging with the previous miss already counted
            cache.get("another_nonexistent")  # Trigger check

            self.assertEqual(mock_log.call_count, 1)
            log_message = mock_log.call_args[0][0]

            self.assertIn("Size: 0/5", log_message)
            self.assertIn("Hits: 0", log_message)
            self.assertIn("Misses: 1", log_message)  # The first miss is logged
            self.assertIn("Hit Rate: 0.00%", log_message)

    def test_stats_logging_with_full_cache(self):
        """Test stats logging when cache is at capacity."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(3, track_stats=True, stats_logging_interval=0.1)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Cause eviction
        cache.put("key4", "value4")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            time.sleep(0.2)
            cache.get("key4")  # Trigger check

            log_message = mock_log.call_args[0][0]
            self.assertIn("Size: 3/3", log_message)
            self.assertIn("Evictions: 1", log_message)
            self.assertIn("Puts: 4", log_message)

    def test_stats_logging_high_hit_rate(self):
        """Test stats logging with high hit rate."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        cache.put("key1", "value1")

        # Many hits
        for _ in range(99):
            cache.get("key1")

        # One miss
        cache.get("nonexistent")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            log_message = mock_log.call_args[0][0]
            self.assertIn("Hit Rate: 99.00%", log_message)
            self.assertIn("Hits: 99", log_message)
            self.assertIn("Misses: 1", log_message)

    def test_stats_logging_without_tracking(self):
        """Test that log_stats_now does nothing when tracking is disabled."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=False)

        cache.put("key1", "value1")
        cache.get("key1")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            # Should not log anything
            self.assertEqual(mock_log.call_count, 0)

    def test_stats_logging_interval_timing(self):
        """Test that stats logging respects the interval timing."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=1.0)

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            # Multiple operations within interval
            for i in range(10):
                cache.put(f"key{i}", f"value{i}")
                cache.get(f"key{i}")
                time.sleep(0.05)  # Total time < 1.0

            # Should not have logged yet
            self.assertEqual(mock_log.call_count, 0)

            # Wait for interval to pass
            time.sleep(0.6)
            cache.get("key1")  # Trigger check

            # Now should have logged once
            self.assertEqual(mock_log.call_count, 1)

    def test_stats_logging_with_updates(self):
        """Test stats logging includes update counts."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        cache.put("key1", "value1")
        cache.put("key1", "updated_value1")  # Update
        cache.put("key1", "updated_again")  # Another update

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            cache.log_stats_now()

            log_message = mock_log.call_args[0][0]
            self.assertIn("Updates: 2", log_message)
            self.assertIn("Puts: 1", log_message)

    def test_stats_logging_combined_with_persistence(self):
        """Test that stats logging and persistence work together."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(
            5,
            track_stats=True,
            persistence_interval=1.0,
            persistence_path=self.test_file,
            stats_logging_interval=0.5,
        )

        cache.put("key1", "value1")

        with patch.object(
            logging.getLogger("nemoguardrails.cache.lfu"), "info"
        ) as mock_log:
            # Wait for stats logging interval
            time.sleep(0.6)
            cache.get("key1")  # Trigger stats log

            self.assertEqual(mock_log.call_count, 1)
            self.assertFalse(os.path.exists(self.test_file))  # Not persisted yet

            # Wait for persistence interval
            time.sleep(0.5)
            cache.get("key1")  # Trigger persistence

            self.assertTrue(os.path.exists(self.test_file))  # Now persisted
            # Stats log might trigger again if interval passed
            self.assertGreaterEqual(mock_log.call_count, 1)

    def test_stats_log_format_percentages(self):
        """Test that percentages in stats log are formatted correctly."""
        import logging
        from unittest.mock import patch

        cache = LFUCache(5, track_stats=True, stats_logging_interval=0.1)

        # Test various hit rates
        test_cases = [
            (0, 0, "0.00%"),  # No requests
            (1, 0, "100.00%"),  # All hits
            (0, 1, "0.00%"),  # All misses
            (1, 1, "50.00%"),  # 50/50
            (2, 1, "66.67%"),  # 2/3
            (99, 1, "99.00%"),  # High hit rate
        ]

        for hits, misses, expected_rate in test_cases:
            cache.reset_stats()

            # Generate hits
            if hits > 0:
                cache.put("hit_key", "value")
                for _ in range(hits):
                    cache.get("hit_key")

            # Generate misses
            for i in range(misses):
                cache.get(f"miss_key_{i}")

            with patch.object(
                logging.getLogger("nemoguardrails.cache.lfu"), "info"
            ) as mock_log:
                cache.log_stats_now()

                if hits > 0 or misses > 0:
                    log_message = mock_log.call_args[0][0]
                    self.assertIn(f"Hit Rate: {expected_rate}", log_message)


class TestContentSafetyCacheStatsConfig(unittest.TestCase):
    """Test cache stats configuration in content safety context."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_file = tempfile.mktemp()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_cache_config_with_stats_disabled(self):
        """Test cache configuration with stats disabled."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True, capacity_per_model=1000, stats=CacheStatsConfig(enabled=False)
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_with_stats_tracking_only(self):
        """Test cache configuration with stats tracking but no logging."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=None),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertFalse(cache.supports_stats_logging())
        self.assertIsNone(cache.stats_logging_interval)

    def test_cache_config_with_stats_logging(self):
        """Test cache configuration with stats tracking and logging."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=60.0),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertTrue(cache.supports_stats_logging())
        self.assertEqual(cache.stats_logging_interval, 60.0)

    def test_cache_config_default_stats(self):
        """Test cache configuration with default stats settings."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import ModelCacheConfig, ModelConfig

        cache_config = ModelCacheConfig(enabled=True)

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertFalse(cache.track_stats)  # Default is disabled
        self.assertFalse(cache.supports_stats_logging())

    def test_cache_config_stats_with_persistence(self):
        """Test cache configuration with both stats and persistence."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CachePersistenceConfig,
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=30.0),
            persistence=CachePersistenceConfig(
                enabled=True, interval=60.0, path=self.test_file
            ),
        )

        model_config = ModelConfig(cache=cache_config)
        manager = ContentSafetyManager(model_config)

        cache = manager.get_cache_for_model("test_model")
        self.assertIsNotNone(cache)
        self.assertTrue(cache.track_stats)
        self.assertTrue(cache.supports_stats_logging())
        self.assertEqual(cache.stats_logging_interval, 30.0)
        self.assertTrue(cache.supports_persistence())
        self.assertEqual(cache.persistence_interval, 60.0)

    def test_cache_config_from_dict(self):
        """Test cache configuration creation from dictionary."""
        from nemoguardrails.rails.llm.config import ModelCacheConfig

        config_dict = {
            "enabled": True,
            "capacity_per_model": 5000,
            "stats": {"enabled": True, "log_interval": 120.0},
        }

        cache_config = ModelCacheConfig(**config_dict)
        self.assertTrue(cache_config.enabled)
        self.assertEqual(cache_config.capacity_per_model, 5000)
        self.assertTrue(cache_config.stats.enabled)
        self.assertEqual(cache_config.stats.log_interval, 120.0)

    def test_cache_config_stats_validation(self):
        """Test cache configuration validation for stats settings."""
        from nemoguardrails.rails.llm.config import CacheStatsConfig

        # Valid configurations
        stats1 = CacheStatsConfig(enabled=True, log_interval=60.0)
        self.assertTrue(stats1.enabled)
        self.assertEqual(stats1.log_interval, 60.0)

        stats2 = CacheStatsConfig(enabled=True, log_interval=None)
        self.assertTrue(stats2.enabled)
        self.assertIsNone(stats2.log_interval)

        stats3 = CacheStatsConfig(enabled=False, log_interval=60.0)
        self.assertFalse(stats3.enabled)
        self.assertEqual(stats3.log_interval, 60.0)

    def test_multiple_model_caches_with_stats(self):
        """Test multiple model caches each with their own stats configuration."""
        from nemoguardrails.library.content_safety.manager import ContentSafetyManager
        from nemoguardrails.rails.llm.config import (
            CacheStatsConfig,
            ModelCacheConfig,
            ModelConfig,
        )

        cache_config = ModelCacheConfig(
            enabled=True,
            capacity_per_model=1000,
            stats=CacheStatsConfig(enabled=True, log_interval=30.0),
        )

        model_config = ModelConfig(
            cache=cache_config, model_mapping={"model_alias": "actual_model"}
        )
        manager = ContentSafetyManager(model_config)

        # Get caches for different models
        cache1 = manager.get_cache_for_model("model1")
        cache2 = manager.get_cache_for_model("model2")
        cache_alias = manager.get_cache_for_model("model_alias")
        cache_actual = manager.get_cache_for_model("actual_model")

        # All should have stats enabled
        self.assertTrue(cache1.track_stats)
        self.assertTrue(cache2.track_stats)
        self.assertTrue(cache_alias.track_stats)

        # Alias should resolve to same cache as actual
        self.assertIs(cache_alias, cache_actual)


if __name__ == "__main__":
    unittest.main()
