# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the ConfigManager class.

These tests verify the configuration management implementation including:
- CRUD operations (create, read, update, delete)
- Exception handling for error cases
- Thread safety
- ConfigId constants
- Pydantic model updates
"""

import threading
from unittest.mock import MagicMock

import pytest

from nemoguardrails.guardrails.config_manager import ConfigId, ConfigManager
from nemoguardrails.rails.llm.config import RailsConfig


@pytest.fixture
def config_manager():
    """Create a fresh ConfigManager instance for testing."""
    return ConfigManager()


@pytest.fixture
def mock_rails_config():
    """Create a mock RailsConfig for testing."""
    config = MagicMock(spec=RailsConfig)
    # Mock model_copy to return a new mock with updated attributes
    config.model_copy.return_value = config
    return config


@pytest.fixture
def mock_rails_config_2():
    """Create a second mock RailsConfig for testing."""
    config = MagicMock(spec=RailsConfig)
    config.model_copy.return_value = config
    return config


class TestConfigIdConstants:
    """Tests for ConfigId constants."""

    def test_default_constant_exists(self):
        """Test that ConfigId.DEFAULT constant exists and has expected value."""
        assert hasattr(ConfigId, "DEFAULT")
        assert ConfigId.DEFAULT == "default"
        assert isinstance(ConfigId.DEFAULT, str)


class TestCreateConfig:
    """Tests for ConfigManager.create_config() method."""

    def test_create_config_success(self, config_manager, mock_rails_config):
        """Test successful creation of a new config."""
        config_manager.create_config("test_config", mock_rails_config)

        # Verify config was stored
        stored_config = config_manager.get_config("test_config")
        assert stored_config == mock_rails_config

    def test_create_config_with_default_id(self, config_manager, mock_rails_config):
        """Test creating a config with the special DEFAULT id."""
        config_manager.create_config(ConfigId.DEFAULT, mock_rails_config)

        stored_config = config_manager.get_config(ConfigId.DEFAULT)
        assert stored_config == mock_rails_config

    def test_create_config_duplicate_raises_error(self, config_manager, mock_rails_config):
        """Test that creating a config with duplicate ID raises ValueError."""
        config_manager.create_config("duplicate_id", mock_rails_config)

        with pytest.raises(ValueError, match="Config with id 'duplicate_id' already exists"):
            config_manager.create_config("duplicate_id", mock_rails_config)

    def test_create_multiple_configs(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test creating multiple configs with different IDs."""
        config_manager.create_config("config_1", mock_rails_config)
        config_manager.create_config("config_2", mock_rails_config_2)

        assert config_manager.get_config("config_1") == mock_rails_config
        assert config_manager.get_config("config_2") == mock_rails_config_2


class TestGetConfig:
    """Tests for ConfigManager.get_config() method."""

    def test_get_config_success(self, config_manager, mock_rails_config):
        """Test successful retrieval of an existing config."""
        config_manager.create_config("test_config", mock_rails_config)

        result = config_manager.get_config("test_config")
        assert result == mock_rails_config

    def test_get_config_not_found_raises_error(self, config_manager):
        """Test that getting a non-existent config raises KeyError."""
        with pytest.raises(KeyError, match="Config with id 'nonexistent' not found"):
            config_manager.get_config("nonexistent")

    def test_get_config_after_create(self, config_manager, mock_rails_config):
        """Test that get_config returns the exact config that was created."""
        config_manager.create_config("test_config", mock_rails_config)

        retrieved = config_manager.get_config("test_config")
        assert retrieved is mock_rails_config


class TestListConfigs:
    """Tests for ConfigManager.list_configs() method."""

    def test_list_configs_empty(self, config_manager):
        """Test listing configs when ConfigManager is empty."""
        result = config_manager.list_configs()

        assert result == {}
        assert isinstance(result, dict)

    def test_list_configs_single(self, config_manager, mock_rails_config):
        """Test listing configs with a single config."""
        config_manager.create_config("test_config", mock_rails_config)

        result = config_manager.list_configs()

        assert len(result) == 1
        assert "test_config" in result
        assert result["test_config"] == mock_rails_config

    def test_list_configs_multiple(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test listing configs with multiple configs."""
        config_manager.create_config("config_1", mock_rails_config)
        config_manager.create_config("config_2", mock_rails_config_2)

        result = config_manager.list_configs()

        assert len(result) == 2
        assert result["config_1"] == mock_rails_config
        assert result["config_2"] == mock_rails_config_2

    def test_list_configs_returns_copy(self, config_manager, mock_rails_config):
        """Test that list_configs returns a copy, not the internal dict."""
        config_manager.create_config("test_config", mock_rails_config)

        result1 = config_manager.list_configs()
        result2 = config_manager.list_configs()

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2


class TestReplaceConfig:
    """Tests for ConfigManager.replace_config() method."""

    def test_replace_config_success(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test successful replacement of an existing config."""
        config_manager.create_config("test_config", mock_rails_config)

        # Replace with new config
        config_manager.replace_config("test_config", mock_rails_config_2)

        stored_config = config_manager.get_config("test_config")
        assert stored_config == mock_rails_config_2
        assert stored_config != mock_rails_config

    def test_replace_config_not_found_raises_error(self, config_manager, mock_rails_config):
        """Test that replacing a non-existent config raises KeyError."""
        with pytest.raises(KeyError, match="Config with id 'nonexistent' not found"):
            config_manager.replace_config("nonexistent", mock_rails_config)

    def test_replace_config_preserves_id(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test that replacing a config preserves the same ID."""
        config_manager.create_config("test_config", mock_rails_config)
        config_manager.replace_config("test_config", mock_rails_config_2)

        configs = config_manager.list_configs()
        assert "test_config" in configs
        assert len(configs) == 1


class TestUpdateConfig:
    """Tests for ConfigManager.update_config() method."""

    def test_update_config_success(self, config_manager, mock_rails_config):
        """Test successful update of a config."""
        config_manager.create_config("test_config", mock_rails_config)

        updates = {"verbose": True, "some_field": "new_value"}
        config_manager.update_config("test_config", updates)

        # Verify model_copy was called with the updates
        mock_rails_config.model_copy.assert_called_once_with(update=updates)

    def test_update_config_not_found_raises_error(self, config_manager):
        """Test that updating a non-existent config raises KeyError."""
        with pytest.raises(KeyError, match="Config with id 'nonexistent' not found"):
            config_manager.update_config("nonexistent", {"field": "value"})

    def test_update_config_empty_updates(self, config_manager, mock_rails_config):
        """Test updating a config with an empty updates dict."""
        config_manager.create_config("test_config", mock_rails_config)

        updates = {}
        config_manager.update_config("test_config", updates)

        # Should still call model_copy even with empty dict
        mock_rails_config.model_copy.assert_called_once_with(update=updates)

    def test_update_config_multiple_fields(self, config_manager, mock_rails_config):
        """Test updating multiple fields at once."""
        config_manager.create_config("test_config", mock_rails_config)

        updates = {
            "field1": "value1",
            "field2": 42,
            "field3": True,
        }
        config_manager.update_config("test_config", updates)

        mock_rails_config.model_copy.assert_called_once_with(update=updates)


class TestDeleteConfig:
    """Tests for ConfigManager.delete_config() method."""

    def test_delete_config_success(self, config_manager, mock_rails_config):
        """Test successful deletion of a config."""
        config_manager.create_config("test_config", mock_rails_config)
        assert "test_config" in config_manager.list_configs()

        config_manager.delete_config("test_config")
        assert "test_config" not in config_manager.list_configs()

    def test_delete_config_not_found_raises_error(self, config_manager):
        """Test that deleting a non-existent config raises KeyError."""
        with pytest.raises(KeyError, match="Config with id 'nonexistent' not found"):
            config_manager.delete_config("nonexistent")

    def test_delete_config_cannot_get_after_delete(self, config_manager, mock_rails_config):
        """Test that get_config fails after deletion."""
        config_manager.create_config("test_config", mock_rails_config)
        config_manager.delete_config("test_config")

        with pytest.raises(KeyError, match="Config with id 'test_config' not found"):
            config_manager.get_config("test_config")

    def test_delete_one_of_multiple_configs(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test deleting one config among multiple configs."""
        config_manager.create_config("config_1", mock_rails_config)
        config_manager.create_config("config_2", mock_rails_config_2)

        config_manager.delete_config("config_1")

        configs = config_manager.list_configs()
        assert "config_1" not in configs
        assert "config_2" in configs
        assert len(configs) == 1


class TestThreadSafety:
    """Tests for thread safety of ConfigManager operations."""

    def test_concurrent_creates(self, config_manager):
        """Test that concurrent create operations are thread-safe."""
        num_threads = 10
        threads = []
        errors = []

        def create_config(index):
            try:
                config = MagicMock(spec=RailsConfig)
                config_manager.create_config(f"config_{index}", config)
            except Exception as e:
                errors.append(e)

        # Create configs concurrently
        for i in range(num_threads):
            thread = threading.Thread(target=create_config, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all configs were created
        configs = config_manager.list_configs()
        assert len(configs) == num_threads

    def test_concurrent_reads(self, config_manager, mock_rails_config):
        """Test that concurrent read operations are thread-safe."""
        config_manager.create_config("test_config", mock_rails_config)

        num_threads = 20
        threads = []
        results = []
        errors = []

        def read_config():
            try:
                config = config_manager.get_config("test_config")
                results.append(config)
            except Exception as e:
                errors.append(e)

        # Read config concurrently
        for _ in range(num_threads):
            thread = threading.Thread(target=read_config)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all reads returned the same config
        assert len(results) == num_threads
        assert all(config == mock_rails_config for config in results)

    def test_concurrent_mixed_operations(self, config_manager):
        """Test that concurrent mixed operations are thread-safe."""
        num_threads = 15
        threads = []
        errors = []

        # Pre-populate with some configs
        for i in range(5):
            config = MagicMock(spec=RailsConfig)
            config.model_copy.return_value = config
            config_manager.create_config(f"existing_{i}", config)

        def mixed_operations(index):
            try:
                # Create
                if index % 3 == 0:
                    config = MagicMock(spec=RailsConfig)
                    config_manager.create_config(f"new_{index}", config)
                # Read
                elif index % 3 == 1:
                    config_manager.list_configs()
                # Update
                else:
                    try:
                        config_manager.update_config(f"existing_{index % 5}", {"field": index})
                    except KeyError:
                        pass  # Config might have been deleted
            except Exception as e:
                errors.append(e)

        # Perform mixed operations concurrently
        for i in range(num_threads):
            thread = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no unexpected errors occurred
        assert len(errors) == 0


class TestCRUDWorkflow:
    """Integration tests for complete CRUD workflows."""

    def test_full_crud_lifecycle(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test a complete CRUD lifecycle: create, read, update, replace, delete."""
        # Create
        config_manager.create_config("lifecycle_test", mock_rails_config)
        assert config_manager.get_config("lifecycle_test") == mock_rails_config

        # Read
        configs = config_manager.list_configs()
        assert "lifecycle_test" in configs

        # Update
        config_manager.update_config("lifecycle_test", {"field": "updated"})
        mock_rails_config.model_copy.assert_called_with(update={"field": "updated"})

        # Replace
        config_manager.replace_config("lifecycle_test", mock_rails_config_2)
        assert config_manager.get_config("lifecycle_test") == mock_rails_config_2

        # Delete
        config_manager.delete_config("lifecycle_test")
        assert "lifecycle_test" not in config_manager.list_configs()

    def test_create_after_delete_same_id(self, config_manager, mock_rails_config, mock_rails_config_2):
        """Test that a config can be re-created with the same ID after deletion."""
        # Create and delete
        config_manager.create_config("reusable_id", mock_rails_config)
        config_manager.delete_config("reusable_id")

        # Create again with same ID
        config_manager.create_config("reusable_id", mock_rails_config_2)

        # Verify new config is stored
        stored = config_manager.get_config("reusable_id")
        assert stored == mock_rails_config_2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_config_id(self, config_manager, mock_rails_config):
        """Test that empty config_id raises ValueError."""
        with pytest.raises(ValueError, match="No config_id provided"):
            config_manager.create_config("", mock_rails_config)

    def test_very_long_config_id(self, config_manager, mock_rails_config):
        """Test config with a very long ID."""
        long_id = "x" * 1000
        config_manager.create_config(long_id, mock_rails_config)
        assert config_manager.get_config(long_id) == mock_rails_config
