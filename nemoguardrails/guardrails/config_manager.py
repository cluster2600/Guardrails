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

"""Configuration manager for managing multiple RailsConfig instances.

This module provides a ConfigManager class that implements CRUD operations
for RailsConfig objects, maintaining them in-memory as a simple data store.
"""

import logging
import threading
from typing import Dict

from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)


class ConfigId:
    """Special configuration ID constants."""

    DEFAULT = "default"


class ConfigManager:
    """Manages RailsConfig instances with CRUD operations.

    This class provides thread-safe in-memory storage for RailsConfig objects.
    It is a simple data store with no knowledge of LLMRails or other runtime objects.
    """

    def __init__(self):
        """Initialize the ConfigManager."""
        self._configs: Dict[str, RailsConfig] = {}
        self._lock = threading.Lock()

    def create_config(self, config_id: str, config: RailsConfig) -> None:
        """Create a new config with the given ID.

        Args:
            config_id: Unique identifier for the config
            config: RailsConfig instance to store

        Raises:
            ValueError: If config_id is empty or a config with the given ID already exists
        """
        with self._lock:
            if not config_id:
                raise ValueError("No config_id provided")

            if config_id in self._configs:
                raise ValueError(f"Config with id '{config_id}' already exists")

            self._configs[config_id] = config
            log.debug(f"Created config with id '{config_id}'")

    def get_config(self, config_id: str) -> RailsConfig:
        """Retrieve a config by ID.

        Args:
            config_id: The ID of the config to retrieve

        Returns:
            The RailsConfig instance

        Raises:
            KeyError: If no config exists with the given ID
        """
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")
            return self._configs[config_id]

    def list_configs(self) -> Dict[str, RailsConfig]:
        """List all managed configs.

        Returns:
            Dictionary mapping config_id to RailsConfig instances
        """
        with self._lock:
            return dict(self._configs)

    def replace_config(self, config_id: str, config: RailsConfig) -> None:
        """Replace an entire config with a new one.

        Args:
            config_id: The ID of the config to replace
            config: New RailsConfig instance

        Raises:
            KeyError: If no config exists with the given ID
        """
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")

            self._configs[config_id] = config
            log.debug(f"Replaced config with id '{config_id}'")

    def update_config(self, config_id: str, updates: dict) -> None:
        """Update specific fields of a config.

        This performs a partial update by merging the updates dict into the
        existing config using Pydantic's model_copy.

        Args:
            config_id: The ID of the config to update
            updates: Dictionary of field updates to apply

        Raises:
            KeyError: If no config exists with the given ID
        """
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")

            # Use Pydantic's model_copy to create updated config
            updated_config = self._configs[config_id].model_copy(update=updates)
            self._configs[config_id] = updated_config
            log.debug(f"Updated config with id '{config_id}'")

    def delete_config(self, config_id: str) -> None:
        """Delete a config by ID.

        Args:
            config_id: The ID of the config to delete

        Raises:
            KeyError: If no config exists with the given ID
        """
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")

            del self._configs[config_id]
            log.debug(f"Deleted config with id '{config_id}'")
