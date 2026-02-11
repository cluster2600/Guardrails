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

"""Top-level Guardrails interface module.

This module provides a simplified, user-friendly interface for interacting with
NeMo Guardrails. The Guardrails class wraps the LLMRails functionality and provides
a streamlined API for generating LLM responses with programmable guardrails.
"""

import logging
from enum import Enum
from typing import AsyncIterator, Optional, Tuple, TypeAlias, Union, overload

from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue
from nemoguardrails.guardrails.config_manager import ConfigId, ConfigManager
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse

# Queue configuration constants
MAX_QUEUE_SIZE = 100
MAX_CONCURRENCY = 10

log = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Enumeration of message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CONTEXT = "context"
    EVENT = "event"
    TOOL = "tool"


LLMMessages: TypeAlias = list[dict[str, str]]


class Guardrails:
    """Top-level interface for NeMo Guardrails functionality."""

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initialize a Guardrails instance."""

        self.config = config
        self.llm = llm
        self.verbose = verbose

        # Config management
        self._config_manager = ConfigManager()
        self._llmrails_instances: dict[str, LLMRails] = {}

        # Register default config and create LLMRails instance
        self._config_manager.create_config(ConfigId.DEFAULT, config)
        self.llmrails = LLMRails(config, llm, verbose)
        self._llmrails_instances[ConfigId.DEFAULT] = self.llmrails

        # Async work queue for managing concurrent generate_async requests
        self._generate_async_queue: AsyncWorkQueue = AsyncWorkQueue(
            name="generate_async_queue",
            max_queue_size=MAX_QUEUE_SIZE,
            max_concurrency=MAX_CONCURRENCY,
            reject_on_full=True,
        )

        # List of all queues for lifecycle management
        self._queues = [self._generate_async_queue]

    @staticmethod
    def _convert_to_messages(prompt: str | None = None, messages: LLMMessages | None = None) -> LLMMessages:
        """Convert prompt or messages to LLMRails standard format"""

        # Priority: messages first, then prompt
        if messages:
            return messages

        if prompt:
            # Convert string prompt to standard format
            return [{"role": "user", "content": prompt}]

        raise ValueError("Neither prompt nor messages provided for generation")

    # Configuration CRUD public methods

    def create_config(self, config_id: str, config: RailsConfig) -> None:
        """Create a new config and associated LLMRails instance.

        Args:
            config_id: Unique identifier for the config
            config: RailsConfig instance to store

        Raises:
            ValueError: If a config with the given ID already exists
        """
        self._config_manager.create_config(config_id, config)
        self._llmrails_instances[config_id] = LLMRails(config, self.llm, self.verbose)

    def get_config(self, config_id: str) -> RailsConfig:
        """Get a config by ID.

        Args:
            config_id: The ID of the config to retrieve

        Returns:
            The RailsConfig instance

        Raises:
            KeyError: If no config exists with the given ID
        """
        return self._config_manager.get_config(config_id)

    def list_configs(self) -> dict[str, RailsConfig]:
        """List all configs.

        Returns:
            Dictionary mapping config_id to RailsConfig instances
        """
        return self._config_manager.list_configs()

    def replace_config(self, config_id: str, config: RailsConfig) -> None:
        """Replace a config and recreate its LLMRails instance.

        Args:
            config_id: The ID of the config to replace
            config: New RailsConfig instance

        Raises:
            KeyError: If no config exists with the given ID
        """
        self._config_manager.replace_config(config_id, config)
        self._llmrails_instances[config_id] = LLMRails(config, self.llm, self.verbose)

    def update_config(self, config_id: str, updates: dict) -> None:
        """Update a config and recreate its LLMRails instance.

        Args:
            config_id: The ID of the config to update
            updates: Dictionary of field updates to apply

        Raises:
            KeyError: If no config exists with the given ID
        """
        self._config_manager.update_config(config_id, updates)
        updated_config = self._config_manager.get_config(config_id)
        self._llmrails_instances[config_id] = LLMRails(updated_config, self.llm, self.verbose)

    def delete_config(self, config_id: str) -> None:
        """Delete a config and its LLMRails instance.

        Args:
            config_id: The ID of the config to delete

        Raises:
            KeyError: If no config exists with the given ID
        """
        self._config_manager.delete_config(config_id)
        if config_id in self._llmrails_instances:
            del self._llmrails_instances[config_id]

    def get_llmrails(self, config_id: str = ConfigId.DEFAULT) -> LLMRails:
        """Get the LLMRails instance for a specific config.

        Args:
            config_id: The ID of the config (defaults to ConfigId.DEFAULT)

        Returns:
            The LLMRails instance

        Raises:
            KeyError: If no LLMRails instance exists for the given config ID
        """
        if config_id not in self._llmrails_instances:
            raise KeyError(f"No LLMRails instance for config '{config_id}'")
        return self._llmrails_instances[config_id]

    # Datapath inference methods

    def generate(
        self,
        prompt: str | None = None,
        messages: LLMMessages | None = None,
        config_id: str = ConfigId.DEFAULT,
        **kwargs,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate an LLM response synchronously with guardrails applied.

        Args:
            prompt: Optional text prompt
            messages: Optional list of message dicts
            config_id: Config ID to use (defaults to ConfigId.DEFAULT)
            **kwargs: Additional arguments passed to LLMRails.generate()

        Returns:
            Generated response from the LLM
        """
        messages = self._convert_to_messages(prompt, messages)
        llmrails = self.get_llmrails(config_id)
        return llmrails.generate(messages=messages, **kwargs)

    @overload
    async def generate_async(self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs) -> str: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> dict: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> tuple[dict, dict]: ...

    async def generate_async(
        self,
        prompt: str | None = None,
        messages: LLMMessages | None = None,
        config_id: str = ConfigId.DEFAULT,
        **kwargs,
    ) -> str | dict | GenerationResponse | tuple[dict, dict]:
        """Generate an LLM response asynchronously with guardrails applied.

        Args:
            prompt: Optional text prompt
            messages: Optional list of message dicts
            config_id: Config ID to use (defaults to ConfigId.DEFAULT)
            **kwargs: Additional arguments passed to LLMRails.generate_async()

        Returns:
            Generated response from the LLM
        """
        messages = self._convert_to_messages(prompt, messages)
        llmrails = self.get_llmrails(config_id)

        # Submit to work queue for concurrency control
        response = await self._generate_async_queue.submit(llmrails.generate_async, messages=messages, **kwargs)
        return response

    def stream_async(
        self,
        prompt: str | None = None,
        messages: LLMMessages | None = None,
        config_id: str = ConfigId.DEFAULT,
        **kwargs,
    ) -> AsyncIterator[str | dict]:
        """Generate an LLM response asynchronously with streaming support.

        Args:
            prompt: Optional text prompt
            messages: Optional list of message dicts
            config_id: Config ID to use (defaults to ConfigId.DEFAULT)
            **kwargs: Additional arguments passed to LLMRails.stream_async()

        Returns:
            Async iterator yielding response chunks
        """
        messages = self._convert_to_messages(prompt, messages)
        llmrails = self.get_llmrails(config_id)
        return llmrails.stream_async(messages=messages, **kwargs)

    def explain(self) -> ExplainInfo:
        """Get the latest ExplainInfo object for debugging."""
        return self.llmrails.explain()

    def update_llm(self, llm: Union[BaseLLM, BaseChatModel]) -> None:
        """Replace the main LLM with a new one."""
        self.llm = llm
        self.llmrails.update_llm(llm)

    async def startup(self) -> None:
        """Lifecycle method to create worker threads and infrastructure"""
        for queue in self._queues:
            await queue.start()

    async def shutdown(self) -> None:
        """Lifecycle method to cleanly shutdown worker threads and infrastructure"""
        for queue in self._queues:
            await queue.stop()

    async def __aenter__(self):
        """Async context manager entry - starts the queues."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - shuts down the queues."""
        await self.shutdown()
