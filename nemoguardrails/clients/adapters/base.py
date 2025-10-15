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

"""Base adapter interface for LLM providers.

This module defines the abstract base class that all provider adapters
must implement to work with the LLMRails wrapper.
"""

from abc import ABC, abstractmethod
from typing import Any, List

from ..types import RailsMessage
from ..validators import MessageValidator


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific adapters.

    Each LLM provider (OpenAI, Anthropic, etc.) needs an adapter that
    knows how to:
    1. Extract conversation messages from method arguments for input checking
    2. Extract conversation + response messages for output checking
    3. Determine which methods to intercept
    """

    def __init__(self, validate_messages: bool = False):
        """Initialize adapter.

        Args:
            validate_messages: If True, validate extracted messages
        """
        self.validator = MessageValidator() if validate_messages else None

    @abstractmethod
    def extract_messages_for_input_check(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> List[RailsMessage]:
        """Extract full conversation messages for input rails checking.

        This method extracts all messages from the method call arguments
        and converts them to NemoGuardrails message format. This includes
        system messages, user messages, assistant messages (with tool_calls),
        and tool messages.

        Args:
            method_name: Name of the method being called (e.g., "create")
            *args: Positional arguments to the method
            **kwargs: Keyword arguments to the method

        Returns:
            List of messages in NemoGuardrails format

        Example:
            For OpenAI:
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"}
                ]
            Returns:
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"}
                ]
        """
        pass

    @abstractmethod
    def extract_messages_for_output_check(
        self, method_name: str, response: Any, input_messages: List[RailsMessage]
    ) -> List[RailsMessage]:
        """Extract conversation + response messages for output rails checking.

        This method combines the input conversation history with the bot's
        response to create a complete conversation for output rails checking.
        Tool calls in the assistant response should be preserved.

        Args:
            method_name: Name of the method being called
            response: Response object from the LLM provider
            input_messages: Original input messages from extract_messages_for_input_check

        Returns:
            List of messages including conversation history and bot response

        Example:
            input_messages = [{"role": "user", "content": "Hello"}]
            response = <assistant response: "Hi there!">
            Returns:
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
        """
        pass

    @abstractmethod
    def get_intercept_paths(self) -> list[str]:
        """Get list of attribute paths to intercept.

        Returns:
            List of attribute names that should be wrapped with guardrails

        Example:
            For OpenAI: ["chat", "completions", "responses"]
            For Anthropic: ["messages"]
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get human-readable provider name.

        Returns:
            Provider name (e.g., "openai", "anthropic")
        """
        pass

    def should_wrap_method(self, method_name: str) -> bool:
        """Check if a method should be wrapped with guardrails.

        Args:
            method_name: Name of the method

        Returns:
            True if method should be wrapped, False otherwise

        Note:
            Default implementation wraps "create" methods. Override if needed.
        """
        return method_name == "create"

    def _validate_if_enabled(self, messages: List[RailsMessage]) -> None:
        """Validate messages if validation is enabled.

        Args:
            messages: List of messages to validate
        """
        if self.validator:
            self.validator.validate_messages(messages)
