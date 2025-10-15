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

"""Multi-provider ClientRails wrapper using adapter pattern.

This module provides a wrapper around NeMo Guardrails that supports multiple
LLM providers (OpenAI, Anthropic, etc.) using the adapter pattern.

The wrapper uses the underlying LLMRails.generate() method with input/output
rail options and enable_rails_exceptions to detect violations.

Usage:
    ```python
    from openai import OpenAI
    from anthropic import Anthropic
    from nemoguardrails import RailsConfig
    from nemoguardrails.rails.llm.llmrails import LLMRails
    from nemoguardrails.clients.guard import ClientRails

    # Create config with enable_rails_exceptions enabled
    config = RailsConfig.from_path("path")
    config.enable_rails_exceptions = True

    # Create the underlying rails
    rails = LLMRails(config)

    # Wrap with the guard client
    guard = ClientRails(rails)

    # Works with OpenAI
    openai_client = OpenAI()
    guarded_openai_client = guard(openai_client)
    response = guarded_openai_client.chat.completions.create(...)

    # Works with Anthropic
    anthropic_client = Anthropic()
    guarded_anthropic_client = guard(anthropic_client)
    response = guarded_anthropic_client.messages.create(...)
    ```
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar, Union, cast, overload

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

ClientT = TypeVar("ClientT")

from .adapters import ProviderAdapter, ProviderAdapterFactory
from .rail_extractors import extract_rails_status_results
from .types import GuardResult, RailsMessage, RailType

__all__ = ["GuardrailViolation", "GuardResult", "ClientRails"]

logger = logging.getLogger(__name__)


class _GuardedClient:
    """Generic proxy wrapper using adapter for provider-specific logic."""

    def __init__(self, client: Any, rails: "ClientRails", adapter: ProviderAdapter):
        self._client = client
        self._rails = rails
        self._adapter = adapter
        self._wrapped_resources: dict[str, _GuardedResource] = {}

    def __getattr__(self, name: str) -> Any:
        """Intercept or delegate based on adapter."""
        # Check if this attribute should be intercepted
        if name in self._adapter.get_intercept_paths():
            # Wrap the resource
            if name not in self._wrapped_resources:
                resource = getattr(self._client, name)
                self._wrapped_resources[name] = _GuardedResource(
                    resource, self._rails, self._adapter, name
                )
            return self._wrapped_resources[name]

        # Delegate to original client
        return getattr(self._client, name)

    def __dir__(self):
        """Return attributes for autocomplete/introspection."""
        own_attrs = list(object.__dir__(self))
        client_attrs = dir(self._client)
        return sorted(set(own_attrs + client_attrs))

    @property
    def unwrapped(self) -> Any:
        """Access the underlying unwrapped client.

        Use this for isinstance() checks or when you need
        the original client instance.

        Returns:
            The original client instance

        Example:
            >>> from openai import OpenAI
            >>> guarded_client = guard(OpenAI())
            >>> isinstance(guarded_client.unwrapped, OpenAI)
            True
        """
        return self._client

    def __repr__(self) -> str:
        return f"_GuardedClient({self._adapter.get_provider_name()})"


class _GuardedResource:
    """Generic guarded resource using adapter for extraction."""

    def __init__(
        self,
        resource: Any,
        rails: "ClientRails",
        adapter: ProviderAdapter,
        resource_name: str,
    ):
        self._resource = resource
        self._rails = rails
        self._adapter = adapter
        self._resource_name = resource_name
        self._wrapped_sub_resources: dict[str, _GuardedResource] = {}

    def __getattr__(self, name: str) -> Any:
        """Intercept methods or delegate."""
        attr = getattr(self._resource, name)

        # If it's a callable method that should be wrapped
        if callable(attr) and self._adapter.should_wrap_method(name):
            return self._create_guarded_method(attr, name)

        # If it's another resource (e.g., chat.completions)
        # Wrap it recursively
        if hasattr(attr, "__dict__") and not callable(attr):
            if name not in self._wrapped_sub_resources:
                self._wrapped_sub_resources[name] = _GuardedResource(
                    attr, self._rails, self._adapter, f"{self._resource_name}.{name}"
                )
            return self._wrapped_sub_resources[name]

        return attr

    def __dir__(self):
        """Return attributes for autocomplete/introspection."""
        own_attrs = list(object.__dir__(self))
        resource_attrs = dir(self._resource)
        return sorted(set(own_attrs + resource_attrs))

    @property
    def unwrapped(self) -> Any:
        """Access the underlying unwrapped resource.

        Use this for isinstance() checks or when you need
        the original resource instance.

        Returns:
            The original resource instance

        Example:
            >>> guarded_chat = guarded_client.chat
            >>> original_chat = guarded_chat.unwrapped
        """
        return self._resource

    def _create_guarded_method(self, original_method: Any, method_name: str) -> Any:
        """Create guarded version of a method."""
        is_async = inspect.iscoroutinefunction(original_method)

        if is_async:
            return self._create_async_guarded_method(original_method, method_name)
        else:
            return self._create_sync_guarded_method(original_method, method_name)

    def _create_async_guarded_method(
        self, original_method: Any, method_name: str
    ) -> Any:
        """Create async guarded method."""

        async def guarded_method(*args: Any, **kwargs: Any) -> Any:
            input_messages = self._adapter.extract_messages_for_input_check(
                method_name, *args, **kwargs
            )

            if self._rails.has_input_rails() and input_messages:
                self._rails.check_input(input_messages)

            response = await original_method(*args, **kwargs)

            if self._rails.has_output_rails():
                output_messages = self._adapter.extract_messages_for_output_check(
                    method_name, response, input_messages
                )
                self._rails.check_output(output_messages)

            return response

        return guarded_method

    def _create_sync_guarded_method(
        self, original_method: Any, method_name: str
    ) -> Any:
        """Create sync guarded method."""

        def guarded_method(*args: Any, **kwargs: Any) -> Any:
            input_messages = self._adapter.extract_messages_for_input_check(
                method_name, *args, **kwargs
            )

            if self._rails.has_input_rails() and input_messages:
                self._rails.check_input(input_messages)

            response = original_method(*args, **kwargs)

            if self._rails.has_output_rails():
                output_messages = self._adapter.extract_messages_for_output_check(
                    method_name, response, input_messages
                )
                self._rails.check_output(output_messages)

            return response

        return guarded_method


class GuardrailViolation(Exception):
    """Raised when a guardrail check fails."""

    def __init__(
        self, message: str, rail_type: RailType, guard_name: str | None = None
    ):
        self.rail_type = rail_type
        self.guard_name = guard_name
        self.message = message

        formatted_msg = f"{rail_type.value} rails `{guard_name}` failed"
        if message and not message.startswith("execute"):
            formatted_msg += f": {message}"

        super().__init__(formatted_msg)


class ClientRails:
    """Multi-provider wrapper for adding guardrails to LLM clients.

    This class wraps the NeMo Guardrails LLMRails with functionality to wrap
    LLM clients (OpenAI, Anthropic, etc.) with custom guardrail functions.
    Provider-specific logic is handled by adapters.
    """

    def __init__(
        self,
        rails: LLMRails,
        raise_on_violation: bool = True,
    ):
        """Initialize the ClientRails wrapper.

        Args:
            rails: The underlying NeMo Guardrails LLMRails instance
            raise_on_violation: Whether to raise exceptions on guardrail violations (default: True)
        """
        self._rails = rails
        self.raise_on_violation = raise_on_violation

    @overload
    def __call__(self, client: "OpenAI", provider_hint: str | None = None) -> "OpenAI":
        ...

    @overload
    def __call__(
        self, client: "Anthropic", provider_hint: str | None = None
    ) -> "Anthropic":
        ...

    @overload
    def __call__(self, client: ClientT, provider_hint: str | None = None) -> ClientT:
        ...

    def __call__(self, client: Any, provider_hint: str | None = None) -> Any:
        """Wrap an LLM client with guardrails.

        Provider is auto-detected from the client type.

        Args:
            client: LLM client instance (OpenAI, Anthropic, etc.)
            provider_hint: Optional provider hint to skip detection

        Returns:
            Wrapped client with guardrail interception

        Raises:
            ValueError: If provider cannot be detected

        Example:
            >>> from openai import OpenAI
            >>> rails = ClientRails(...)
            >>> guarded = rails(OpenAI())
            >>> guarded = rails(OpenAI(), provider_hint="openai")
        """
        adapter = ProviderAdapterFactory.create_adapter_for_client(
            client, provider_hint
        )

        logger.info(
            "Wrapping %s client with guardrails",
            adapter.get_provider_name(),
        )

        return _GuardedClient(client, self, adapter)

    def has_input_rails(self) -> bool:
        """Check if input rails are configured.

        Returns:
            True if input rails are configured, False otherwise
        """
        return len(self._rails.config.rails.input.flows) > 0

    def has_output_rails(self) -> bool:
        """Check if output rails are configured.

        Returns:
            True if output rails are configured, False otherwise
        """
        return len(self._rails.config.rails.output.flows) > 0

    def check_input(self, messages: List[RailsMessage]) -> list[GuardResult]:
        """Run input rails on messages.

        Args:
            messages: List of conversation messages in NemoGuardrails format

        Returns:
            List of GuardResult objects (one per input rail)

        Raises:
            GuardrailViolation: If any guard fails and raise_on_violation=True
        """
        options = {
            "rails": {
                "input": True,
                "output": False,
                "dialog": False,
                "retrieval": False,
            }
        }

        response = self._rails.generate(
            messages=cast(List[dict], messages), options=options
        )
        results = extract_rails_status_results(response, RailType.INPUT)

        if self.raise_on_violation:
            for result in results:
                if not result.passed:
                    raise GuardrailViolation(
                        result.reason,
                        rail_type=RailType.INPUT,
                        guard_name=result.guard_name,
                    )

        return results

    def check_output(self, messages: List[RailsMessage]) -> list[GuardResult]:
        """Run all output rails on messages.

        Args:
            messages: List of conversation messages including bot response

        Returns:
            List of GuardResult objects (one per output rail)

        Raises:
            GuardrailViolation: If any guard fails and raise_on_violation=True
        """
        options = {
            "rails": {
                "input": False,
                "output": True,
                "dialog": False,
                "retrieval": False,
            }
        }

        response = self._rails.generate(
            messages=cast(List[dict], messages), options=options
        )
        results = extract_rails_status_results(response, RailType.OUTPUT)

        if self.raise_on_violation:
            for result in results:
                if not result.passed:
                    raise GuardrailViolation(
                        result.reason,
                        rail_type=RailType.OUTPUT,
                        guard_name=result.guard_name,
                    )

        return results
