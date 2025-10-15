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

"""Provider adapter factory for auto-detection.

This module provides automatic detection of LLM provider types and
creates the appropriate adapter for each provider.
"""

import logging
from threading import RLock
from typing import Any

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .genai import GenaiAdapter
from .openai import OpenAIAdapter

logger = logging.getLogger(__name__)


class ProviderAdapterFactory:
    """Factory for creating provider-specific adapters.

    This factory automatically detects the LLM provider based on the
    client instance and creates the appropriate adapter.

    Thread-safe for registration operations.
    """

    _lock = RLock()

    _PROVIDER_PATTERNS = {
        "openai": ["openai", "azure"],
        "anthropic": ["anthropic"],
        "genai": ["genai", "google.genai"],
    }

    _ADAPTERS = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "genai": GenaiAdapter,
    }

    @classmethod
    def detect_provider(cls, client: Any, hint: str | None = None) -> str:
        """Detect provider type from client instance.

        Args:
            client: LLM client instance (OpenAI, Anthropic, etc.)
            hint: Optional provider hint to skip detection

        Returns:
            Provider name ("openai", "anthropic", etc.)

        Raises:
            ValueError: If provider cannot be detected

        Example:
            >>> from openai import OpenAI
            >>> client = OpenAI()
            >>> ProviderAdapterFactory.detect_provider(client)
            'openai'
            >>> ProviderAdapterFactory.detect_provider(client, hint="openai")
            'openai'
        """
        if hint and hint in cls._ADAPTERS:
            logger.debug("Using hint provider '%s'", hint)
            return hint

        if hasattr(client, "__llm_provider__"):
            provider = client.__llm_provider__
            if provider in cls._ADAPTERS:
                logger.debug("Using __llm_provider__ attribute: '%s'", provider)
                return provider

        detected = cls._detect_by_duck_typing(client)
        if detected:
            return detected

        return cls._detect_by_module_name(client)

    @classmethod
    def _detect_by_duck_typing(cls, client: Any) -> str | None:
        """Detect provider by checking for provider-specific attributes.

        Args:
            client: LLM client instance

        Returns:
            Provider name if detected, None otherwise
        """
        if hasattr(client, "chat") and hasattr(client, "completions"):
            logger.info(
                "Detected 'openai' provider via duck typing (chat + completions)"
            )
            return "openai"

        if hasattr(client, "messages") and not hasattr(client, "chat"):
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                logger.info(
                    "Detected 'genai' provider via duck typing (models.generate_content)"
                )
                return "genai"
            logger.info("Detected 'anthropic' provider via duck typing (messages)")
            return "anthropic"

        if hasattr(client, "models") and callable(
            getattr(client.models, "generate_content", None)
        ):
            logger.info(
                "Detected 'genai' provider via duck typing (models.generate_content)"
            )
            return "genai"

        return None

    @classmethod
    def _detect_by_module_name(cls, client: Any) -> str:
        """Fallback detection using module name pattern matching.

        Args:
            client: LLM client instance

        Returns:
            Provider name

        Raises:
            ValueError: If provider cannot be detected
        """
        client_class = type(client).__name__
        module_name = type(client).__module__.lower()

        logger.debug(
            "Detecting provider for class=%s, module=%s",
            client_class,
            module_name,
        )

        for provider, patterns in cls._PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if pattern in module_name:
                    logger.info(
                        "Detected provider '%s' from module '%s'",
                        provider,
                        module_name,
                    )
                    return provider

        raise ValueError(
            f"Unknown provider: {client_class} from module {module_name}. "
            f"Supported providers: {list(cls._ADAPTERS.keys())}"
        )

    @classmethod
    def create_adapter(cls, provider: str) -> ProviderAdapter:
        """Create adapter for specific provider.

        Args:
            provider: Provider name ("openai", "anthropic", etc.)

        Returns:
            Provider adapter instance

        Raises:
            ValueError: If provider is not supported

        Example:
            >>> adapter = ProviderAdapterFactory.create_adapter("openai")
            >>> adapter.get_provider_name()
            'openai'
        """
        adapter_class = cls._ADAPTERS.get(provider)

        if not adapter_class:
            raise ValueError(
                f"No adapter for provider '{provider}'. "
                f"Supported: {list(cls._ADAPTERS.keys())}"
            )

        logger.debug("Creating adapter for provider '%s'", provider)
        return adapter_class()

    @classmethod
    def create_adapter_for_client(
        cls, client: Any, hint: str | None = None
    ) -> ProviderAdapter:
        """Auto-detect provider and create appropriate adapter.

        This is the main entry point for automatic provider detection.

        Args:
            client: LLM client instance
            hint: Optional provider hint to skip detection

        Returns:
            Provider adapter instance

        Raises:
            ValueError: If provider cannot be detected or is not supported

        Example:
            >>> from openai import OpenAI
            >>> client = OpenAI()
            >>> adapter = ProviderAdapterFactory.create_adapter_for_client(client)
            >>> isinstance(adapter, OpenAIAdapter)
            True
            >>> adapter = ProviderAdapterFactory.create_adapter_for_client(client, hint="openai")
            >>> isinstance(adapter, OpenAIAdapter)
            True
        """
        provider = cls.detect_provider(client, hint)
        return cls.create_adapter(provider)

    @classmethod
    def register_adapter(
        cls, provider: str, adapter_class: type[ProviderAdapter], patterns: list[str]
    ) -> None:
        """Register a new provider adapter (for extensibility).

        This allows users to add support for custom providers.
        Thread-safe operation.

        Args:
            provider: Provider name (e.g., "custom")
            adapter_class: Adapter class implementing ProviderAdapter
            patterns: List of module name patterns for detection

        Example:
            >>> class CustomAdapter(ProviderAdapter):
            ...     # Implementation
            ...     pass
            >>> ProviderAdapterFactory.register_adapter(
            ...     "custom",
            ...     CustomAdapter,
            ...     ["custom_llm"]
            ... )
        """
        with cls._lock:
            cls._ADAPTERS[provider] = adapter_class
            cls._PROVIDER_PATTERNS[provider] = patterns
            logger.info("Registered adapter for provider '%s'", provider)

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported provider names.

        Returns:
            List of provider names

        Example:
            >>> ProviderAdapterFactory.get_supported_providers()
            ['openai', 'anthropic']
        """
        return list(cls._ADAPTERS.keys())
