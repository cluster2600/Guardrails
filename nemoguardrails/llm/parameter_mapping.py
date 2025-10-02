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

"""Module for transforming LLM parameters between internal and provider-specific formats."""

import logging
from typing import Any, Dict, Optional

from langchain.base_language import BaseLanguageModel

log = logging.getLogger(__name__)

_llm_parameter_mappings = {}

PROVIDER_PARAMETER_MAPPINGS = {
    "huggingface": {
        "max_tokens": "max_new_tokens",
    },
    "google_vertexai": {
        "max_tokens": "max_output_tokens",
    },
}


def register_llm_parameter_mapping(
    provider: str, model_name: str, parameter_mapping: Dict[str, Optional[str]]
) -> None:
    """Register a parameter mapping for a specific provider and model combination.

    Args:
        provider: The LLM provider name
        model_name: The model name
        parameter_mapping: The parameter mapping dictionary
    """
    key = (provider, model_name)
    _llm_parameter_mappings[key] = parameter_mapping
    log.debug("Registered parameter mapping for %s/%s", provider, model_name)


def get_llm_parameter_mapping(
    provider: str, model_name: str
) -> Optional[Dict[str, Optional[str]]]:
    """Get the registered parameter mapping for a provider and model combination.

    Args:
        provider: The LLM provider name
        model_name: The model name

    Returns:
        The parameter mapping if registered, None otherwise
    """
    return _llm_parameter_mappings.get((provider, model_name))


def _infer_provider_from_module(llm: BaseLanguageModel) -> Optional[str]:
    """Infer provider name from the LLM's module path.

    This function extracts the provider name from LangChain package naming conventions:
    - langchain_openai -> openai
    - langchain_anthropic -> anthropic
    - langchain_google_genai -> google_genai
    - langchain_nvidia_ai_endpoints -> nvidia_ai_endpoints
    - langchain_community.chat_models.ollama -> ollama

    Args:
        llm: The LLM instance

    Returns:
        The inferred provider name, or None if it cannot be determined
    """
    module = type(llm).__module__

    if module.startswith("langchain_"):
        package = module.split(".")[0]
        provider = package.replace("langchain_", "")

        if provider == "community":
            parts = module.split(".")
            if len(parts) >= 3:
                provider = parts[-1]
                log.debug(
                    "Inferred provider '%s' from community module %s", provider, module
                )
                return provider
        else:
            log.debug("Inferred provider '%s' from module %s", provider, module)
            return provider

    log.debug("Could not infer provider from module %s", module)
    return None


def get_llm_provider(llm: BaseLanguageModel) -> Optional[str]:
    """Get the provider name for an LLM instance by inferring from module path.

    This function extracts the provider name from LangChain package naming conventions.
    See _infer_provider_from_module for details on the inference logic.

    Args:
        llm: The LLM instance

    Returns:
        The provider name if it can be inferred, None otherwise
    """
    return _infer_provider_from_module(llm)


def transform_llm_params(
    llm_params: Dict[str, Any],
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    parameter_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Transform LLM parameters using provider-specific or custom mappings.

    Args:
        llm_params: The original parameters dictionary
        provider: Optional provider name
        model_name: Optional model name
        parameter_mapping: Custom mapping dictionary. If None, uses built-in provider mappings.
                          Key is the internal parameter name, value is the provider parameter name.
                          If value is None, the parameter is dropped.

    Returns:
        Transformed parameters dictionary
    """
    if not llm_params:
        return llm_params

    if parameter_mapping is not None:
        return _apply_mapping(llm_params, parameter_mapping)

    has_instance_mapping = (provider, model_name) in _llm_parameter_mappings
    has_builtin_mapping = provider in PROVIDER_PARAMETER_MAPPINGS

    if not has_instance_mapping and not has_builtin_mapping:
        return llm_params

    mapping = None
    if has_instance_mapping:
        mapping = _llm_parameter_mappings.get((provider, model_name))
        log.debug("Using registered parameter mapping for %s/%s", provider, model_name)
    if not mapping and has_builtin_mapping:
        mapping = PROVIDER_PARAMETER_MAPPINGS[provider]
        log.debug("Using built-in parameter mapping for provider: %s", provider)

    return _apply_mapping(llm_params, mapping) if mapping else llm_params


def _apply_mapping(
    llm_params: Dict[str, Any], mapping: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """Apply parameter mapping transformation.

    Args:
        llm_params: The original parameters dictionary
        mapping: The parameter mapping dictionary

    Returns:
        Transformed parameters dictionary
    """
    transformed_params = {}

    for param_name, param_value in llm_params.items():
        if param_name in mapping:
            mapped_name = mapping[param_name]
            if mapped_name is not None:
                transformed_params[mapped_name] = param_value
                log.debug("Mapped parameter %s -> %s", param_name, mapped_name)
            else:
                log.debug("Dropped parameter %s", param_name)
        else:
            transformed_params[param_name] = param_value

    return transformed_params
