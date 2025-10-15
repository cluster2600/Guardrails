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

"""Input parameter extractors for composable message extraction.

This module provides reusable extractors that can be composed to handle
different input parameter formats across LLM provider APIs.
"""

from typing import Any, Callable, List, Optional

from ...types import RailsMessage


class ParameterExtractor:
    """Extracts messages from a specific parameter name.

    This is a composable extractor that checks for a parameter name
    and applies a converter function if found.

    Example:
        >>> extractor = ParameterExtractor("messages", messages_converter)
        >>> result = extractor("create", messages=[...])
        [{"role": "user", "content": "Hello"}]
    """

    def __init__(
        self,
        param_name: str,
        converter: Callable[[Any], Optional[List[RailsMessage]]],
    ):
        """Initialize parameter extractor.

        Args:
            param_name: Name of parameter to extract (e.g., "messages", "input")
            converter: Function to convert parameter value to RailsMessage list
        """
        self.param_name = param_name
        self.converter = converter

    def __call__(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Optional[List[RailsMessage]]:
        """Try to extract messages from parameter.

        Args:
            method_name: Method being called
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of RailsMessage if parameter found and converted, None otherwise
        """
        if self.param_name in kwargs:
            return self.converter(kwargs[self.param_name])
        return None
