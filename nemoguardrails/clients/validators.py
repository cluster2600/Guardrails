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

"""Message validation utilities.

This module provides validators for RailsMessage structures to catch
errors early and provide clear error messages.
"""

import logging
from typing import List

from .types import RailsMessage, ToolCall

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


class MessageValidator:
    """Validates RailsMessage structure."""

    VALID_ROLES = {"user", "assistant", "system", "tool"}

    def validate(self, message: RailsMessage) -> None:
        """Validate a RailsMessage.

        Args:
            message: Message to validate

        Raises:
            ValidationError: If message is invalid
        """
        if "role" not in message:
            raise ValidationError("Message missing 'role' field")

        role = message["role"]
        if role not in self.VALID_ROLES:
            raise ValidationError(
                f"Invalid role: {role}. Valid roles: {self.VALID_ROLES}"
            )

        if "content" not in message:
            raise ValidationError(f"Message with role '{role}' missing 'content' field")

        if role == "assistant" and "tool_calls" in message:
            self._validate_tool_calls(message["tool_calls"])

        if role == "tool" and "tool_call_id" not in message:
            raise ValidationError("Tool message missing 'tool_call_id' field")

    def validate_messages(self, messages: List[RailsMessage]) -> None:
        """Validate a list of messages.

        Args:
            messages: List of messages to validate

        Raises:
            ValidationError: If any message is invalid
        """
        for i, message in enumerate(messages):
            try:
                self.validate(message)
            except ValidationError as e:
                raise ValidationError(f"Message at index {i}: {e}") from e

    def _validate_tool_calls(self, tool_calls: List[ToolCall]) -> None:
        """Validate tool calls structure.

        Args:
            tool_calls: List of tool calls to validate

        Raises:
            ValidationError: If tool calls are invalid
        """
        if not isinstance(tool_calls, list):
            raise ValidationError("tool_calls must be a list")

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                raise ValidationError(f"Tool call at index {i} must be a dict")

            if not tc.get("name"):
                raise ValidationError(f"Tool call at index {i} missing 'name' field")

            if "args" not in tc:
                raise ValidationError(f"Tool call at index {i} missing 'args' field")

            if not tc.get("id"):
                raise ValidationError(f"Tool call at index {i} missing 'id' field")

            if tc.get("type") != "tool_call":
                raise ValidationError(
                    f"Tool call at index {i} has invalid type: {tc.get('type')}"
                )
