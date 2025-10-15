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

"""Rail result extraction strategies.

This module provides different strategies for extracting rail check results
from the underlying LLMRails response, supporting both the new rails_status
API and legacy exception-based detection.
"""

import logging
from typing import Any

from .types import GuardResult, RailType

logger = logging.getLogger(__name__)


def extract_rails_status_results(
    response: Any, rail_type: RailType
) -> list[GuardResult]:
    """Extract rail results from response.rails_status.

    This function extracts guardrail check results using the new rails_status API.

    Args:
        response: Response from LLMRails.generate()
        rail_type: Type of rail (input or output)

    Returns:
        List of GuardResult objects
    """
    if not (hasattr(response, "rails_status") and response.rails_status):
        logger.debug("No rails_status found in response")
        return [
            GuardResult(
                passed=True,
                reason=f"All {rail_type.value} rails passed",
                guard_name=rail_type.value,
            )
        ]

    rails = (
        response.rails_status.input_rails
        if rail_type == RailType.INPUT
        else response.rails_status.output_rails
    )

    if not rails:
        return [
            GuardResult(
                passed=True,
                reason=f"All {rail_type.value} rails passed",
                guard_name=rail_type.value,
            )
        ]

    results = []
    for rail_result in rails:
        results.append(
            GuardResult(
                passed=rail_result.passed,
                reason=rail_result.message or "Rail passed",
                guard_name=rail_result.rail_name,
            )
        )

    return results


class ExceptionBasedExtractor:
    """Extract rail results using legacy exception message detection."""

    def extract_results(self, response: Any, rail_type: RailType) -> list[GuardResult]:
        """Extract results by checking for exception messages.

        Args:
            response: Response from LLMRails.generate()
            rail_type: Type of rail (input or output)

        Returns:
            List of GuardResult objects
        """
        response_list = response.response if hasattr(response, "response") else response

        if isinstance(response_list, list):
            for msg in response_list:
                if isinstance(msg, dict) and msg.get("role") == "exception":
                    return [self._create_failure_result(msg)]

        if isinstance(response_list, dict) and response_list.get("role") == "exception":
            return [self._create_failure_result(response_list)]

        return [
            GuardResult(
                passed=True,
                reason=f"All {rail_type.value} rails passed",
                guard_name=rail_type.value,
            )
        ]

    def _create_failure_result(self, exception_msg: dict) -> GuardResult:
        """Create GuardResult from exception message.

        Args:
            exception_msg: Exception message dict

        Returns:
            GuardResult with failure
        """
        exception_content = exception_msg.get("content", {})
        exception_type = exception_content.get("type", "")
        exception_message = exception_content.get("message", "Rail triggered")

        return GuardResult(
            passed=False,
            reason=exception_message,
            guard_name=exception_type,
        )
