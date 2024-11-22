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

import inspect

from guardrails import Guard

from nemoguardrails.rails.llm.llmrails import LLMRails

try:
    from guardrails import Guard
except ImportError:
    raise ImportError(
        "Could not import guardrails-ai, please install it with "
        "`pip install guardrails-ai`."
    )


def register_guardrails_guard_actions(rails: LLMRails, guard: Guard, guard_name: str):
    async def fix_action(text, metadata={}):
        response = guard.validate(llm_output=text, metadata=metadata)
        if inspect.iscoroutine(response):
            response = await response

        return (
            response.validated_output if response.validation_passed is True else False
        )

    async def validate_action(text, metadata={}):
        response = guard.validate(llm_output=text, metadata=metadata)
        if inspect.iscoroutine(response):
            response = await response

        if response.validation_passed is True:
            return response.validated_output == response.raw_llm_output
        return response.validation_passed

    rails.register_action(fix_action, f"{guard_name}_fix")
    rails.register_action(validate_action, f"{guard_name}_validate")
