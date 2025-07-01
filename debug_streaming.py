#!/usr/bin/env python3

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

import asyncio

from nemoguardrails.rails.llm.buffer import RollingBuffer


async def fake_streaming_handler():
    """Simulate the FakeLLM streaming behavior"""
    response = (
        "This is a safe and compliant high quality joke that should pass all checks."
    )
    chunks = response.split(" ")
    for i in range(len(chunks)):
        if i < len(chunks) - 1:
            chunk = chunks[i] + " "  # Each chunk except the last includes a space
        else:
            chunk = chunks[i]
        print(f"Original chunk {i}: '{chunk}'")
        yield chunk


async def debug_buffer_strategy():
    """Debug the buffer strategy to see where spaces are lost"""
    print("=== Testing Buffer Strategy ===")

    # Create buffer strategy with same config as tests
    buffer_strategy = RollingBuffer(buffer_context_size=2, buffer_chunk_size=4)
    streaming_handler = fake_streaming_handler()

    chunk_list_results = []
    chunk_str_rep_results = []

    async for chunk_list, chunk_str_rep in buffer_strategy(streaming_handler):
        print(f"\nBuffer yielded:")
        print(f"  chunk_list: {chunk_list}")
        print(f"  chunk_str_rep: '{chunk_str_rep}'")

        chunk_list_results.extend(chunk_list)
        chunk_str_rep_results.append(chunk_str_rep)

    print(f"\n=== Final Results ===")
    print(f"All chunk_list items: {chunk_list_results}")
    print(f"All chunk_str_rep items: {chunk_str_rep_results}")
    print(f"Joined chunk_list: '{''.join(chunk_list_results)}'")
    print(f"Joined chunk_str_rep: '{''.join(chunk_str_rep_results)}'")


if __name__ == "__main__":
    asyncio.run(debug_buffer_strategy())
