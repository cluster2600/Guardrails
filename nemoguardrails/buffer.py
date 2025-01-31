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

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Tuple

from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig, RailsConfig


class BufferStrategy(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: OutputRailsStreamingConfig) -> "BufferStrategy":
        pass

    @abstractmethod
    async def __call__(self, streaming_handler) -> AsyncGenerator:
        pass

    @abstractmethod
    def generate_chunk_str(self, *args, **kwargs) -> str:
        pass


class SlidingWindow(BufferStrategy):
    """DRFAT: A minimal buffer strategy that buffers chunks and yields them when the buffer is full."""

    # - **chunk_size (X)**: This would correspond to the number of tokens in each chunk processed by the `streaming_handler`.
    # - **max_validation_length (N)**: This would correspond to the `look_back_size` parameter in the code, representing the maximum number of lookback chunks.
    #
    # In the code:
    # - `window_size` represents the number of chunks to process in each window.
    # - `look_back_size` represents the number of previous chunks to include in the window for context.

    def __init__(self, look_back_size: int = 5, window_size: int = 10):
        self.look_back_size = look_back_size
        self.window_size = window_size
        self.last_index = 0

    @classmethod
    def from_config(cls, config: OutputRailsStreamingConfig):
        return cls(look_back_size=config.look_back_size, window_size=config.window_size)

    async def __call__(
        self, streaming_handler
    ) -> AsyncGenerator[Tuple[List[str], str], None]:
        buffer = []
        index = 0

        async for chunk in streaming_handler:
            buffer.append(chunk)
            index += 1
            # TODO: this is done in StreamingHandler, we need to find away to remove this duplication
            # print(f"\033[92m{chunk}\033[0m", end="", flush=True)
            # the hackish solution in StreamingHandler is resolved in Chat ClI, we should not alter interfaces
            # when we have stream_async we must use it everywhere, adding enable_print will cause headaches
            # then this hackish solution will cause a cancer of this hackish solution and will contaminate the whole codebase

            if len(buffer) >= self.window_size:
                yield (
                    # buffer is used to apply output rails
                    buffer[-self.window_size - self.look_back_size :],
                    # this is what gets printed in the console or yield to user
                    # to avoid repeating the already streamed/printed chunk
                    self.generate_chunk_str(
                        buffer[-self.window_size - self.look_back_size :], index
                    ),
                )
                buffer = buffer[-self.look_back_size :]

        # Yield any remaining buffer if it's not empty
        if buffer:
            yield (
                buffer,
                self.generate_chunk_str(
                    buffer[-self.window_size - self.look_back_size :], index
                ),
            )

    def generate_chunk_str(self, buffer, current_index) -> str:
        if current_index <= self.last_index:
            return ""

        new_chunks = buffer[self.last_index - current_index :]
        self.last_index = current_index
        # TODO: something causes duplicate whitespaces between tokens, figure out why,
        # If using `return "".join(new_chunks)` works, then the issue might be elsewhere in the code where the chunks are being generated or processed.
        # Ensure that the chunks themselves do not contain extra spaces.
        # WAR: return "".join(new_chunks)
        return "".join(new_chunks)


def get_buffer_strategy(config: OutputRailsStreamingConfig) -> BufferStrategy:
    # TODO: use a factory function or class
    # currently we only have SlidingWindo, in future we have a registry
    return SlidingWindow.from_config(config)


def is_blocked(result: Tuple[bool, str]) -> bool:
    """Check output rials status."""
    # result is a tuple of (return_value, success|failure)
    # return_value is not unified among all the actions
    # sometimes it is a bool where True means allowed, sometimes True means blocked
    # sometimes it is a score where allowed/blocked is dictated in flows.co
    # lack of stable interface for output rails cause this issue
    # for self_check_output following holds
    return not result[0]
