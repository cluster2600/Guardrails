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

from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig


class BufferStrategy(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: OutputRailsStreamingConfig) -> "BufferStrategy":
        pass

    # The abstract method is not async to ensure the return type
    # matches the async generator in the concrete implementation.
    @abstractmethod
    def __call__(
        self, streaming_handler
    ) -> AsyncGenerator[Tuple[List[str], List[str]], None]:
        pass

    @abstractmethod
    def generate_chunk_str(self, *args, **kwargs) -> str:
        pass


class RollingBuffer(BufferStrategy):
    """A minimal buffer strategy that buffers chunks and yields them when the buffer is full.

    Args:
        buffer_context_size (int): The number of tokens carried over from the previous chunk to provide context for continuity in processing.
        buffer_chunk_size (int): The number of tokens in each processing chunk. This is the size of the token block on which output rails are applied.
    """

    def __init__(self, buffer_context_size: int = 5, buffer_chunk_size: int = 10):
        self.buffer_context_size = buffer_context_size
        self.buffer_chunk_size = buffer_chunk_size
        self.total_yielded = 0  # Track total chunks yielded to user

    @classmethod
    def from_config(cls, config: OutputRailsStreamingConfig):
        return cls(
            buffer_context_size=config.context_size, buffer_chunk_size=config.chunk_size
        )

    async def __call__(
        self, streaming_handler
    ) -> AsyncGenerator[Tuple[List[str], List[str]], None]:
        # Reset state for each streaming session
        self.total_yielded = 0
        buffer = []
        total_chunks = 0

        async for chunk in streaming_handler:
            buffer.append(chunk)
            total_chunks += 1

            if len(buffer) >= self.buffer_chunk_size:
                # Calculate how many new chunks should be yielded
                new_chunks_to_yield = min(
                    self.buffer_chunk_size, total_chunks - self.total_yielded
                )

                # Create the processing buffer (includes context)
                processing_buffer = buffer[
                    -self.buffer_chunk_size - self.buffer_context_size :
                ]

                # Get the new chunks to yield to user (preserve original token format)
                if new_chunks_to_yield > 0:
                    # The new chunks are at the end of the buffer
                    chunks_to_yield = buffer[-new_chunks_to_yield:]
                    self.total_yielded += new_chunks_to_yield
                else:
                    chunks_to_yield = []

                yield (processing_buffer, chunks_to_yield)
                buffer = buffer[-self.buffer_context_size :]

        # Yield any remaining buffer if it's not empty
        if buffer:
            new_chunks_to_yield = len(buffer) - max(
                0, self.total_yielded - (total_chunks - len(buffer))
            )
            if new_chunks_to_yield > 0:
                chunks_to_yield = (
                    buffer[-new_chunks_to_yield:]
                    if new_chunks_to_yield <= len(buffer)
                    else buffer
                )
            else:
                chunks_to_yield = []

            yield (buffer, chunks_to_yield)

    def generate_chunk_str(self, buffer, current_index) -> str:
        """Legacy method - logic moved to __call__ method."""
        # This method is kept for compatibility but should not be used
        return ""

    def get_new_chunks(self, buffer, current_index) -> List[str]:
        """Legacy method - logic moved to __call__ method."""
        # This method is kept for compatibility but should not be used
        return []


def get_buffer_strategy(config: OutputRailsStreamingConfig) -> BufferStrategy:
    # TODO: use a factory function or class
    # currently we only have RollingBuffer, in future we use a registry
    return RollingBuffer.from_config(config)
