# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fake LLM providers for deterministic benchmarking.

These providers introduce controlled, configurable latency without
touching any real model endpoint.  They are used to isolate
orchestration cost from provider I/O in benchmark results.
"""

import asyncio
import json
import time
from typing import AsyncIterator


class FakeProvider:
    """Async provider with configurable latency."""

    def __init__(self, latency_ms: int = 10, response: str | None = None):
        self.latency_ms = latency_ms
        self.response = response or json.dumps({"answer": "safe response"})
        self._call_count = 0

    async def generate(self, prompt: str) -> str:
        self._call_count += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        return self.response

    @property
    def call_count(self) -> int:
        return self._call_count


class FakeStreamingProvider:
    """Async streaming provider that yields tokens with configurable delay."""

    def __init__(
        self,
        latency_ms: int = 10,
        tokens: list[str] | None = None,
        inter_token_ms: int = 5,
    ):
        self.latency_ms = latency_ms
        self.tokens = tokens or [
            "The ",
            "capital ",
            "of ",
            "France ",
            "is ",
            "Paris.",
            " ",
            "It ",
            "is ",
            "known ",
            "for ",
            "the ",
            "Eiffel ",
            "Tower.",
        ]
        self.inter_token_ms = inter_token_ms
        self._call_count = 0

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        self._call_count += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        for token in self.tokens:
            yield token
            if self.inter_token_ms > 0:
                await asyncio.sleep(self.inter_token_ms / 1000)

    @property
    def call_count(self) -> int:
        return self._call_count


class FakeSyncProvider:
    """Sync provider for thread-based benchmarks."""

    def __init__(self, latency_ms: int = 10, response: str | None = None):
        self.latency_ms = latency_ms
        self.response = response or json.dumps({"answer": "safe response"})
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        self._call_count += 1
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
        return self.response

    @property
    def call_count(self) -> int:
        return self._call_count
