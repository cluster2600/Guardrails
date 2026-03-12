# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Comprehensive tests for the streaming rail check infrastructure.

Covers three stories:
GARDR-23 -- Chunk-boundary dispatcher tests (RollingBuffer logic)
GARDR-24 -- stream_first mode / OutputRailsStreamingConfig tests
GARDR-25 -- Sliding-window context retention tests
"""

from __future__ import annotations

import json
from typing import AsyncIterator, List

import pytest

from nemoguardrails.rails.llm.buffer import (
    BufferStrategy,
    ChunkBatch,
    RollingBuffer,
    get_buffer_strategy,
)
from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_iter(tokens: List[str]) -> AsyncIterator[str]:
    for t in tokens:
        yield t


async def _async_iter_dicts(tokens: List[str]) -> AsyncIterator[dict]:
    for t in tokens:
        yield {"text": t}


async def _collect_batches(buffer: RollingBuffer, tokens: List[str], as_dicts: bool = False) -> List[ChunkBatch]:
    stream = _async_iter_dicts(tokens) if as_dicts else _async_iter(tokens)
    batches: List[ChunkBatch] = []
    async for batch in buffer.process_stream(stream):
        batches.append(batch)
    return batches


# ===========================================================================
#  GARDR-23 -- Chunk-boundary dispatcher tests
# ===========================================================================


class TestGARDR23ChunkBoundaryDispatcher:
    @pytest.mark.asyncio
    async def test_yields_at_chunk_size_threshold(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].user_output_chunks == ["a", "b", "c", "d", "e"]
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_remaining_chunks_yielded_at_end_of_stream(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=4)
        tokens = ["a", "b", "c", "d", "e"]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].user_output_chunks == ["a", "b", "c", "d"]
        assert batches[-1].user_output_chunks == ["e"]

    @pytest.mark.asyncio
    async def test_processing_context_includes_previous_context(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].processing_context == ["a", "b", "c", "d", "e"]
        assert "d" in batches[1].processing_context
        assert "e" in batches[1].processing_context

    @pytest.mark.asyncio
    async def test_user_output_chunks_only_new_tokens(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=4)
        tokens = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        batches = await _collect_batches(buf, tokens)
        all_user = []
        for b in batches:
            all_user.extend(b.user_output_chunks)
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_user_output_no_duplicates_with_context(self):
        buf = RollingBuffer(buffer_context_size=3, buffer_chunk_size=5)
        tokens = [f"w{i}" for i in range(12)]
        batches = await _collect_batches(buf, tokens)
        all_user = []
        for b in batches:
            all_user.extend(b.user_output_chunks)
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        batches = await _collect_batches(buf, [])
        assert batches == []

    @pytest.mark.asyncio
    async def test_single_token(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        batches = await _collect_batches(buf, ["only"])
        assert len(batches) == 1
        assert batches[0].user_output_chunks == ["only"]

    @pytest.mark.asyncio
    async def test_exact_chunk_size_boundary(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=4)
        tokens = ["a", "b", "c", "d"]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].user_output_chunks == ["a", "b", "c", "d"]
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_two_tokens_chunk_size_three(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)
        batches = await _collect_batches(buf, ["hello", " world"])
        assert len(batches) == 1
        assert batches[0].user_output_chunks == ["hello", " world"]

    def test_format_chunks_joins_tokens(self):
        buf = RollingBuffer()
        assert buf.format_chunks(["Hello", " ", "world", "!"]) == "Hello world!"

    def test_format_chunks_empty_list(self):
        assert RollingBuffer().format_chunks([]) == ""

    def test_format_chunks_single_token(self):
        assert RollingBuffer().format_chunks(["solo"]) == "solo"

    def test_format_chunks_preserves_whitespace(self):
        assert RollingBuffer().format_chunks(["  padded  ", " end"]) == "  padded   end"

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="buffer_chunk_size must be non-negative"):
            RollingBuffer(buffer_context_size=0, buffer_chunk_size=-1)

    def test_negative_context_size_raises(self):
        with pytest.raises(ValueError, match="buffer_context_size must be non-negative"):
            RollingBuffer(buffer_context_size=-5, buffer_chunk_size=10)

    def test_both_negative_raises(self):
        with pytest.raises(ValueError):
            RollingBuffer(buffer_context_size=-1, buffer_chunk_size=-1)

    @pytest.mark.asyncio
    async def test_dict_wrapped_tokens(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)
        tokens = ["Hello", " ", "world"]
        batches = await _collect_batches(buf, tokens, as_dicts=True)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_dict_wrapped_longer_stream(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)
        tokens = ["The ", "quick ", "brown ", "fox ", "jumps"]
        batches = await _collect_batches(buf, tokens, as_dicts=True)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_total_yielded_accurate(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=4)
        tokens = [f"t{i}" for i in range(10)]
        await _collect_batches(buf, tokens)
        assert buf.total_yielded == len(tokens)

    @pytest.mark.asyncio
    async def test_total_yielded_resets_between_sessions(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=4)
        await _collect_batches(buf, ["a", "b", "c", "d"])
        assert buf.total_yielded == 4
        # Session 2 resets; ctx=1/chunk=4, 8 tokens -> 4+3=7 via threshold
        await _collect_batches(buf, [f"t{i}" for i in range(8)])
        assert buf.total_yielded == 7

    def test_chunk_batch_is_named_tuple(self):
        batch = ChunkBatch(processing_context=["a"], user_output_chunks=["a"])
        assert batch.processing_context == ["a"]
        assert batch[0] == ["a"]
        assert batch[1] == ["a"]

    def test_chunk_batch_unpacking(self):
        ctx, user = ChunkBatch(processing_context=["x"], user_output_chunks=["y"])
        assert ctx == ["x"]
        assert user == ["y"]

    @pytest.mark.asyncio
    async def test_callable_interface(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)
        tokens = ["a", "b", "c", "d"]
        via_call = []
        async for batch in buf(_async_iter(tokens)):
            via_call.append(batch)

        buf2 = RollingBuffer(buffer_context_size=1, buffer_chunk_size=3)
        via_method = []
        async for batch in buf2.process_stream(_async_iter(tokens)):
            via_method.append(batch)

        assert len(via_call) == len(via_method)
        for a, b in zip(via_call, via_method):
            assert a.user_output_chunks == b.user_output_chunks


# ===========================================================================
#  GARDR-24 -- stream_first mode / OutputRailsStreamingConfig tests
# ===========================================================================


class TestGARDR24StreamFirstConfig:
    def test_default_enabled_is_false(self):
        assert OutputRailsStreamingConfig().enabled is False

    def test_default_chunk_size(self):
        assert OutputRailsStreamingConfig().chunk_size == 200

    def test_default_context_size(self):
        assert OutputRailsStreamingConfig().context_size == 50

    def test_default_stream_first_is_true(self):
        assert OutputRailsStreamingConfig().stream_first is True

    def test_all_defaults_at_once(self):
        cfg = OutputRailsStreamingConfig()
        assert cfg.enabled is False
        assert cfg.chunk_size == 200
        assert cfg.context_size == 50
        assert cfg.stream_first is True

    def test_custom_enabled(self):
        assert OutputRailsStreamingConfig(enabled=True).enabled is True

    def test_custom_chunk_size(self):
        assert OutputRailsStreamingConfig(chunk_size=50).chunk_size == 50

    def test_custom_context_size(self):
        assert OutputRailsStreamingConfig(context_size=10).context_size == 10

    def test_custom_stream_first_false(self):
        assert OutputRailsStreamingConfig(stream_first=False).stream_first is False

    def test_all_custom_values(self):
        cfg = OutputRailsStreamingConfig(
            enabled=True,
            chunk_size=100,
            context_size=25,
            stream_first=False,
        )
        assert cfg.enabled is True
        assert cfg.chunk_size == 100
        assert cfg.context_size == 25
        assert cfg.stream_first is False

    def test_extra_fields_allowed(self):
        cfg = OutputRailsStreamingConfig(
            enabled=True,
            custom_param="hello",
            another_extra=42,
        )
        assert cfg.custom_param == "hello"
        assert cfg.another_extra == 42

    def test_extra_fields_do_not_affect_defaults(self):
        cfg = OutputRailsStreamingConfig(my_extra="value")
        assert cfg.chunk_size == 200
        assert cfg.context_size == 50

    def test_dict_round_trip(self):
        original = OutputRailsStreamingConfig(enabled=True, chunk_size=64, context_size=8, stream_first=False)
        restored = OutputRailsStreamingConfig(**original.model_dump())
        assert restored.enabled == original.enabled
        assert restored.chunk_size == original.chunk_size

    def test_json_round_trip(self):
        original = OutputRailsStreamingConfig(enabled=True, chunk_size=128, context_size=16, stream_first=True)
        restored = OutputRailsStreamingConfig(**json.loads(original.model_dump_json()))
        assert restored.enabled == original.enabled
        assert restored.chunk_size == original.chunk_size

    def test_dict_round_trip_with_extras(self):
        original = OutputRailsStreamingConfig(enabled=True, bonus_field="keep_me")
        as_dict = original.model_dump()
        assert "bonus_field" in as_dict
        restored = OutputRailsStreamingConfig(**as_dict)
        assert restored.bonus_field == "keep_me"

    def test_json_round_trip_preserves_types(self):
        data = json.loads(
            OutputRailsStreamingConfig(
                enabled=False, chunk_size=999, context_size=0, stream_first=True
            ).model_dump_json()
        )
        assert isinstance(data["enabled"], bool)
        assert isinstance(data["chunk_size"], int)

    def test_model_dump_contains_all_fields(self):
        d = OutputRailsStreamingConfig().model_dump()
        for field in ("enabled", "chunk_size", "context_size", "stream_first"):
            assert field in d


# ===========================================================================
#  GARDR-25 -- Sliding-window context tests
# ===========================================================================


class TestGARDR25SlidingWindowContext:
    @pytest.mark.asyncio
    async def test_context_tokens_retained_between_chunks(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        tokens = [f"t{i}" for i in range(8)]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].processing_context == ["t0", "t1", "t2", "t3", "t4"]
        assert "t3" in batches[1].processing_context
        assert "t4" in batches[1].processing_context

    @pytest.mark.asyncio
    async def test_context_size_one_minimal_retention(self):
        buf = RollingBuffer(buffer_context_size=1, buffer_chunk_size=4)
        tokens = ["a", "b", "c", "d", "e", "f", "g"]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].user_output_chunks == ["a", "b", "c", "d"]
        assert batches[1].processing_context[0] == "d"
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_context_size_larger_than_chunk_size(self):
        buf = RollingBuffer(buffer_context_size=5, buffer_chunk_size=3)
        tokens = [f"w{i}" for i in range(9)]
        batches = await _collect_batches(buf, tokens)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens
        assert len(batches[1].processing_context) > len(batches[1].user_output_chunks)

    @pytest.mark.asyncio
    async def test_very_large_context_size(self):
        buf = RollingBuffer(buffer_context_size=100, buffer_chunk_size=3)
        tokens = ["a", "b", "c", "d", "e"]
        batches = await _collect_batches(buf, tokens)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_context_continuity_multiple_chunks(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        tokens = [f"t{i}" for i in range(11)]
        batches = await _collect_batches(buf, tokens)
        assert batches[0].processing_context == ["t0", "t1", "t2", "t3", "t4"]
        assert batches[1].processing_context[0] == "t3"
        assert batches[1].processing_context[1] == "t4"
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens

    @pytest.mark.asyncio
    async def test_context_continuity_with_remainder(self):
        buf = RollingBuffer(buffer_context_size=2, buffer_chunk_size=5)
        tokens = [f"t{i}" for i in range(12)]
        batches = await _collect_batches(buf, tokens)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens
        for i in range(1, len(batches)):
            if not batches[i - 1].user_output_chunks:
                continue
            prev_user = batches[i - 1].user_output_chunks
            assert prev_user[-1] in batches[i].processing_context

    @pytest.mark.asyncio
    async def test_context_overlap_content(self):
        buf = RollingBuffer(buffer_context_size=3, buffer_chunk_size=6)
        tokens = [f"w{i}" for i in range(15)]
        batches = await _collect_batches(buf, tokens)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert all_user == tokens
        for i in range(1, len(batches)):
            if batches[i].user_output_chunks:
                assert len(batches[i].processing_context) >= len(batches[i].user_output_chunks)

    def test_get_buffer_strategy_returns_rolling_buffer(self):
        cfg = OutputRailsStreamingConfig(chunk_size=10, context_size=3)
        strategy = get_buffer_strategy(cfg)
        assert isinstance(strategy, RollingBuffer)
        assert isinstance(strategy, BufferStrategy)

    def test_get_buffer_strategy_applies_config(self):
        cfg = OutputRailsStreamingConfig(chunk_size=42, context_size=7)
        strategy = get_buffer_strategy(cfg)
        assert strategy.buffer_chunk_size == 42
        assert strategy.buffer_context_size == 7

    def test_get_buffer_strategy_with_defaults(self):
        strategy = get_buffer_strategy(OutputRailsStreamingConfig())
        assert strategy.buffer_chunk_size == 200
        assert strategy.buffer_context_size == 50

    def test_from_config_creates_instance(self):
        cfg = OutputRailsStreamingConfig(chunk_size=20, context_size=5)
        buf = RollingBuffer.from_config(cfg)
        assert buf.buffer_chunk_size == 20
        assert buf.buffer_context_size == 5

    def test_from_config_with_defaults(self):
        buf = RollingBuffer.from_config(OutputRailsStreamingConfig())
        assert buf.buffer_chunk_size == 200
        assert buf.buffer_context_size == 50

    def test_from_config_small_values(self):
        buf = RollingBuffer.from_config(OutputRailsStreamingConfig(chunk_size=1, context_size=0))
        assert buf.buffer_chunk_size == 1
        assert buf.buffer_context_size == 0

    def test_from_config_large_values(self):
        buf = RollingBuffer.from_config(OutputRailsStreamingConfig(chunk_size=10000, context_size=5000))
        assert buf.buffer_chunk_size == 10000
        assert buf.buffer_context_size == 5000

    @pytest.mark.asyncio
    async def test_full_text_reconstruction_with_context(self):
        for ctx in (1, 3, 10, 50):
            buf = RollingBuffer(buffer_context_size=ctx, buffer_chunk_size=4)
            tokens = ["The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog."]
            batches = await _collect_batches(buf, tokens)
            reconstructed = [t for b in batches for t in b.user_output_chunks]
            assert reconstructed == tokens, f"Failed with context_size={ctx}"

    @pytest.mark.asyncio
    async def test_format_chunks_end_to_end(self):
        buf = RollingBuffer(buffer_context_size=3, buffer_chunk_size=5)
        tokens = ["Hello", " ", "world", ",", " ", "how", " ", "are", " ", "you", "?"]
        original_text = "".join(tokens)
        batches = await _collect_batches(buf, tokens)
        all_user = [t for b in batches for t in b.user_output_chunks]
        assert buf.format_chunks(all_user) == original_text
