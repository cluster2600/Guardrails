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

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import logging
from time import time
from typing import Dict, Optional

# The @action() decorator stamps an `action_meta` dict onto the decorated
# callable.  The ActionDispatcher uses that attribute to discover, register,
# and route guardrail actions by their canonical name.  See actions.py in
# nemoguardrails/actions for the full implementation.
from nemoguardrails.actions import action

# Context variable holding per-request LLM call metadata (used for tracing).
from nemoguardrails.context import llm_call_info_var

# Thin HTTP helpers that forward detection work to a remote microservice.
from nemoguardrails.library.jailbreak_detection.request import (
    jailbreak_detection_heuristics_request,
    jailbreak_detection_model_request,
    jailbreak_nim_request,
)

# Cache abstraction — pluggable back-ends (in-memory, Redis, etc.) that
# store jailbreak verdicts so repeated identical prompts skip inference.
from nemoguardrails.llm.cache import CacheInterface
from nemoguardrails.llm.cache.utils import (
    CacheEntry,
    create_normalized_cache_key,
    get_from_cache_and_restore_stats,
)
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Heuristic-based jailbreak detection action
# --------------------------------------------------------------------------- #
# The @action() decorator (with no arguments) registers this coroutine under
# its own function name — "jailbreak_detection_heuristics" — in the
# ActionDispatcher.  Internally it attaches an `action_meta` dict that the
# dispatcher inspects at registration and routing time.
@action()
async def jailbreak_detection_heuristics(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Checks the user's prompt to determine if it is attempt to jailbreak the model."""
    # Pull detection thresholds from the guardrails YAML configuration.
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection

    jailbreak_api_url = jailbreak_config.server_endpoint
    # length-per-perplexity threshold — a high ratio signals an adversarial
    # prompt that is long yet gibberish (low perplexity denominator).
    lp_threshold = jailbreak_config.length_per_perplexity_threshold
    # prefix/suffix perplexity threshold — catches GCG-style attacks where
    # random token strings are appended or prepended to a benign prompt.
    ps_ppl_threshold = jailbreak_config.prefix_suffix_perplexity_threshold

    prompt = context.get("user_message")

    # ---- In-process fallback path (no remote endpoint configured) ----
    if not jailbreak_api_url:
        # Lazy import: the heuristics module pulls in PyTorch and GPT-2,
        # which are heavyweight.  Deferring the import avoids paying that
        # cost when a remote endpoint handles detection instead.
        from nemoguardrails.library.jailbreak_detection.heuristics.checks import (
            check_jailbreak_length_per_perplexity,
            check_jailbreak_prefix_suffix_perplexity,
        )

        log.warning("No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION.")
        # The @cpu_bound decorator (from thread_pool.py) stamps a
        # `_cpu_bound = True` flag on synchronous functions.  When the
        # flag is present we offload the call to a thread-pool executor
        # so that the long-running GPT-2 perplexity computation does
        # not block the async event loop.
        if getattr(check_jailbreak_length_per_perplexity, "_cpu_bound", False):
            loop = asyncio.get_running_loop()
            lp_check = await loop.run_in_executor(
                None, functools.partial(check_jailbreak_length_per_perplexity, prompt, lp_threshold)
            )
            ps_ppl_check = await loop.run_in_executor(
                None, functools.partial(check_jailbreak_prefix_suffix_perplexity, prompt, ps_ppl_threshold)
            )
        else:
            # Fallback: if @cpu_bound was unavailable at import time
            # (e.g. missing dependency), the functions run synchronously
            # on the current thread — acceptable for testing only.
            lp_check = check_jailbreak_length_per_perplexity(prompt, lp_threshold)
            ps_ppl_check = check_jailbreak_prefix_suffix_perplexity(prompt, ps_ppl_threshold)
        # A prompt is flagged if *either* heuristic triggers.
        jailbreak = any([lp_check["jailbreak"], ps_ppl_check["jailbreak"]])
        return jailbreak

    # ---- Remote endpoint path (preferred for production) ----
    jailbreak = await jailbreak_detection_heuristics_request(prompt, jailbreak_api_url, lp_threshold, ps_ppl_threshold)
    if jailbreak is None:
        log.warning("Jailbreak endpoint not set up properly.")
        # Fail-open: if the endpoint is misconfigured, assume the prompt
        # is benign rather than blocking every request.
        return False
    else:
        return jailbreak


# --------------------------------------------------------------------------- #
#  Model-based jailbreak detection action
# --------------------------------------------------------------------------- #
# Registered via the same @action() pattern.  This variant uses a trained
# binary classifier (local or remote) rather than perplexity heuristics.
@action()
async def jailbreak_detection_model(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    # model_caches is a dict mapping rail names to CacheInterface instances.
    # The "jailbreak_detection" key holds a pluggable cache (e.g. in-memory
    # or Redis) that maps normalised prompts to previously computed verdicts.
    model_caches: Optional[Dict[str, CacheInterface]] = None,
) -> bool:
    """Uses a trained classifier to determine if a user input is a jailbreak attempt"""
    prompt: str = ""
    jailbreak_config = llm_task_manager.config.rails.config.jailbreak_detection

    # Three possible back-ends, checked in priority order below:
    #   1. NVIDIA NIM inference micro-service  (nim_base_url)
    #   2. Custom REST endpoint                (jailbreak_api_url)
    #   3. Local in-process model              (fallback)
    jailbreak_api_url = jailbreak_config.server_endpoint
    nim_base_url = jailbreak_config.nim_base_url
    nim_classification_path = jailbreak_config.nim_server_endpoint
    nim_auth_token = jailbreak_config.get_api_key()

    if context is not None:
        prompt = context.get("user_message", "")

    # Inject an LLMCallInfo so the tracing / explain subsystem treats this
    # action as if it were an LLM call, even though it may be a classifier.
    llm_call_info_var.set(LLMCallInfo(task="jailbreak_detection_model"))

    # ----- Cache lookup ----- #
    # The caching strategy normalises the prompt text into a deterministic
    # key and stores the boolean verdict together with optional LLM metadata.
    # On a hit, we return immediately, avoiding the cost of model inference
    # or an HTTP round-trip to the detection endpoint.
    cache = model_caches.get("jailbreak_detection") if model_caches else None

    if cache:
        cache_key = create_normalized_cache_key(prompt)  # deterministic hash of the prompt
        cache_read_start = time()
        cached_result = get_from_cache_and_restore_stats(cache, cache_key)
        if cached_result is not None:
            # Populate tracing metadata so cached results are still visible
            # in the explain / processing log output.
            cache_read_duration = time() - cache_read_start
            llm_call_info = llm_call_info_var.get()
            if llm_call_info:
                llm_call_info.from_cache = True
                llm_call_info.duration = cache_read_duration
                llm_call_info.started_at = time() - cache_read_duration
                llm_call_info.finished_at = time()

            log.debug("Jailbreak detection cache hit")
            return cached_result["jailbreak"]  # bool — True means jailbreak detected

    jailbreak_result = None
    api_start_time = time()  # used to measure wall-clock duration for tracing

    # ---- In-process fallback (no remote endpoint) ---- #
    if not jailbreak_api_url and not nim_base_url:
        # Lazy import keeps heavyweight dependencies (torch, sklearn) out of
        # the critical path when a remote endpoint is configured.
        from nemoguardrails.library.jailbreak_detection.model_based.checks import (
            check_jailbreak,
        )

        log.warning("No jailbreak detection endpoint set. Running in-process, NOT RECOMMENDED FOR PRODUCTION.")
        try:
            # Same @cpu_bound / fallback pattern as the heuristics action:
            # offload to the thread-pool if the decorator is available,
            # otherwise run synchronously (blocks the event loop — dev only).
            if getattr(check_jailbreak, "_cpu_bound", False):
                loop = asyncio.get_running_loop()
                jailbreak = await loop.run_in_executor(None, functools.partial(check_jailbreak, prompt=prompt))
            else:
                jailbreak = check_jailbreak(prompt=prompt)
            log.info(f"Local model jailbreak detection result: {jailbreak}")
            jailbreak_result = jailbreak["jailbreak"]
        except RuntimeError as e:
            # Model weights not found or device unavailable — fail-open.
            log.error(f"Jailbreak detection model not available: {e}")
            jailbreak_result = False
        except ImportError as e:
            # scikit-learn or torch missing from the environment.
            log.error(
                "Failed to import required dependencies for local model. Install scikit-learn and torch, or use NIM-based approach",
                exc_info=e,
            )
            jailbreak_result = False
    else:
        # ---- Remote endpoint paths ---- #
        if nim_base_url:
            # Preferred: NVIDIA NIM inference micro-service.
            jailbreak = await jailbreak_nim_request(
                prompt=prompt,
                nim_url=nim_base_url,
                nim_auth_token=nim_auth_token,
                nim_classification_path=nim_classification_path,
            )
        elif jailbreak_api_url:
            # Fallback: generic REST endpoint for model-based detection.
            jailbreak = await jailbreak_detection_model_request(prompt=prompt, api_url=jailbreak_api_url)

        if jailbreak is None:
            log.warning("Jailbreak endpoint not set up properly.")
            jailbreak_result = False  # fail-open on endpoint misconfiguration
        else:
            jailbreak_result = jailbreak

    api_duration = time() - api_start_time

    # ----- Populate tracing metadata for the processing log ----- #
    llm_call_info = llm_call_info_var.get()
    if llm_call_info:
        llm_call_info.from_cache = False  # this was a live call, not cached
        llm_call_info.duration = api_duration
        llm_call_info.started_at = api_start_time
        llm_call_info.finished_at = time()

        # Append to the per-request processing log so the explain endpoint
        # can surface timing and provenance for this detection step.
        processing_log = processing_log_var.get()
        if processing_log is not None:
            processing_log.append(
                {
                    "type": "llm_call_info",
                    "timestamp": time(),
                    "data": llm_call_info,
                }
            )

    # ----- Write-through cache update ----- #
    # After a live inference call, persist the verdict so that subsequent
    # identical prompts can be served from the cache (see lookup above).
    if cache:
        from nemoguardrails.llm.cache.utils import extract_llm_metadata_for_cache

        cache_key = create_normalized_cache_key(prompt)
        cache_entry: CacheEntry = {
            "result": {"jailbreak": jailbreak_result},  # the boolean verdict
            "llm_stats": None,  # no token-level stats for a classifier
            "llm_metadata": extract_llm_metadata_for_cache(),
        }
        cache.put(cache_key, cache_entry)
        log.debug("Jailbreak detection result cached")

    return jailbreak_result
