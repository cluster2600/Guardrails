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

import os

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --------------------------------------------------------------------------- #
#  @cpu_bound decorator with graceful fallback
# --------------------------------------------------------------------------- #
# The `cpu_bound` decorator (defined in nemoguardrails.rails.llm.thread_pool)
# stamps a `_cpu_bound = True` attribute onto the decorated synchronous
# function.  The action dispatcher and the calling code in actions.py inspect
# that flag: when it is present the function is dispatched to a thread-pool
# executor via `loop.run_in_executor()`, preventing long-running perplexity
# computations from blocking the async event loop.
#
# If the import fails (e.g. the thread_pool module or its dependencies are
# absent), we fall back to a no-op identity decorator so that every function
# decorated with @cpu_bound still works — it will simply run synchronously on
# the calling thread.  The actions.py caller detects this situation by
# checking `getattr(fn, "_cpu_bound", False)` and adjusts its behaviour
# accordingly.
try:
    from nemoguardrails.rails.llm.thread_pool import cpu_bound
except ImportError:

    def cpu_bound(fn):
        # Identity fallback — no thread-pool offloading available.
        return fn


# --------------------------------------------------------------------------- #
#  Module-level model initialisation (singleton / caching strategy)
# --------------------------------------------------------------------------- #
# The GPT-2 model and tokeniser are loaded once at import time and kept as
# module-level globals.  Every subsequent call to `get_perplexity` re-uses the
# same objects, avoiding repeated disk I/O and GPU memory allocation.  This is
# the primary caching strategy for the perplexity computation — the model
# weights live in memory for the lifetime of the process.
#
# The device can be overridden via the JAILBREAK_CHECK_DEVICE environment
# variable (e.g. "cuda:0"), defaulting to CPU.
device = os.environ.get("JAILBREAK_CHECK_DEVICE", "cpu")
model_id = "gpt2-large"  # ~774 M parameters — large enough for reliable perplexity
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)  # weights cached by HuggingFace Hub
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)  # fast Rust-backed tokeniser


# --------------------------------------------------------------------------- #
#  Sliding-window perplexity computation
# --------------------------------------------------------------------------- #
# Perplexity measures how "surprised" the language model is by the input.
# Legitimate natural-language prompts yield moderate perplexity, whereas
# adversarial / gibberish token sequences produce extreme values.  This
# metric underpins both heuristic checks below.
@cpu_bound  # offloaded to the thread-pool when available (see fallback above)
def get_perplexity(input_string: str) -> bool:
    """
    Function to compute sliding window perplexity of `input_string`

    Args
        input_string: The prompt to be sent to the model
    """
    encodings = tokenizer(input_string, return_tensors="pt")

    max_length = model.config.n_positions  # GPT-2 context window (1024 tokens)
    stride = 512  # overlap between successive windows — balances accuracy and speed
    seq_len = encodings.input_ids.size(1)  # total number of tokens in the input

    nlls = list()  # accumulates negative log-likelihoods per window
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Mask tokens outside the current target span with -100 so they do
        # not contribute to the cross-entropy loss.
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():  # inference only — no gradient computation needed
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss  # mean cross-entropy for this window

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break  # we have covered the entire input

    # Perplexity = exp(mean NLL).  Lower values indicate the model finds the
    # text unsurprising; very high values indicate gibberish or adversarial tokens.
    perplexity = torch.exp(torch.stack(nlls).mean())

    # Detach from the computation graph and return a plain Python float.
    return perplexity.cpu().detach().numpy().item()


# --------------------------------------------------------------------------- #
#  Heuristic 1 — Length-per-perplexity check
# --------------------------------------------------------------------------- #
# Rationale: many jailbreak prompts are unusually long yet linguistically
# incoherent (low perplexity denominator).  Dividing character length by
# perplexity yields a ratio that is disproportionately large for adversarial
# prompts compared to normal user queries.
@cpu_bound  # thread-pool dispatch when the decorator is available
def check_jailbreak_length_per_perplexity(input_string: str, threshold: float) -> dict:
    """
    Check whether the input string has length/perplexity greater than the threshold.

    Args
        input_string: The prompt to be sent to the model
        lp_threshold: Threshold for determining whether `input_string` is a jailbreak (Default: 89.79)
    """
    perplexity = get_perplexity(input_string)
    # score = character count / model perplexity.  A high value suggests the
    # prompt is long yet gibberish — characteristic of adversarial payloads.
    score = len(input_string) / perplexity
    result = {"jailbreak": score >= threshold}  # True when score exceeds the configured threshold
    return result


# --------------------------------------------------------------------------- #
#  Heuristic 2 — Prefix / suffix perplexity check (GCG attack detection)
# --------------------------------------------------------------------------- #
# Rationale: GCG (Greedy Coordinate Gradient) attacks append or prepend a
# sequence of optimised adversarial tokens to a benign-looking prompt.  Those
# token sequences are near-random and produce very high perplexity when
# evaluated in isolation.  This check extracts the first and last 19 words of
# the prompt and flags the input if either fragment exceeds the threshold.
#
# Note: There is no YARA rule integration in this module.  Detection relies
# entirely on statistical perplexity thresholds rather than pattern-matching
# rule sets.
@cpu_bound  # thread-pool dispatch when the decorator is available
def check_jailbreak_prefix_suffix_perplexity(input_string: str, threshold: float) -> dict:
    """
    Check whether the input string has prefix or suffix perplexity greater than the threshold.

    Args
        input_string: The prompt to be sent to the model
        ps_ppl_threshold: Threshold for determining whether `input_string` is a jailbreak (Default: 1845.65)
    """
    split_string = input_string.strip().split()
    # Short prompts (fewer than 20 whitespace-delimited tokens) cannot
    # meaningfully carry a GCG-style adversarial suffix, so we skip them.
    if len(split_string) < 20:
        return {"jailbreak": False}

    # Extract the last 19 words (suffix) and first 19 words (prefix).
    # These windows are evaluated independently for perplexity.
    suffix = " ".join(split_string[-20:-1])
    prefix = " ".join(split_string[0:19])

    suffix_ppl = get_perplexity(suffix)  # high value → likely adversarial noise
    prefix_ppl = get_perplexity(prefix)  # high value → likely adversarial noise

    # Flag as jailbreak if *either* the prefix or suffix exceeds the
    # configured perplexity threshold (default 1845.65).
    if suffix_ppl >= threshold or prefix_ppl >= threshold:
        jb_ps = True
    else:
        jb_ps = False

    result = {"jailbreak": jb_ps}
    return result
