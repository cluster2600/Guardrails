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

import asyncio  # Used to obtain the running event loop for executor offloading
import functools  # Provides functools.partial for binding keyword arguments before offloading
import logging
from functools import lru_cache  # Caches the heavyweight analyser so it is only initialised once

# Presidio is an optional dependency; the imports are wrapped in a try/except
# so that the module can still be loaded without it installed.  The actual
# ImportError is surfaced later, when _get_analyzer is first called.
try:
    from presidio_analyzer import PatternRecognizer  # Enables custom regex-based entity recognisers
    from presidio_analyzer.nlp_engine import NlpEngineProvider  # Factory for the spaCy-backed NLP engine
    from presidio_anonymizer import AnonymizerEngine  # Performs the actual text masking / anonymisation
    from presidio_anonymizer.entities import OperatorConfig  # Configures how each entity type is anonymised
except ImportError:
    # The exception about installing presidio will be on the first call to the analyzer
    pass

from nemoguardrails import RailsConfig  # Top-level configuration object for the guardrails runtime
from nemoguardrails.actions import action  # Decorator that registers a callable as a guardrails action
from nemoguardrails.rails.llm.config import (
    SensitiveDataDetection,  # Pydantic model holding per-source SDD settings and custom recognisers
    SensitiveDataDetectionOptions,  # Per-source (input/output/retrieval) entity list and score threshold
)
from nemoguardrails.rails.llm.dag_scheduler import (
    get_cpu_executor,  # Returns a thread-pool executor sized for CPU-bound work
)
from nemoguardrails.rails.llm.dag_scheduler import get_cpu_executor

log = logging.getLogger(__name__)


# The @lru_cache decorator ensures the analyser (and its underlying spaCy model)
# is only initialised once per unique score_threshold value, avoiding repeated
# multi-second model loads.
@lru_cache
def _get_analyzer(score_threshold: float = 0.4):
    # Guard against nonsensical threshold values before doing any heavy lifting
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be a float between 0 and 1 (inclusive).")
    try:
        # Deferred import: only attempt to load AnalyzerEngine when actually needed,
        # so the rest of the module remains usable without Presidio installed.
        from presidio_analyzer import AnalyzerEngine

    except ImportError:
        raise ImportError(
            "Could not import presidio, please install it with `pip install presidio-analyzer presidio-anonymizer`."
        )

    try:
        import spacy
    except ImportError:
        raise RuntimeError("The spacy module is not installed. Please install it using pip: pip install spacy.")

    # Presidio requires a spaCy language model for tokenisation and NER;
    # en_core_web_lg is the large English model recommended for production use.
    if not spacy.util.is_package("en_core_web_lg"):
        raise RuntimeError(
            "The en_core_web_lg Spacy model was not found. "
            "Please install using `python -m spacy download en_core_web_lg`"
        )

    # We provide this explicitly to avoid the default warning.
    # Explicitly specifying the NLP engine configuration suppresses the
    # default Presidio warning about missing configuration and ensures
    # the correct spaCy model is loaded deterministically.
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }

    # Create NLP engine based on configuration
    # The NlpEngineProvider acts as a factory, constructing a spaCy-backed
    # NLP engine instance from the configuration dictionary above.
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()  # Loads the spaCy model into memory

    # TODO: One needs to experiment with the score threshold to get the right value
    # Construct the Presidio AnalyzerEngine with the custom NLP engine and the
    # caller-specified confidence threshold.  Entities scoring below this
    # threshold are silently discarded by the analyser.
    return AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=score_threshold)


def _get_ad_hoc_recognizers(sdd_config: SensitiveDataDetection):
    """Helper to compute the ad hoc recognizers for a config."""
    ad_hoc_recognizers = []
    # Iterate over user-defined recogniser dictionaries (typically regex
    # patterns declared in the YAML config) and rehydrate each one into
    # a Presidio PatternRecognizer instance that the analyser can use.
    for recognizer in sdd_config.recognizers:
        ad_hoc_recognizers.append(PatternRecognizer.from_dict(recognizer))
    return ad_hoc_recognizers


# ──────────────────────────────────────────────────────────────────────
# Output mapping function
# ──────────────────────────────────────────────────────────────────────
# The detect_sensitive_data_mapping function serves as the output_mapping
# callback for the detect_sensitive_data action.  The guardrails runtime
# calls this with the action's return value to decide whether the rail
# should block.  Because detect_sensitive_data already returns True when
# sensitive data is found (i.e. the request should be blocked), the
# mapping is an identity function — it simply passes the boolean through.
def detect_sensitive_data_mapping(result: bool) -> bool:
    """
    Mapping for detect_sensitive_data.

    Since the function returns True when sensitive data is detected,
    we block if result is True.
    """
    return result


# The @action decorator registers this coroutine with the guardrails action
# registry.  is_system_action=True marks it as a built-in action (as
# opposed to a user-defined one), and output_mapping wires up the mapping
# function above so the runtime knows how to interpret the return value.
@action(is_system_action=True, output_mapping=detect_sensitive_data_mapping)
async def detect_sensitive_data(
    source: str,  # One of "input", "output", or "retrieval" — indicates where the text originated
    text: str,  # The raw text to scan for sensitive entities
    config: RailsConfig,  # The full guardrails configuration, injected by the runtime
    **kwargs,  # Absorbs any extra keyword arguments the runtime may pass
):
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        True if any sensitive data has been detected, False otherwise.
    """
    # Based on the source of the data, we use the right options
    # Retrieve the top-level sensitive-data-detection configuration block
    sdd_config = config.rails.config.sensitive_data_detection
    # Validate the source parameter to avoid silently ignoring misconfiguration
    if source not in ["input", "output", "retrieval"]:
        raise ValueError("source must be one of 'input', 'output', or 'retrieval'")
    # Dynamically fetch the per-source options (e.g. sdd_config.input,
    # sdd_config.output, or sdd_config.retrieval) so that each data
    # direction can specify its own entity list and score threshold.
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)
    # Extract the confidence score threshold; entities below this score are ignored
    default_score_threshold = getattr(options, "score_threshold")

    # If we don't have any entities specified, we stop
    # Short-circuit: if the operator hasn't configured any entities for
    # this source direction, there is nothing to detect.
    if len(options.entities) == 0:
        return False

    # Obtain (or retrieve from cache) the Presidio AnalyzerEngine,
    # initialised with the appropriate confidence threshold.
    analyzer = _get_analyzer(score_threshold=default_score_threshold)

    # Offload the Presidio/spacy NLP analysis to a worker thread so it
    # doesn't block the asyncio event loop.  On free-threaded Python 3.14t
    # this runs in true parallel with other guardrail checks.
    # We use functools.partial to bind keyword arguments because
    # run_in_executor only accepts a zero-argument callable (or a
    # callable with positional args).  get_cpu_executor() returns a
    # shared ThreadPoolExecutor sized for CPU-bound workloads.
    loop = asyncio.get_running_loop()  # Grab the currently running event loop
    results = await loop.run_in_executor(
        get_cpu_executor(),  # Shared thread-pool executor for CPU-intensive tasks
        functools.partial(
            analyzer.analyze,
            text=text,
            language="en",  # Presidio currently only supports English in this integration
            entities=options.entities,  # The list of entity types to look for (e.g. PERSON, EMAIL_ADDRESS)
            ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),  # User-defined regex recognisers
        ),
    )

    # If we have any
    # If the analyser returned one or more detected entities, signal that
    # sensitive data was found so the rail can block the request.
    if results:
        return True

    # No sensitive entities detected — allow the request to proceed
    return False


# The mask_sensitive_data action is registered without an output_mapping
# because it returns the (possibly modified) text directly rather than
# a boolean gate value.  is_system_action=True keeps it in the built-in
# action namespace.
@action(is_system_action=True)
async def mask_sensitive_data(source: str, text: str, config: RailsConfig):
    """Checks whether the provided text contains any sensitive data.

    Args
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns
        The altered text, if applicable.
    """
    # Based on the source of the data, we use the right options
    # Retrieve the SDD configuration and resolve the per-source options,
    # following the same pattern as detect_sensitive_data above.
    sdd_config = config.rails.config.sensitive_data_detection
    assert source in ["input", "output", "retrieval"]  # Hard assertion — programmer error if violated
    options: SensitiveDataDetectionOptions = getattr(sdd_config, source)

    # If we don't have any entities specified, we stop
    # No entities configured means no masking is needed; return text unchanged.
    if len(options.entities) == 0:
        return text

    analyzer = _get_analyzer()  # Uses the default score threshold (0.4)
    # Build an operator mapping: for every entity type the user wants to
    # detect, configure a "replace" operator so Presidio replaces the
    # detected span with a placeholder like <PERSON> or <EMAIL_ADDRESS>.
    operators = {}
    for entity in options.entities:
        operators[entity] = OperatorConfig("replace")  # "replace" swaps the entity with its type label

    # Offload the NLP analysis and anonymisation to a worker thread.
    # Both the analysis and the anonymisation are bundled into a single
    # closure so only one executor dispatch is needed, avoiding extra
    # scheduling overhead and keeping the two steps sequential (the
    # anonymiser depends on the analyser's output).
    def _analyse_and_mask() -> str:
        # Run Presidio's NER-based analysis to locate sensitive spans
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=options.entities,
            ad_hoc_recognizers=_get_ad_hoc_recognizers(sdd_config),
        )
        # Instantiate the anonymiser engine (lightweight — no heavy model loading)
        anonymizer = AnonymizerEngine()
        # Apply the configured operators to each detected span, producing
        # a new string with sensitive values replaced by type labels.
        masked_results = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        return masked_results.text  # Return the anonymised text as a plain string

    loop = asyncio.get_running_loop()
    # Dispatch the combined analyse-and-mask work to the CPU executor,
    # keeping the async event loop free for other concurrent guardrails.
    return await loop.run_in_executor(get_cpu_executor(), _analyse_and_mask)
