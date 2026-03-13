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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

yara = None
try:
    import yara
except ImportError:
    pass

from nemoguardrails import RailsConfig  # noqa: E402
from nemoguardrails.actions import action  # noqa: E402
from nemoguardrails.library.injection_detection.yara_config import ActionOptions, Rules  # noqa: E402

YARA_DIR = Path(__file__).resolve().parent.joinpath("yara_rules")

log = logging.getLogger(__name__)


class InjectionDetectionResult(TypedDict):
    is_injection: bool
    text: str
    detections: List[str]


def _check_yara_available():
    if yara is None:
        raise ImportError(
            "The yara module is required for injection detection. Please install it using: pip install yara-python"
        )


def _validate_injection_config(config: RailsConfig) -> None:
    """
    Validates the injection detection configuration.

    This function performs early validation of the user-supplied configuration
    before any YARA compilation or matching takes place. It ensures that the
    configuration block is present, the chosen action is one of the recognised
    options, and that any custom ``yara_path`` actually points to a valid
    directory on disc.

    Args:
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Raises:
        ValueError: If the configuration is missing or invalid.
        FileNotFoundError: If the provided `yara_path` is not a directory.
    """
    command_injection_config = config.rails.config.injection_detection

    # Fail fast if the entire injection_detection block is absent — the
    # caller should never reach this point without one, but we guard
    # against mis-configuration nonetheless.
    if command_injection_config is None:
        msg = "Injection detection configuration is missing in the provided RailsConfig."
        log.error(msg)
        raise ValueError(msg)

    # Validate that the chosen action is one of "reject", "omit", or
    # "sanitize". Any other value is a configuration error.
    action_option = command_injection_config.action
    if action_option not in ActionOptions:
        msg = "Expected 'reject', 'omit', or 'sanitize' action in injection config but got %s" % action_option
        log.error(msg)
        raise ValueError(msg)

    # When no inline yara_rules dictionary is provided, the user may
    # optionally supply a custom yara_path directory.  We validate it
    # early so that errors surface at initialisation time rather than
    # at the first match attempt.
    if not command_injection_config.yara_rules:
        yara_path = command_injection_config.yara_path
        if yara_path and isinstance(yara_path, str):
            yara_path = Path(yara_path)
            if not yara_path.exists() or not yara_path.is_dir():
                msg = "Provided `yara_path` value in injection config %s is not a directory." % yara_path
                log.error(msg)
                raise FileNotFoundError(msg)
        elif yara_path and not isinstance(yara_path, str):
            msg = "Expected a string value for `yara_path` but got %r instead." % type(yara_path)
            log.error(msg)
            raise ValueError(msg)


def _extract_injection_config(
    config: RailsConfig,
) -> Tuple[str, Path, Tuple[str], Optional[Dict[str, str]]]:
    """
    Extracts and processes the injection detection configuration values.

    Normalises the raw configuration into a consistent tuple of values that
    downstream functions (_load_rules, the action handlers) can consume
    without having to re-inspect the config object.

    Args:
        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Returns:
        Tuple[str, Path, Tuple[str], Optional[Dict[str, str]]]: A tuple containing the action option,
        the YARA path, the injection rules, and optional yara_rules dictionary.

    Raises:
        ValueError: If the injection rules contain invalid elements.
    """
    command_injection_config = config.rails.config.injection_detection
    yara_rules = command_injection_config.yara_rules

    # Determine the directory containing .yara files.  When inline
    # yara_rules are provided the path is only used for validation
    # purposes, so we fall back to the bundled YARA_DIR.
    if yara_rules:
        # Inline rules take precedence; the path is kept only for
        # structural consistency.
        yara_path = YARA_DIR
    else:
        # Fall back to the user-supplied path or, if absent, the
        # built-in rules directory shipped with this library.
        yara_path = command_injection_config.yara_path or YARA_DIR
        if isinstance(yara_path, str):
            yara_path = Path(yara_path)

    # Convert to tuple for immutability — prevents accidental mutation
    # of the configured rule list during processing.
    injection_rules = tuple(command_injection_config.injections)

    # When loading from files (no inline yara_rules), verify that every
    # requested rule name either belongs to the built-in set or has a
    # corresponding .yara file in the target directory.  This avoids
    # opaque YARA compilation errors later.
    if not yara_rules and not set(injection_rules) <= Rules:
        if not all([yara_path.joinpath(f"{module_name}.yara").is_file() for module_name in injection_rules]):
            default_rule_names = ", ".join([member.value for member in Rules])
            msg = (
                "Provided set of `injections` in injection config %r contains elements not in available rules. "
                "Provided rules are in %r."
            ) % (injection_rules, default_rule_names)
            log.error(msg)
            raise ValueError(msg)

    return command_injection_config.action, yara_path, injection_rules, yara_rules


def _load_rules(
    yara_path: Path, rule_names: Tuple, yara_rules: Optional[Dict[str, str]] = None
) -> Union["yara.Rules", None]:
    """
    Loads and compiles YARA rules from either file paths or direct rule strings.

    Compilation is performed eagerly so that syntax errors are surfaced at
    initialisation time rather than at the first match attempt.  The
    compiled ``yara.Rules`` object is thread-safe for matching and can be
    shared across multiple invocations.

    Args:
        yara_path (Path): The path to the directory containing YARA rule files.
        rule_names (Tuple): A tuple of YARA rule names to load.
        yara_rules (Optional[Dict[str, str]]): Dictionary mapping rule names to YARA rule strings.

    Returns:
        Union['yara.Rules', None]: The compiled YARA rules object if successful,
        or None if no rule names are provided.

    Raises:
        yara.SyntaxError: If there is a syntax error in the YARA rules.
        ImportError: If the yara module is not installed.
    """

    if len(rule_names) == 0:
        log.warning("Injection config was provided but no modules were specified. Returning None.")
        return None

    try:
        if yara_rules:
            # Inline rule strings: filter the dictionary to only the
            # requested rule names, then compile from source strings.
            rules_source = {name: rule for name, rule in yara_rules.items() if name in rule_names}
            rules = yara.compile(sources={rule_name: rules_source[rule_name] for rule_name in rule_names})
        else:
            # File-based rules: build a mapping of rule name to its
            # on-disc .yara file path, then compile from filepaths.
            rules_to_load = {rule_name: str(yara_path.joinpath(f"{rule_name}.yara")) for rule_name in rule_names}
            rules = yara.compile(filepaths=rules_to_load)
    except yara.SyntaxError as e:
        # Gracefully degrade rather than crashing the entire guardrail
        # pipeline — log the error and return None so the caller can
        # decide how to proceed.
        msg = f"Failed to initialise injection detection due to configuration or YARA rule error: YARA compilation failed: {e}"
        log.error(msg)
        return None
    return rules


def _omit_injection(text: str, matches: list["yara.Match"]) -> Tuple[bool, str]:
    """
    Attempts to strip the offending injection attempts from the provided text.

    This is a plain synchronous helper.  The previous implementation used a
    ``@cpu_bound`` decorator which wrapped the function in its own
    ``run_in_executor`` call.  That decorator was removed because the caller
    (``injection_detection``) already offloads CPU-bound work to a thread
    via ``loop.run_in_executor``.  Nesting two executor calls would add
    unnecessary overhead and complicate error propagation without any
    concurrency benefit.

    Note:
        This method may not be completely effective and could still result in
        malicious activity.

    Args:
        text (str): The text to check for command injection.
        matches (list['yara.Match']): A list of YARA rule matches.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if injection was detected and modified,
                    False if the text is safe (i.e., not modified).
            - str: The text, with detected injections stripped out if modified.

    Raises:
        ImportError: If the yara module is not installed.
    """

    original_text = text
    modified_text = text
    is_injection = False

    # Walk through every matched YARA rule and every string instance
    # within that rule, removing the matched substring from the text.
    for match in matches:
        if match.strings:
            for match_string in match.strings:
                for instance in match_string.instances:
                    try:
                        # Decode the raw byte match back to a UTF-8 string
                        # so we can perform a safe string replacement.
                        plaintext = instance.plaintext().decode("utf-8")
                        if plaintext in modified_text:
                            modified_text = modified_text.replace(plaintext, "")
                    except (AttributeError, UnicodeDecodeError) as e:
                        log.warning(f"Error processing match: {e}")

    # Only flag as an injection if the text was actually altered.
    if modified_text != original_text:
        is_injection = True
        return is_injection, modified_text
    else:
        is_injection = False
        return is_injection, original_text


def _sanitize_injection(text: str, matches: list["yara.Match"]) -> Tuple[bool, str]:
    """
    Attempts to sanitise the offending injection attempts in the provided text.
    This is done by 'de-fanging' the offending content, transforming it into a
    state that will not execute downstream commands.

    Like ``_omit_injection`` and ``_reject_injection``, this helper is a plain
    synchronous function.  The ``@cpu_bound`` decorator that previously wrapped
    it was removed for the same reason: the caller already handles thread
    offloading, so a nested executor call would be redundant.

    Note:
        This method may not be completely effective and could still result in
        malicious activity. Sanitising malicious input instead of rejecting or
        omitting it is inherently risky and generally not recommended.

    Args:
        text (str): The text to check for command injection.
        matches (list['yara.Match']): A list of YARA rule matches.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if injection was detected, False otherwise.
            - str: The sanitised text, or original text depending on sanitisation outcome.
                   Currently, this function will always raise NotImplementedError.

    Raises:
        NotImplementedError: If the sanitisation logic is not implemented.
        ImportError: If the yara module is not installed.
    """
    # Sanitisation logic has not been implemented yet.  Raising here
    # ensures callers fail loudly rather than silently passing through
    # potentially dangerous input.
    raise NotImplementedError("Injection sanitisation is not yet implemented. Please use 'reject' or 'omit'")
    # Hypothetical logic if implemented, to match existing behaviour in injection_detection:
    # sanitised_text_attempt = "..." # result of sanitisation
    # if sanitised_text_attempt != text:
    #     return True, text  # Original text returned, marked as injection detected
    # else:
    #     return False, sanitised_text_attempt


def _reject_injection(text: str, rules: "yara.Rules") -> Tuple[bool, List[str]]:
    """
    Detects whether the provided text contains potential injection attempts.

    This is a plain synchronous function.  The ``@cpu_bound`` decorator that
    previously wrapped it was removed because the async caller
    (``injection_detection``) already offloads this work to a thread pool via
    ``loop.run_in_executor``.  Keeping the decorator would have caused a
    double thread-hop — the outer executor would spawn a thread, and the
    inner ``@cpu_bound`` wrapper would spawn yet another — wasting resources
    and making exception handling more difficult.

    Args:
        text (str): The text to check for command injection.
        rules ('yara.Rules'): The pre-compiled YARA rules object.

    Returns:
        Tuple[bool, List[str]]: A tuple containing:
            - bool: True if attempted exploitation is detected, False otherwise.
            - List[str]: List of matched rule names.

    Raises:
        ValueError: If the `action` parameter in the configuration is invalid.
        ImportError: If the yara module is not installed.
    """

    if rules is None:
        log.warning(
            "reject_injection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return False, []

    # Perform the actual YARA match — this is the CPU-bound operation
    # that justifies the run_in_executor offloading in the caller.
    matches = rules.match(data=text)
    if matches:
        matched_rules = [match_name.rule for match_name in matches]
        log.info(f"Input matched on rule {', '.join(matched_rules)}.")
        return True, matched_rules
    else:
        return False, []


@action()
async def injection_detection(text: str, config: RailsConfig) -> InjectionDetectionResult:
    """
    Detects and mitigates potential injection attempts in the provided text.

    This is the main entry point registered as a NeMo Guardrails action.  It
    orchestrates the full detection pipeline:

      1. Validate and extract the user-supplied configuration.
      2. Compile (or retrieve) the YARA rules.
      3. Perform pattern matching against the input text.
      4. Apply the configured mitigation strategy (reject / omit / sanitise).

    All CPU-bound work — in particular YARA pattern matching — is offloaded
    to the default thread-pool executor via ``loop.run_in_executor``.  This
    is critical because ``rules.match()`` can block for a non-trivial amount
    of time on large inputs or complex rule sets, and running it on the
    asyncio event loop would starve other coroutines.  The helper functions
    (``_reject_injection``, ``_omit_injection``, ``_sanitize_injection``) are
    plain synchronous callables precisely so they can be handed to the
    executor without any nested async machinery.

    Depending on the configuration, this function can omit or sanitise the
    detected injection attempts. If the action is set to "reject", it
    delegates to the ``_reject_injection`` helper.

    Args:
        text (str): The text to check for command injection.

        config (RailsConfig): The Rails configuration object containing injection detection settings.

    Returns:
        InjectionDetectionResult: A TypedDict containing:
            - is_injection (bool): Whether an injection was detected. True if any injection is detected,
                            False if no injection is detected.
            - text (str): The sanitised or original text
            - detections (List[str]): List of matched rule names if any injection is detected

    Raises:
        ValueError: If the `action` parameter in the configuration is invalid.
        NotImplementedError: If an unsupported action is encountered.
        ImportError: If the yara module is not installed.
    """
    # Ensure the yara-python package is available before proceeding.
    _check_yara_available()

    # Validate configuration early so that misconfiguration errors are
    # raised before any expensive YARA compilation takes place.
    _validate_injection_config(config)

    # Extract the normalised configuration tuple (action, path, rule
    # names, and optional inline rule strings).
    action_option, yara_path, rule_names, yara_rules = _extract_injection_config(config)

    # Compile the YARA rules.  This returns None when no rule names
    # have been specified, which we treat as a no-op below.
    rules = _load_rules(yara_path, rule_names, yara_rules)

    if rules is None:
        log.warning(
            "injection detection guardrail was invoked but no rules were specified in the InjectionDetection config."
        )
        return InjectionDetectionResult(is_injection=False, text=text, detections=[])

    # ----------------------------------------------------------------
    # Offload CPU-bound YARA matching to a thread pool.
    #
    # YARA's ``rules.match()`` performs regex and byte-pattern scanning
    # which is entirely CPU-bound.  Running it directly inside the
    # async function would block the event loop and degrade throughput
    # for all concurrent requests.
    #
    # ``loop.run_in_executor(None, ...)`` dispatches the callable to
    # the default ``ThreadPoolExecutor``.  We use ``functools.partial``
    # to bind the arguments because ``run_in_executor`` does not accept
    # keyword arguments.
    # ----------------------------------------------------------------
    loop = asyncio.get_running_loop()

    if action_option == "reject":
        # Reject mode: determine whether any rule fires and return
        # the original text unchanged alongside the detection flag.
        is_injection, detected_rules = await loop.run_in_executor(
            None, functools.partial(_reject_injection, text, rules)
        )
        return InjectionDetectionResult(is_injection=is_injection, text=text, detections=detected_rules)
    else:
        # For omit / sanitise modes we need the full list of YARA
        # match objects (not just a boolean) so the helpers can
        # inspect individual string instances.
        matches = await loop.run_in_executor(None, functools.partial(rules.match, data=text))
        if matches:
            detected_rules_list = [match_name.rule for match_name in matches]
            log.info(f"Input matched on rule {', '.join(detected_rules_list)}.")

            if action_option == "omit":
                # Omit mode: strip matched substrings from the text.
                # The helper is offloaded to the executor because it
                # iterates over potentially many match instances and
                # performs repeated string replacements.
                is_injection, result_text = await loop.run_in_executor(
                    None, functools.partial(_omit_injection, text, matches)
                )
                return InjectionDetectionResult(
                    is_injection=is_injection,
                    text=result_text,
                    detections=detected_rules_list,
                )
            elif action_option == "sanitize":
                # Sanitise mode: not yet implemented.
                # _sanitize_injection will raise NotImplementedError
                # before returning a tuple.  The assignment below is
                # kept for structural consistency should it be
                # implemented in future.
                is_injection, result_text = _sanitize_injection(text, matches)
                return InjectionDetectionResult(
                    is_injection=is_injection,
                    text=result_text,
                    detections=detected_rules_list,
                )
            else:
                raise NotImplementedError(
                    f"Expected `action` parameter to be 'reject', 'omit', or 'sanitize' but got {action_option} instead."
                )
        # No YARA rules matched — the input is considered safe.
        else:
            return InjectionDetectionResult(is_injection=False, text=text, detections=[])
