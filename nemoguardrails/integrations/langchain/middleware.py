# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""LangChain AgentMiddleware integration for NeMo Guardrails.

This module provides middleware classes that slot into LangChain's agent
middleware pipeline, allowing NeMo Guardrails' input and output rails to
be evaluated transparently before and after the LLM model call.

Three concrete classes are exported:

* ``GuardrailsMiddleware`` -- the full-featured middleware that can
  enforce both input *and* output rails.
* ``InputRailsMiddleware`` -- a convenience specialisation that only
  checks input rails (useful when output checking is handled elsewhere
  or is not required).
* ``OutputRailsMiddleware`` -- a convenience specialisation that only
  checks output rails.

Integration pattern
-------------------
LangChain agents accept a list of ``AgentMiddleware`` instances.  Each
middleware may define hook methods such as ``before_model`` /
``after_model`` (synchronous) and ``abefore_model`` / ``aafter_model``
(asynchronous).  The agent runtime invokes these hooks at the
appropriate points in its execution loop.

This module hooks into those extension points to run the configured
NeMo Guardrails checks on the conversation messages, blocking, modifying,
or allowing them to pass through depending on the rail evaluation result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nemoguardrails._langchain_compat import _patch_langchain_dict_shadow

# ---------------------------------------------------------------------------
# LangChain import -- we try the modern ``langchain`` package first.  If
# that fails (e.g. langchain-classic or an incompatible Python version)
# we fall back to ``langchain_classic``.
# ---------------------------------------------------------------------------
try:
    from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
except (ImportError, TypeError):
    # langchain 1.x (langchain-classic) or Python 3.14 fallback
    try:
        from langchain_classic.agents.middleware.types import (  # type: ignore[no-redef]
            AgentMiddleware,
            AgentState,
            hook_config,
        )
    except ImportError:
        raise ImportError(
            "Could not import AgentMiddleware from langchain. On Python >= 3.14, langchain >= 1.0.0 is required."
        )
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

# Apply the compatibility patch *after* langchain modules have been loaded
# into ``sys.modules`` so the shim can locate and patch them correctly.
_patch_langchain_dict_shadow()

if TYPE_CHECKING:
    from langgraph.runtime import Runtime as LangGraphRuntime
from nemoguardrails.integrations.langchain.exceptions import GuardrailViolation  # noqa: E402
from nemoguardrails.integrations.langchain.message_utils import (  # noqa: E402
    create_ai_message,
    is_ai_message,
    is_human_message,
    messages_to_dicts,
)
from nemoguardrails.rails.llm.config import RailsConfig  # noqa: E402
from nemoguardrails.rails.llm.llmrails import LLMRails  # noqa: E402
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType  # noqa: E402
from nemoguardrails.utils import get_or_create_event_loop  # noqa: E402

log = logging.getLogger(__name__)


class GuardrailsMiddleware(AgentMiddleware):
    """Core middleware that integrates NeMo Guardrails with LangChain agents.

    This class implements LangChain's ``AgentMiddleware`` interface and
    registers ``before_model`` / ``after_model`` hooks (plus their async
    counterparts) so that every LLM call within an agent run is
    automatically wrapped by the configured NeMo rails.

    Behaviour overview
    ------------------
    * **Input rails** are evaluated in the ``before_model`` hook.  If the
      latest user message is blocked, the middleware short-circuits the
      agent loop by jumping to ``"end"`` and returning a canned blocked
      message.  If the rail *modifies* the input, the user message is
      replaced in-place so that the model sees the sanitised version.
    * **Output rails** are evaluated in the ``after_model`` hook.  A
      blocked or modified AI response is swapped out for the canned
      blocked message or the modified content, respectively.

    Parameters
    ----------
    config_path : str, optional
        Filesystem path to a NeMo Guardrails configuration directory.
    config_yaml : str, optional
        Raw YAML string containing the guardrails configuration.  Exactly
        one of ``config_path`` or ``config_yaml`` must be supplied.
    raise_on_violation : bool
        When ``True``, a ``GuardrailViolation`` exception is raised
        instead of silently replacing the blocked content with a canned
        message.  This is useful in scenarios where the caller wants to
        handle violations programmatically (e.g. returning an HTTP 422
        from a web service).  Defaults to ``False``.
    blocked_input_message : str
        The replacement message returned when an input rail blocks a
        user message.
    blocked_output_message : str
        The replacement message returned when an output rail blocks an
        AI response.
    enable_input_rails : bool
        Master toggle for input rail checking.  When ``False``, the
        ``before_model`` hook becomes a no-op regardless of configuration.
    enable_output_rails : bool
        Master toggle for output rail checking.  When ``False``, the
        ``after_model`` hook becomes a no-op regardless of configuration.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
        blocked_output_message: str = "I cannot provide this response due to content policy.",
        enable_input_rails: bool = True,
        enable_output_rails: bool = True,
    ):
        # Initialise the underlying ``LLMRails`` engine from whichever
        # configuration source was provided.
        if config_path is not None:
            config = RailsConfig.from_path(config_path)
        elif config_yaml is not None:
            config = RailsConfig.from_content(config_yaml)
        else:
            raise ValueError("Either 'config_path' or 'config_yaml' must be provided to GuardrailsMiddleware")

        # The ``LLMRails`` instance holds all compiled rail flows and is
        # responsible for actually executing the safety checks.
        self.rails = LLMRails(config=config)

        # Store behavioural flags for use at hook invocation time.
        self.raise_on_violation = raise_on_violation
        self.blocked_input_message = blocked_input_message
        self.blocked_output_message = blocked_output_message
        self.enable_input_rails = enable_input_rails
        self.enable_output_rails = enable_output_rails

    # ------------------------------------------------------------------
    # Configuration introspection helpers
    # ------------------------------------------------------------------

    def _has_input_rails(self) -> bool:
        """Return ``True`` if at least one input rail flow is configured."""
        return len(self.rails.config.rails.input.flows) > 0

    def _has_output_rails(self) -> bool:
        """Return ``True`` if at least one output rail flow is configured."""
        return len(self.rails.config.rails.output.flows) > 0

    # ------------------------------------------------------------------
    # Message conversion and lookup helpers
    # ------------------------------------------------------------------

    def _convert_to_rails_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Serialise LangChain ``BaseMessage`` objects into the dict format
        expected by the NeMo Guardrails ``check`` / ``check_async`` API.

        The actual serialisation logic lives in ``message_utils`` so that
        it can be shared with other integration points.
        """
        return messages_to_dicts(messages)

    def _get_last_user_message(self, messages: List[BaseMessage]) -> Optional[HumanMessage]:
        """Walk *backwards* through the message list and return the most
        recent ``HumanMessage``, or ``None`` if there is none.

        Searching in reverse is an optimisation -- in a typical
        conversation the last user message is near the end of the list.
        """
        for msg in reversed(messages):
            if is_human_message(msg):
                return msg
        return None

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        """Walk *backwards* through the message list and return the most
        recent ``AIMessage``, or ``None`` if there is none.
        """
        for msg in reversed(messages):
            if is_ai_message(msg):
                return msg
        return None

    # ------------------------------------------------------------------
    # Guardrail failure handler
    # ------------------------------------------------------------------

    def _handle_guardrail_failure(
        self,
        result: RailsResult,
        rail_type: str,
        blocked_message: str,
    ) -> None:
        """Centralised handler invoked when a rail evaluation returns a
        ``BLOCKED`` status.

        Behaviour depends on ``self.raise_on_violation``:

        * If ``True``, a ``GuardrailViolation`` exception is raised
          immediately.  This allows callers higher up the stack to catch
          violations and handle them however they see fit (e.g. returning
          a structured error response from an API endpoint).
        * If ``False`` (the default), a warning is logged but execution
          continues.  The caller (``abefore_model`` / ``aafter_model``)
          is responsible for substituting the blocked content with a
          user-friendly canned message.
        """
        if result.status == RailStatus.BLOCKED:
            failure_message = f"{rail_type.capitalize()} blocked by {result.rail or 'unknown rail'}"

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=failure_message,
                    result=result,
                    rail_type=rail_type,
                )

            log.warning(failure_message)

    # ------------------------------------------------------------------
    # Async hooks -- the primary implementation
    # ------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """Asynchronous *input rail* hook, called by the agent runtime
        just before the LLM model is invoked.

        Workflow
        --------
        1. Early-exit if input rails are disabled or none are configured.
        2. Extract the most recent user (``HumanMessage``) from state.
        3. Convert all messages to the NeMo dict format and run the
           input rail check asynchronously.
        4. Depending on the result:
           - ``BLOCKED`` -- optionally raise, then return a canned AI
             message and instruct the agent loop to jump to ``"end"``,
             effectively short-circuiting the rest of the turn.
           - ``MODIFIED`` -- replace the last user message with the
             sanitised content so the model receives a cleaned version.
           - ``ALLOWED`` (or any other status) -- return ``None`` to
             let the agent continue normally.

        The ``@hook_config(can_jump_to=["end"])`` decorator informs the
        agent runtime that this hook is permitted to issue a ``jump_to``
        directive to the ``"end"`` node.

        Error handling
        --------------
        If the rail check itself raises an unexpected exception:
        - When ``raise_on_violation`` is ``True`` the exception is
          wrapped in a ``GuardrailViolation`` and re-raised.
        - Otherwise, the conversation is defensively blocked (fail-closed)
          to avoid leaking potentially unsafe content past the rails.
        """
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_user_message = self._get_last_user_message(messages)
        if not last_user_message:
            return None

        # Serialise LangChain messages into the dict representation that
        # the NeMo Guardrails engine expects.
        rails_messages = self._convert_to_rails_messages(messages)

        try:
            # Execute only the INPUT rail flows asynchronously.
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.INPUT])

            if result.status == RailStatus.BLOCKED:
                # Notify via exception or log, depending on configuration.
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="input",
                    blocked_message=self.blocked_input_message,
                )
                # Append a canned AI response and jump to "end" so the
                # model is never called with the violating input.
                blocked_msg = create_ai_message(self.blocked_input_message)
                return {"messages": messages + [blocked_msg], "jump_to": "end"}

            if result.status == RailStatus.MODIFIED:
                # The rail rewrote the user's input (e.g. redacting PII).
                # Replace the original message in the conversation so
                # subsequent processing uses the sanitised version.
                log.info("Input modified by rail '%s': content replaced", result.rail or "unknown rail")
                modified_msg = last_user_message.model_copy(update={"content": result.content})
                return {"messages": self._replace_last_human_message(messages, modified_msg)}

            # RailStatus.ALLOWED -- nothing to do; let the agent proceed.
            return None

        except GuardrailViolation:
            # Re-raise violations that we ourselves created so they
            # propagate to the caller without being caught below.
            raise
        except Exception as e:
            # Catch-all for unexpected errors during rail evaluation.
            # We adopt a *fail-closed* strategy: block the request
            # rather than allowing potentially unsafe content through.
            log.error(f"Error checking input rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Input rail execution error: {str(e)}",
                    rail_type="input",
                )

            blocked_msg = create_ai_message(self.blocked_input_message)
            return {"messages": messages + [blocked_msg], "jump_to": "end"}

    # ------------------------------------------------------------------
    # Message replacement helpers
    # ------------------------------------------------------------------

    def _replace_last_human_message(self, messages: list, replacement: HumanMessage) -> list:
        """Return a *new* message list with the last ``HumanMessage``
        swapped out for ``replacement``.

        The original list is not mutated.  If no ``HumanMessage`` is
        found (an unlikely edge case), the replacement is appended.
        """
        for i in range(len(messages) - 1, -1, -1):
            if is_human_message(messages[i]):
                return messages[:i] + [replacement] + messages[i + 1 :]
        return messages + [replacement]

    def _replace_last_ai_message(self, messages: list, replacement: AIMessage) -> list:
        """Return a *new* message list with the last ``AIMessage`` swapped
        out for ``replacement``.

        This is used by the output-rail hook to substitute the model's
        response when it is blocked or modified by a rail.  As with
        ``_replace_last_human_message``, the original list is not
        mutated; a shallow copy with the replacement is returned.

        If no ``AIMessage`` exists in the list (again, an unlikely edge
        case), the replacement is appended instead.
        """
        for i in range(len(messages) - 1, -1, -1):
            if is_ai_message(messages[i]):
                return messages[:i] + [replacement] + messages[i + 1 :]
        return messages + [replacement]

    # ------------------------------------------------------------------
    # Async output-rail hook
    # ------------------------------------------------------------------

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """Asynchronous *output rail* hook, called by the agent runtime
        immediately after the LLM model returns its response.

        The logic mirrors ``abefore_model`` but targets the model's
        *output* rather than the user's input:

        1. Early-exit if output rails are disabled or unconfigured.
        2. Extract the most recent ``AIMessage`` from state.
        3. Run the output rail check asynchronously.
        4. On ``BLOCKED``, replace the AI message with a canned safe
           response (and optionally raise).
        5. On ``MODIFIED``, swap in the rewritten content.
        6. On ``ALLOWED``, return ``None`` to leave the response intact.

        Note that unlike ``abefore_model`` this hook does **not** use
        ``jump_to``; it simply replaces the offending message in the
        existing message list so that subsequent middleware or the final
        response handler sees only the safe version.

        Error handling follows the same fail-closed pattern as the input
        hook -- an unexpected exception results in the AI response being
        replaced with the canned blocked message.
        """
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        # Serialise the full conversation (including the AI reply) for
        # the NeMo Guardrails engine.
        rails_messages = self._convert_to_rails_messages(messages)

        try:
            # Execute only the OUTPUT rail flows asynchronously.
            result = await self.rails.check_async(rails_messages, rail_types=[RailType.OUTPUT])

            if result.status == RailStatus.BLOCKED:
                self._handle_guardrail_failure(
                    result=result,
                    rail_type="output",
                    blocked_message=self.blocked_output_message,
                )
                # Replace the model's response with the canned message.
                blocked_msg = create_ai_message(self.blocked_output_message)
                return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

            if result.status == RailStatus.MODIFIED:
                # The rail rewrote the model's response (e.g. removing
                # sensitive data).  Swap in the sanitised version.
                log.info("Output modified by rail '%s': content replaced", result.rail or "unknown rail")
                modified_msg = last_ai_message.model_copy(update={"content": result.content})
                return {"messages": self._replace_last_ai_message(messages, modified_msg)}

            # RailStatus.ALLOWED -- the response is safe; pass through.
            return None

        except GuardrailViolation:
            raise
        except Exception as e:
            # Fail-closed: replace the AI response with a safe default.
            log.error(f"Error checking output rails: {e}", exc_info=True)

            if self.raise_on_violation:
                raise GuardrailViolation(
                    message=f"Output rail execution error: {str(e)}",
                    rail_type="output",
                )

            blocked_msg = create_ai_message(self.blocked_output_message)
            return {"messages": self._replace_last_ai_message(messages, blocked_msg)}

    # ------------------------------------------------------------------
    # Synchronous hooks -- thin wrappers around the async versions
    # ------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """Synchronous *input rail* hook.

        This is a thin wrapper that delegates to ``abefore_model`` by
        running it on an event loop obtained via
        ``get_or_create_event_loop()``.  It exists because not all
        LangChain agent runtimes invoke hooks asynchronously; the sync
        variant ensures input rails are still enforced in those contexts.

        The same early-exit checks (``enable_input_rails``,
        ``_has_input_rails``, non-empty messages) are duplicated here to
        avoid the overhead of creating / obtaining an event loop when
        the hook would be a no-op anyway.
        """
        if not self.enable_input_rails or not self._has_input_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.abefore_model(state, runtime))

    def after_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """Synchronous *output rail* hook.

        Mirrors ``before_model`` -- delegates to the async
        ``aafter_model`` implementation via an event loop.  The same
        early-exit guards are applied to skip unnecessary async
        overhead.
        """
        if not self.enable_output_rails or not self._has_output_rails():
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_message = self._get_last_ai_message(messages)
        if not last_ai_message:
            return None

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.aafter_model(state, runtime))


# ======================================================================
# Specialised middleware subclasses
# ======================================================================


class InputRailsMiddleware(GuardrailsMiddleware):
    """Convenience subclass that *only* enforces input rails.

    This is useful when output-rail checking is performed by a separate
    middleware instance or is not required at all.  By explicitly
    disabling output rails at initialisation time and overriding the
    ``aafter_model`` hook to be a no-op, we guarantee zero overhead on
    the output path.

    Using ``InputRailsMiddleware`` instead of manually configuring
    ``GuardrailsMiddleware(enable_output_rails=False)`` also makes the
    intent clearer to anyone reading the agent setup code.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_input_message: str = "I cannot process this request due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message=blocked_input_message,
            blocked_output_message="",  # Never used; output rails are disabled.
            enable_input_rails=True,
            enable_output_rails=False,
        )

    async def aafter_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """No-op -- output rails are intentionally disabled."""
        return None

    def after_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """No-op -- this specialisation does not act after the agent."""
        return None


class OutputRailsMiddleware(GuardrailsMiddleware):
    """Convenience subclass that *only* enforces output rails.

    The symmetric counterpart to ``InputRailsMiddleware``.  Input rails
    are disabled, and the ``abefore_model`` hook is overridden to be a
    no-op so that no pre-model checking occurs.

    This is handy when input validation is handled upstream (e.g. by an
    API gateway or a dedicated ``InputRailsMiddleware`` instance) and
    only the model's responses need to be guarded.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_yaml: Optional[str] = None,
        raise_on_violation: bool = False,
        blocked_output_message: str = "I cannot provide this response due to content policy.",
    ):
        super().__init__(
            config_path=config_path,
            config_yaml=config_yaml,
            raise_on_violation=raise_on_violation,
            blocked_input_message="",  # Never used; input rails are disabled.
            blocked_output_message=blocked_output_message,
            enable_input_rails=False,
            enable_output_rails=True,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """No-op -- input rails are intentionally disabled."""
        return None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: LangGraphRuntime) -> Optional[Dict[str, Any]]:
        """No-op -- this specialisation does not act before the agent."""
        return None
