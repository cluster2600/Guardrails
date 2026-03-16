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

"""Architectural boundary tests for NeMo Guardrails.

These tests enforce structural invariants across the codebase so that
regressions in thread safety, configuration completeness, and coding
standards are caught early in CI rather than surfacing as subtle
production bugs.

Each test targets a specific architectural rule.  Tests are deliberately
written to be resilient to minor refactors (e.g. new fields, renamed
modules) whilst still catching genuine violations.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from pathlib import Path
from typing import List, Set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Resolve once so every test class shares the same root, regardless of
# working directory at pytest invocation time.
_NEMO_ROOT = Path(__file__).resolve().parent.parent / "nemoguardrails"


def _python_files(root: Path) -> List[Path]:
    """Yield all ``.py`` files under *root*, excluding test directories."""
    results = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".py"):
                results.append(Path(dirpath) / fname)
    return results


# ---------------------------------------------------------------------------
# 1. All caches must be bounded
# ---------------------------------------------------------------------------


class TestAllCachesAreBounded:
    """Ensure that no unbounded ``dict`` caches creep into the source tree.

    Plain ``self._*cache = {}`` or ``self._*cache: dict`` assignments
    indicate a cache that can grow without limit, eventually exhausting
    memory in long-running services.  Every cache should use either
    ``ThreadSafeCache`` (which wraps a bounded ``OrderedDict``) or an
    ``OrderedDict`` with an explicit ``maxsize`` parameter.

    Known exceptions (e.g. ``InMemoryCacheStore`` in
    ``nemoguardrails/embeddings/cache.py``) are listed in an allowlist
    so that the test remains practical rather than dogmatic.
    """

    # Files and variable names that are deliberately unbounded.
    # Format: (relative-to-nemoguardrails path, attribute name).
    # Every entry here is a conscious decision — add a comment explaining
    # *why* the cache is allowed to be unbounded when extending this set.
    _ALLOWLIST: Set[tuple] = {
        # InMemoryCacheStore is a user-facing pluggable cache backend;
        # bounding it would be a breaking behavioural change.
        ("embeddings/cache.py", "_cache"),
        # events_history_cache is acknowledged tech debt (see TODO in
        # llmrails.py); it will be removed when the interface returns
        # a proper state object.
        ("rails/llm/llmrails.py", "events_history_cache"),
    }

    @staticmethod
    def _find_unbounded_cache_assignments(filepath: Path) -> List[str]:
        """Return descriptions of assignments that look like unbounded caches."""
        violations = []
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            # AST parsing catches cache patterns structurally rather than
            # via regex, which avoids false positives from commented-out
            # code or string literals.
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            return violations

        for node in ast.walk(tree):
            # Match: self._*cache = {} (plain dict literal).
            # An empty dict literal assigned to an attribute whose name
            # contains "cache" is the canonical unbounded-cache pattern.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and "cache" in target.attr.lower()
                        and isinstance(node.value, ast.Dict)
                        and len(node.value.keys) == 0
                    ):
                        violations.append(f"L{node.lineno}: {target.attr} = {{}} (unbounded plain dict cache)")

            # Match: self._*cache: dict = ... (annotated assignment).
            # Pydantic models and dataclasses often use annotations; this
            # catches the type-hinted variant of the same anti-pattern.
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and "cache" in target.attr.lower()
                ):
                    ann = node.annotation
                    if isinstance(ann, ast.Name) and ann.id == "dict":
                        violations.append(f"L{node.lineno}: {target.attr}: dict (unbounded plain dict cache)")

        return violations

    def test_all_caches_are_bounded(self) -> None:
        """Scan nemoguardrails source for unbounded dict-based caches.

        Removing this test would allow developers to introduce memory
        leaks by adding ``self._foo_cache = {}`` without any eviction
        policy — a common source of OOM in long-running guardrail
        servers.
        """
        all_violations: List[str] = []

        for pyfile in _python_files(_NEMO_ROOT):
            rel = pyfile.relative_to(_NEMO_ROOT)
            violations = self._find_unbounded_cache_assignments(pyfile)

            # Strip out known-acceptable cases so the test stays practical.
            filtered = []
            for v in violations:
                attr_match = re.match(r"L\d+: (\w+)", v)
                attr_name = attr_match.group(1) if attr_match else ""
                if (str(rel), attr_name) not in self._ALLOWLIST:
                    filtered.append(f"  {rel}: {v}")

            all_violations.extend(filtered)

        assert not all_violations, (
            "Found unbounded dict-based caches.  Use ThreadSafeCache or "
            "OrderedDict with a maxsize instead:\n" + "\n".join(all_violations)
        )


# ---------------------------------------------------------------------------
# 2. _join_config covers all RailsConfig fields
# ---------------------------------------------------------------------------


class TestJoinConfigCoversAllConfigFields:
    """Ensure ``_join_config()`` handles every field on ``RailsConfig``.

    When a new field is added to ``RailsConfig`` but not to
    ``_join_config()``, merging multiple configuration files silently
    drops the new field.  This test compares the two sets and flags
    any gaps.
    """

    # Fields that are intentionally not merged by _join_config because
    # they are computed, internal, or set only at construction time.
    # Extend this set with a comment when a new non-mergeable field is
    # added to RailsConfig.
    _IGNORE_FIELDS: Set[str] = {
        "config_path",
        "imported_paths",
        # Pydantic internal fields (if any surface via model_fields).
    }

    def test_join_config_covers_all_config_fields(self) -> None:
        """Every ``RailsConfig`` model field must appear in ``_join_config``.

        Removing this test would allow new config fields to be silently
        dropped during multi-file config merges — a subtle, hard-to-debug
        production issue.
        """
        from nemoguardrails.rails.llm.config import RailsConfig, _join_config

        # Inspect the source text rather than calling the function, so we
        # can detect coverage without needing valid config objects.
        source = inspect.getsource(_join_config)

        # All model fields on RailsConfig (public API surface).
        config_fields = set(RailsConfig.model_fields.keys())

        # A field is considered "covered" if its name appears as a quoted
        # string anywhere in _join_config's source — this catches both the
        # additional_fields list and any bespoke merge logic.
        covered_fields: Set[str] = set()
        for field in config_fields:
            if f'"{field}"' in source or f"'{field}'" in source:
                covered_fields.add(field)

        uncovered = config_fields - covered_fields - self._IGNORE_FIELDS
        assert not uncovered, (
            "The following RailsConfig fields are not handled by "
            "_join_config().  Either add them to ``additional_fields``, "
            "write explicit merge logic, or add them to the ignore list "
            "in this test:\n" + "\n".join(f"  - {f}" for f in sorted(uncovered))
        )


# ---------------------------------------------------------------------------
# 3. No plain dict for shared mutable state
# ---------------------------------------------------------------------------


class TestNoPlainDictForSharedMutableState:
    """Verify that known shared-mutable registries use ``ThreadSafeDict``.

    On free-threaded Python (3.14t, PEP 703), concurrent mutations of a
    plain ``dict`` are undefined behaviour.  Critical registries such as
    ``ActionDispatcher._registered_actions`` must use ``ThreadSafeDict``
    to guarantee correctness on both GIL-enabled and free-threaded
    builds.
    """

    # This is the single most critical thread-safety invariant in the
    # codebase.  A plain dict here would cause data races on
    # free-threaded Python when actions are registered concurrently.
    def test_action_dispatcher_uses_thread_safe_dict(self) -> None:
        """``ActionDispatcher._registered_actions`` must be a ``ThreadSafeDict``."""
        from nemoguardrails._thread_safety import ThreadSafeDict
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher._registered_actions, ThreadSafeDict), (
            "ActionDispatcher._registered_actions should be a ThreadSafeDict, "
            f"but is {type(dispatcher._registered_actions).__name__}"
        )

    # Backward-compatibility guard: if ThreadSafeDict stops inheriting
    # from dict, json.dumps() and Pydantic validators that expect a
    # plain dict would break silently.
    def test_action_dispatcher_registry_is_dict_compatible(self) -> None:
        """``ThreadSafeDict`` must pass ``isinstance(x, dict)`` checks.

        Third-party code and internal helpers may perform ``isinstance``
        checks against ``dict``.  ``ThreadSafeDict`` subclasses ``dict``
        specifically to satisfy this requirement.
        """
        from nemoguardrails.actions.action_dispatcher import ActionDispatcher

        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher._registered_actions, dict), (
            "ThreadSafeDict should be a dict subclass so that isinstance(obj, dict) checks pass"
        )


# ---------------------------------------------------------------------------
# 4. Computed fields on property mixins
# ---------------------------------------------------------------------------


class TestComputedFieldsOnPropertyMixins:
    """Verify that ``@property`` methods on Pydantic models that are part
    of the serialised API are decorated with ``@computed_field``.

    Properties that should be serialised (e.g. ``flows`` on
    ``_RailSectionMixin``) must have ``@computed_field`` so that
    ``model_dump()`` and ``model_dump_json()`` include them.

    Properties that expose internal / private state (e.g.
    ``compiled_patterns``) or are purely for introspection (e.g.
    ``has_dependencies``) should *not* have ``@computed_field``.
    """

    # Properties that MUST have @computed_field — without it, model_dump()
    # omits them and downstream YAML/JSON serialisation is incomplete.
    _REQUIRE_COMPUTED_FIELD = {
        "flows",
    }

    # Properties that must NOT have @computed_field — exposing them in
    # serialised output would leak compiled regex objects or internal
    # dependency metadata to API consumers.
    _MUST_NOT_HAVE_COMPUTED_FIELD = {
        "compiled_patterns",
        "has_dependencies",
    }

    # Catches the case where someone removes @computed_field from flows,
    # which would silently break config serialisation and merging.
    def test_flows_property_has_computed_field(self) -> None:
        """The ``flows`` property on rail section classes must be a computed field."""
        from nemoguardrails.rails.llm.config import _RailSectionMixin

        # Pydantic v2 exposes computed fields via model_computed_fields.
        computed = _RailSectionMixin.model_computed_fields
        assert "flows" in computed, (
            "_RailSectionMixin.flows must be decorated with @computed_field "
            "so that it is included in model_dump() output"
        )

    # Prevents internal helpers from accidentally appearing in API
    # responses or exported YAML when someone adds @computed_field
    # during a refactor.
    def test_internal_properties_are_not_computed_fields(self) -> None:
        """Internal properties must not leak into serialised output."""
        from nemoguardrails.rails.llm.config import (
            RegexDetectionOptions,
            _RailSectionMixin,
        )

        for cls, prop_name in [
            (_RailSectionMixin, "has_dependencies"),
            (RegexDetectionOptions, "compiled_patterns"),
        ]:
            computed = cls.model_computed_fields
            assert prop_name not in computed, (
                f"{cls.__name__}.{prop_name} should NOT be a @computed_field — "
                f"it is an internal helper, not part of the serialised API"
            )


# ---------------------------------------------------------------------------
# 5. Environment variable parsing has error handling
# ---------------------------------------------------------------------------


class TestEnvVarParsingHasErrorHandling:
    """Ensure that ``int(os.environ.get(...))`` patterns are wrapped in
    try/except blocks.

    Bare ``int(os.environ.get("VAR", "default"))`` calls will raise a
    ``ValueError`` if the environment variable contains a non-numeric
    string.  All such conversions should be wrapped in a try/except
    with a sensible fallback or error message.
    """

    # Regex that matches bare int() wrapping an env-var read.  All three
    # common forms are covered: os.environ.get(), os.environ[], os.getenv().
    _PATTERN = re.compile(
        r"int\s*\(\s*os\.(environ\.get|environ\[|getenv)\s*\(",
        re.MULTILINE,
    )

    def test_env_var_parsing_has_error_handling(self) -> None:
        """Scan for unprotected int(os.environ.get(...)) calls.

        Removing this test would allow unguarded int() conversions to
        ship — a single mis-set environment variable would then crash
        the entire guardrails server at import time with a ValueError.
        """
        violations: List[str] = []

        for pyfile in _python_files(_NEMO_ROOT):
            try:
                source = pyfile.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for match in self._PATTERN.finditer(source):
                lineno = source[: match.start()].count("\n") + 1

                # Heuristic: walk backwards up to 20 lines looking for
                # a ``try:`` that has not yet been closed by an
                # ``except`` or ``finally:``.  This is an approximation
                # (not a full control-flow analysis), but it catches the
                # vast majority of cases without needing an AST.
                lines = source[: match.start()].splitlines()
                inside_try = False
                for line in reversed(lines[-20:]):
                    stripped = line.strip()
                    if stripped.startswith("try:") or stripped.startswith("try "):
                        inside_try = True
                        break
                    if stripped.startswith(("except", "finally:")):
                        break

                if not inside_try:
                    rel = pyfile.relative_to(_NEMO_ROOT)
                    violations.append(f"  {rel}:{lineno}")

        assert not violations, "Found int(os.environ.get(...)) calls without try/except error handling:\n" + "\n".join(
            violations
        )


# ---------------------------------------------------------------------------
# 6. thread_pool in _join_config additional_fields
# ---------------------------------------------------------------------------


class TestThreadPoolInJoinConfig:
    """Verify that ``thread_pool`` is listed in the ``additional_fields``
    within ``_join_config()``.

    If ``thread_pool`` is missing from ``additional_fields``, merging
    multiple config files will silently discard the thread-pool
    configuration, leading to ``@cpu_bound`` actions running inline
    and blocking the event loop.
    """

    def test_thread_pool_in_join_config(self) -> None:
        """``thread_pool`` must be in ``_join_config``'s additional_fields.

        Removing this test would allow a refactor of _join_config to
        accidentally drop thread_pool from the merge list, causing
        @cpu_bound actions to silently fall back to inline execution
        and block the event loop.
        """
        from nemoguardrails.rails.llm.config import _join_config

        source = inspect.getsource(_join_config)

        # First pass: confirm the string appears at all.
        assert '"thread_pool"' in source or "'thread_pool'" in source, (
            "thread_pool is not referenced in _join_config().  "
            "Add it to the additional_fields list so that thread-pool "
            "configuration is preserved when merging config files."
        )

        # Second pass: confirm it lives inside the additional_fields
        # list specifically, not in a comment or unrelated variable.
        match = re.search(
            r"additional_fields\s*=\s*\[(.*?)\]",
            source,
            re.DOTALL,
        )
        assert match is not None, "Could not locate the additional_fields list in _join_config()"
        additional_fields_source = match.group(1)
        assert "thread_pool" in additional_fields_source, (
            "thread_pool is mentioned in _join_config() but is not inside "
            "the additional_fields list.  Move it there so config merging "
            "picks it up correctly."
        )
