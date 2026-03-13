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

"""Compatibility shim for langchain on Python 3.14+ (PEP 649).

Python 3.14 introduces PEP 649 (deferred evaluation of annotations).
Under PEP 649, pydantic resolves type annotations using vars(cls),
which means a class method named ``dict()`` shadows the builtin
``dict`` type.  langchain 0.3.x defines ``Chain.dict()`` and similar
methods, so ``dict[str, Any]`` annotations resolve to the *method*
rather than the builtin — raising ``TypeError: 'function' object is
not subscriptable``.

The fix landed in langchain 1.x (``langchain-classic``) via
langchain-ai/langchain#33575, which explicitly uses ``builtins.dict``
in annotations.  This module provides a workaround for environments
still running langchain 0.3.x on Python >= 3.14.
"""

import builtins
import logging
import sys

log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Determine once, at import time, whether PEP 649 behaviour is in
# effect.  PEP 649 was accepted for Python 3.14, so any interpreter
# at that version or above will lazily evaluate annotations via
# module-level globals, which is the root cause of the shadowing
# problem described in this module's docstring.
# -----------------------------------------------------------------
_NEEDS_PATCH = sys.version_info >= (3, 14)


def _patch_langchain_dict_shadow():
    """Inject ``builtins.dict`` and ``builtins.list`` into langchain module namespaces.

    **Why this function exists**

    Under PEP 649, annotations are no longer evaluated eagerly at class
    definition time.  Instead, they are evaluated lazily — typically the
    first time pydantic (or ``typing.get_type_hints``) introspects the
    class.  The evaluation context used for resolving those annotations
    is the *module-level* namespace (i.e. ``vars(module)``).

    In langchain 0.3.x several classes (``Chain``, ``Serializable``,
    various runnables) define instance methods called ``dict()`` and
    ``list()``.  When pydantic evaluates an annotation such as
    ``dict[str, Any]``, it looks up ``dict`` in the class namespace
    first.  Because the class has a *method* called ``dict``, that
    method object is found instead of the builtin ``dict`` type, which
    causes ``TypeError: 'function' object is not subscriptable``.

    **Why the patch is applied AFTER imports, not before**

    The patch works by writing ``builtins.dict`` and ``builtins.list``
    directly into the module-level ``__dict__`` of each affected
    langchain module.  We must do this *after* the modules have been
    imported (and therefore exist in ``sys.modules``) for two reasons:

    1. The module object must already exist so we can mutate its
       namespace.  Patching ``sys.modules`` entries that have not been
       imported yet would have no effect — the entry simply does not
       exist.

    2. PEP 649 annotation evaluation happens lazily, so we only need
       the patched names to be present by the time pydantic first
       accesses the annotations — which is always *after* the import
       has completed.  Patching immediately after import is therefore
       the earliest safe moment.

    **Idempotency**

    The function is safe to call multiple times.  A sentinel attribute
    (``__dict_patched__``) is stamped onto each module the first time
    it is patched; subsequent calls skip modules that already carry
    the sentinel, avoiding redundant work.
    """
    # On Python < 3.14, PEP 649 is not active, so annotations are
    # evaluated eagerly (the old ``from __future__ import annotations``
    # behaviour notwithstanding) and no shadowing can occur.  Bail out
    # immediately.
    if not _NEEDS_PATCH:
        return

    # -----------------------------------------------------------------
    # Modules whose classes define a ``dict()`` or ``list()`` method
    # that would shadow the corresponding builtin under PEP 649
    # annotation resolution.  This list covers the known offenders in
    # langchain 0.3.x and langchain_core.
    #
    # If future langchain versions add more such classes, their parent
    # modules should be appended here.
    # -----------------------------------------------------------------
    _MODULES_TO_PATCH = [
        "langchain.chains.base",       # Chain.dict()
        "langchain.chains",            # Re-exported Chain class
        "langchain.schema",            # Various schema base classes
        "langchain.schema.runnable.base",  # Runnable / RunnableSerializable
        "langchain_core.runnables.base",   # Core Runnable hierarchy
        "langchain_core.load.serializable",  # Serializable.dict()
    ]

    patched = []
    for mod_name in _MODULES_TO_PATCH:
        # Only patch modules that are *already imported*.  We look them
        # up in ``sys.modules`` rather than importing them ourselves,
        # because importing a module for the sole purpose of patching
        # it could trigger the very TypeError we are trying to prevent.
        mod = sys.modules.get(mod_name)

        # Skip modules that are not yet loaded or that have already
        # been patched (identified by the ``__dict_patched__`` sentinel).
        if mod is not None and not hasattr(mod, "__dict_patched__"):
            # Overwrite the module-level names ``dict`` and ``list``
            # with the real builtins.  When PEP 649 evaluates an
            # annotation like ``dict[str, Any]`` it will now find the
            # builtin type rather than the shadowing instance method.
            mod.dict = builtins.dict  # type: ignore[attr-defined]
            mod.list = builtins.list  # type: ignore[attr-defined]

            # Stamp a sentinel so that repeated calls to this function
            # (which is invoked after every langchain import helper)
            # do not re-patch the same module unnecessarily.
            mod.__dict_patched__ = True  # type: ignore[attr-defined]
            patched.append(mod_name)

    if patched:
        log.debug("Patched builtins into langchain modules: %s", patched)


def safe_import_langchain():
    """Import the ``langchain`` package with PEP 649 safety.

    This is the primary entry point for any code in NeMo Guardrails that
    needs to use langchain.  It wraps the bare ``import langchain``
    statement with two layers of protection:

    1. **ImportError handling** — provides a clear, actionable message
       if langchain is not installed at all, including a note that
       Python 3.14+ requires langchain >= 1.0.0.

    2. **TypeError handling** — on Python >= 3.14, if the import itself
       triggers the ``'function' object is not subscriptable`` error
       (because annotations are evaluated during module initialisation),
       the exception is caught and re-raised as a descriptive
       ``ImportError`` that points the user to the upstream fix.

    After a successful import, ``_patch_langchain_dict_shadow()`` is
    called to inject the real builtins into the freshly-loaded module
    namespaces.  This ensures that any *subsequent* lazy annotation
    evaluation by pydantic will resolve correctly.

    Returns:
        The imported ``langchain`` module object.

    Raises:
        ImportError: If langchain is missing or incompatible with the
            running Python version.
    """
    try:
        import langchain
    except ImportError:
        # langchain is simply not installed — give the user a helpful
        # installation hint, including the version constraint for 3.14+.
        raise ImportError(
            "langchain is not installed.  Install it with: "
            "pip install langchain  "
            "(On Python >= 3.14, langchain >= 1.0.0 is required.)"
        )
    except TypeError as exc:
        # On Python >= 3.14, a ``TypeError`` during import almost
        # certainly means PEP 649 annotation resolution hit the
        # ``dict()`` method shadow.  Check for the characteristic
        # error message before wrapping the exception.
        if _NEEDS_PATCH and "not subscriptable" in str(exc):
            raise ImportError(
                "langchain 0.3.x is not compatible with Python 3.14+ due to "
                "PEP 649 (deferred annotation evaluation).  The Chain.dict() "
                "method shadows the builtin dict type during annotation "
                "resolution.  Please upgrade to langchain >= 1.0.0 "
                "(langchain-classic) which includes the fix "
                "(langchain-ai/langchain#33575)."
            ) from exc
        # If the TypeError is unrelated to PEP 649, re-raise it
        # unmodified so the original traceback is preserved.
        raise

    # The import succeeded — now patch the loaded modules so that any
    # future lazy annotation evaluation finds the real builtins.
    _patch_langchain_dict_shadow()
    return langchain


def import_init_chat_model():
    """Import ``init_chat_model`` with fallback for langchain 1.x.

    ``init_chat_model`` is the factory function used by NeMo Guardrails
    to instantiate chat model objects.  Its import path changed between
    langchain 0.3.x and langchain 1.x (``langchain-classic``), so this
    helper tries both locations in turn.

    **Fallback strategy**

    1. First, attempt the canonical langchain 0.3.x path:
       ``langchain.chat_models.init_chat_model``.  Both ``ImportError``
       (module not found) and ``TypeError`` (PEP 649 annotation crash)
       are caught and silently suppressed so the fallback can proceed.

    2. If that fails, try the langchain 1.x / ``langchain-classic``
       path: ``langchain_classic.chat_models.init_chat_model``.

    After each successful import, ``_patch_langchain_dict_shadow()`` is
    called to protect any modules that were loaded as a side effect.

    Returns:
        The ``init_chat_model`` callable.

    Raises:
        ImportError: If ``init_chat_model`` cannot be found in any
            known location.
    """
    try:
        from langchain.chat_models import init_chat_model

        # Patch immediately after import so that modules pulled in as
        # transitive dependencies are also covered.
        _patch_langchain_dict_shadow()
        return init_chat_model
    except (ImportError, TypeError):
        # ``ImportError`` — the sub-module does not exist (e.g.
        # langchain is not installed, or this version does not expose
        # the function at this path).
        #
        # ``TypeError`` — PEP 649 annotation resolution failed during
        # import.  Rather than crashing here, fall through to the
        # alternative import path which may be compatible.
        pass

    # langchain 1.x / langchain-classic may expose the function under
    # a different top-level package name.
    try:
        from langchain_classic.chat_models import init_chat_model  # type: ignore[import-not-found]

        _patch_langchain_dict_shadow()
        return init_chat_model
    except ImportError:
        # Neither import path worked — nothing more we can try.
        pass

    raise ImportError(
        "Could not import init_chat_model from langchain.  On Python >= 3.14, langchain >= 1.0.0 is required."
    )


def import_chat_models_base():
    """Import ``langchain.chat_models.base`` with fallback.

    Used by ``providers.py`` to discover supported chat model providers.
    The base module contains the class hierarchy from which all
    langchain chat model implementations inherit.

    **Fallback strategy**

    Identical to :func:`import_init_chat_model` — try the canonical
    langchain 0.3.x path first, then fall back to
    ``langchain_classic.chat_models.base``.

    Unlike the other import helpers in this module, this function
    returns ``None`` instead of raising ``ImportError`` when neither
    path succeeds.  This is intentional: the chat-models-base module
    is only used for *optional* provider discovery, so a missing module
    is not a fatal error — callers should check for ``None`` and
    degrade gracefully.

    Returns:
        The ``langchain.chat_models.base`` module, or ``None`` if it
        could not be imported from any known location.
    """
    try:
        import langchain.chat_models.base as _base

        # Patch any langchain modules that were loaded as a side effect
        # of this import.
        _patch_langchain_dict_shadow()
        return _base
    except (ImportError, TypeError):
        # Same rationale as in ``import_init_chat_model`` — suppress
        # both missing-module and PEP 649 annotation errors so the
        # fallback path can be attempted.
        pass

    try:
        import langchain_classic.chat_models.base as _base  # type: ignore[import-not-found]

        _patch_langchain_dict_shadow()
        return _base
    except ImportError:
        pass

    # Return ``None`` rather than raising, because chat model base
    # discovery is optional — see docstring above.
    return None
