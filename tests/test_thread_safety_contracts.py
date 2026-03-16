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

"""Thread-safety contract tests.

These tests enforce structural invariants on the thread-safety primitives
defined in ``nemoguardrails._thread_safety``.  They do **not** test
concurrent behaviour (that is covered elsewhere); instead they verify
that every public method is properly overridden and acquires the lock,
that downstream consumers use the thread-safe wrappers, and that the
memory-ordering strategy uses ``threading.Event`` rather than a bare
``bool``.
"""

import inspect
import threading

from nemoguardrails._thread_safety import (
    ThreadSafeCache,
    ThreadSafeDict,
    _AtomicInitWrapper,
)
from nemoguardrails.actions.action_dispatcher import ActionDispatcher

# ---- Dict methods that must be overridden --------------------------------
# Every public method on ``dict`` that reads or mutates mapping data must
# be overridden in ThreadSafeDict to acquire the lock.  The exclusion set
# below lists dunders that are pure CPython machinery — they never touch
# the key/value storage, so overriding them would add overhead without
# improving safety.  If CPython adds new data-touching methods in a future
# release, they will be *absent* from this set and the test will flag them.
_DICT_INTERNAL_ONLY = frozenset(
    {
        "__init_subclass__",
        "__subclasshook__",
        "__class_getitem__",
        "__sizeof__",
        "__reduce__",
        "__reduce_ex__",
        "__format__",
        "__dir__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__new__",
        "__hash__",
        "__str__",
        # Comparison dunders are excluded because ThreadSafeDict delegates
        # to the lock-protected __eq__ and does not support ordering.
        "__ge__",
        "__gt__",
        "__le__",
        "__lt__",
        "__ne__",
        # fromkeys is a classmethod that creates a *new* dict — it does not
        # mutate an existing instance's data.
        "fromkeys",
        "__reversed__",
        "__class__",
        # __getstate__ was added to dict in Python 3.12 for pickling support.
        # It does not access mapping data — it delegates to __dict__, which is
        # irrelevant for ThreadSafeDict (we store data in the parent dict slots,
        # not in instance __dict__).  Safe to inherit without a lock wrapper.
        "__getstate__",
    }
)


def _public_dict_methods() -> set:
    """Return the set of public method names defined on ``dict``.

    Includes both regular public methods (get, items, ...) and dunder
    protocol methods (__getitem__, __setitem__, ...) that touch data.
    """
    return {
        name
        for name in dir(dict)
        # Accept non-underscore names (public) OR dunder names (protocol).
        if callable(getattr(dict, name)) and not name.startswith("_") or (name.startswith("__") and name.endswith("__"))
        if name not in _DICT_INTERNAL_ONLY
    }


# This is the cornerstone test class for the thread-safety layer.  It
# uses structural introspection (checking __dict__ and source code) rather
# than runtime concurrency, making it deterministic and fast.  Removing
# it would allow new dict methods inherited from CPython to silently
# bypass the lock — a correctness hazard on free-threaded builds.
class TestThreadSafeDictOverrides:
    """Verify that ThreadSafeDict overrides every relevant dict method."""

    def test_threadsafe_dict_overrides_all_dict_methods(self):
        """Every public dict method that reads or mutates state must be
        overridden in ThreadSafeDict, not inherited from dict.

        If this test fails it means a new method was added to ``dict``
        (or one was missed) and ThreadSafeDict is silently falling back
        to an unguarded implementation.
        """
        dict_methods = _public_dict_methods()
        missing = []
        for name in sorted(dict_methods):
            tsd_attr = getattr(ThreadSafeDict, name, None)
            if tsd_attr is None:
                continue
            # Check the class __dict__ directly — if the method only
            # exists via MRO inheritance from dict, it was never wrapped
            # with lock acquisition and is therefore unsafe.
            if name not in ThreadSafeDict.__dict__:
                missing.append(name)

        assert not missing, (
            f"ThreadSafeDict inherits the following dict methods without overriding them (lock bypass risk): {missing}"
        )

    # Complements the override check above: even if a method *is*
    # overridden, it could forget to acquire the lock.  This test
    # performs a source-level grep for ``self._lock`` as a lightweight
    # proof that the locking discipline is maintained.
    def test_threadsafe_dict_all_methods_acquire_lock(self):
        """Every overridden method in ThreadSafeDict must reference
        ``self._lock`` in its source, proving it acquires the lock.
        """
        unlocked = []
        for name, method in inspect.getmembers(ThreadSafeDict, predicate=inspect.isfunction):
            # Skip private helpers that are not part of the dict API and
            # __init__ (which *creates* the lock rather than acquiring it).
            if name.startswith("_") and not name.startswith("__"):
                continue
            if name == "__init__":
                continue

            source = inspect.getsource(method)
            if "self._lock" not in source:
                unlocked.append(name)

        assert not unlocked, f"The following ThreadSafeDict methods do not reference self._lock: {unlocked}"


# Mirrors the lock-usage check above but for the bounded cache wrapper.
# ThreadSafeCache wraps an OrderedDict with maxsize eviction; every
# public method must hold the lock to prevent concurrent get/put from
# corrupting the eviction order.
class TestThreadSafeCacheLockUsage:
    """Verify that every public method on ThreadSafeCache acquires the lock."""

    def test_threadsafe_cache_all_methods_acquire_lock(self):
        """Every public method on ThreadSafeCache must reference
        ``self._lock`` in its source.
        """
        unlocked = []
        for name, method in inspect.getmembers(ThreadSafeCache, predicate=inspect.isfunction):
            if name.startswith("_") and not name.startswith("__"):
                continue
            if name == "__init__":
                continue

            source = inspect.getsource(method)
            if "self._lock" not in source:
                unlocked.append(name)

        assert not unlocked, f"The following ThreadSafeCache methods do not reference self._lock: {unlocked}"


# This class guards against a subtle but critical bug on ARM64 / Apple
# Silicon.  If someone replaces threading.Event with a bare bool for
# "simplicity", reads and writes can be reordered by the CPU, causing
# one thread to see _done=True before the _result store is visible.
# Removing this test would leave that invariant unprotected.
class TestAtomicInitBarrier:
    """Verify that _AtomicInitWrapper uses threading.Event for memory ordering."""

    def test_atomic_init_uses_event_barrier(self):
        """_AtomicInitWrapper must use ``threading.Event`` (not a bare
        ``bool``) for publication of the initialisation result.

        A bare ``bool`` flag is insufficient on weakly-ordered
        architectures (e.g. ARM64 / Apple Silicon) because the CPU may
        reorder stores, allowing a reader to observe ``_done = True``
        before the ``_result`` write has become visible.
        ``threading.Event`` provides the necessary acquire/release
        fence semantics.
        """
        # Pass a trivial factory — we only care about the barrier type,
        # not the actual initialisation result.
        wrapper = _AtomicInitWrapper(lambda: 42)

        # The ``_done`` attribute must be a threading.Event instance.
        assert isinstance(wrapper._done, threading.Event), (
            f"Expected _done to be threading.Event, got {type(wrapper._done).__name__}"
        )

        # Scan for any sneaky bare-bool sentinel that might have been
        # introduced alongside or instead of the Event.
        for attr_name in dir(wrapper):
            if "done" in attr_name.lower() and attr_name != "_done":
                attr = getattr(wrapper, attr_name)
                assert not isinstance(attr, bool), (
                    f"Found bare bool attribute {attr_name!r} — use threading.Event instead for correct memory ordering"
                )


# Integration-level checks that the thread-safe primitives are actually
# *used* by the framework components that need them.  The previous test
# classes verify the primitives themselves; this class verifies wiring.
class TestDownstreamConsumers:
    """Verify that key framework components use thread-safe primitives."""

    def test_registered_actions_always_threadsafe(self):
        """ActionDispatcher._registered_actions must be a ThreadSafeDict
        so that concurrent action registration and lookup is safe on
        free-threaded Python.
        """
        dispatcher = ActionDispatcher(load_all_actions=False)
        assert isinstance(dispatcher._registered_actions, ThreadSafeDict), (
            f"Expected _registered_actions to be ThreadSafeDict, got {type(dispatcher._registered_actions).__name__}"
        )

    def test_template_caches_are_threadsafe(self):
        """LLMTaskManager's template and variables caches must be
        ThreadSafeCache instances.
        """
        from nemoguardrails.llm.taskmanager import LLMTaskManager

        # Source inspection is used instead of full instantiation because
        # LLMTaskManager.__init__ requires an LLM config object, which
        # would pull in heavy dependencies and slow the test down.
        source = inspect.getsource(LLMTaskManager.__init__)
        assert "ThreadSafeCache" in source, (
            "LLMTaskManager.__init__ does not reference ThreadSafeCache — template caches may not be thread-safe"
        )

        # Belt-and-braces: confirm the import exists at module level so
        # that the reference in __init__ is not a dead name.
        import nemoguardrails.llm.taskmanager as tm_module

        assert hasattr(tm_module, "ThreadSafeCache"), "ThreadSafeCache is not imported in the taskmanager module"


# Backward-compatibility contract tests.  ThreadSafeDict is a drop-in
# replacement for dict; if it stops behaving like one, code throughout
# the framework (and third-party plugins) will break in subtle ways.
class TestThreadSafeDictBackwardCompat:
    """Verify backward compatibility guarantees."""

    # json.dumps(), Pydantic validators, and ``isinstance(x, dict)``
    # checks all rely on dict subclassing.  Removing this test would
    # allow a future refactor to change the base class undetected.
    def test_threadsafe_dict_is_dict_subclass(self):
        """ThreadSafeDict must be a ``dict`` subclass so that
        ``isinstance(obj, dict)`` checks, ``json.dumps``, and Pydantic
        validators continue to work with existing code that expects a
        plain ``dict``.
        """
        d = ThreadSafeDict()
        assert isinstance(d, dict), (
            "ThreadSafeDict is not a dict subclass — this breaks backward compatibility with isinstance checks"
        )
        assert issubclass(ThreadSafeDict, dict)

    # PEP 584 (Python 3.9+) introduced | and |= for dicts.  If these
    # return a plain dict instead of ThreadSafeDict, the result loses
    # its thread-safety guarantees — a silent downgrade.
    def test_threadsafe_dict_pep584_operators(self):
        """PEP 584 merge operators (``|`` and ``|=``) must work and
        return ThreadSafeDict instances.
        """
        a = ThreadSafeDict({"x": 1})
        b = ThreadSafeDict({"y": 2})

        # __or__: a | b — must produce a new ThreadSafeDict.
        merged = a | b
        assert isinstance(merged, ThreadSafeDict), f"a | b returned {type(merged).__name__}, expected ThreadSafeDict"
        assert dict(merged) == {"x": 1, "y": 2}

        # __ior__: a |= b — must mutate in place, preserving the type.
        a |= b
        assert isinstance(a, ThreadSafeDict), f"a |= b changed type to {type(a).__name__}, expected ThreadSafeDict"
        assert dict(a) == {"x": 1, "y": 2}

        # __ror__: plain_dict | threadsafe_dict — exercises the reflected
        # operator.  Python calls b.__ror__(plain) when the left operand's
        # __or__ returns NotImplemented for a ThreadSafeDict right operand.
        plain = {"z": 3}
        result = plain | b
        assert isinstance(result, ThreadSafeDict), (
            f"plain_dict | ThreadSafeDict returned {type(result).__name__}, expected ThreadSafeDict"
        )
        assert dict(result) == {"y": 2, "z": 3}
