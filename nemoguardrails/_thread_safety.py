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

"""Thread-safety primitives for Python 3.14t (free-threaded / no-GIL) builds.

This module provides concurrency utilities that automatically adapt to the
runtime environment:

* On **free-threaded** Python (``Py_GIL_DISABLED=1``), every primitive uses
  real ``threading.RLock`` / ``threading.Lock`` guards so that concurrent
  threads cannot corrupt shared state.
* On **regular** (GIL-enabled) Python, the same primitives still work
  correctly but are effectively zero-overhead because the underlying lock
  objects are never contended under the GIL.

Public API
----------
``is_free_threaded``
    Detect whether the interpreter was built with ``--disable-gil``.

``ThreadSafeDict``
    A ``dict``-like wrapper that guards every mutation with an ``RLock``.

``ThreadSafeCache``
    A bounded LRU cache with lock-protected reads and writes.

``atomic_init``
    A decorator that guarantees a callable is executed exactly once, even
    when multiple threads race to trigger lazy initialisation.
"""

# PEP 563 deferred evaluation of annotations — avoids circular import
# issues and allows forward references in type hints without quoting.
from __future__ import annotations

import functools
import sysconfig
import threading
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)

__all__ = [
    "is_free_threaded",
    "ThreadSafeDict",
    "ThreadSafeCache",
    "atomic_init",
]

# ---------------------------------------------------------------------------
# Runtime detection
# ---------------------------------------------------------------------------

# Module-level sentinel used to cache the result of is_free_threaded().
# Starts as None so the first call performs the sysconfig lookup; every
# subsequent call simply returns the cached boolean.
_FREE_THREADED: Optional[bool] = None


def is_free_threaded() -> bool:
    """Return ``True`` if the running Python interpreter is a free-threaded build.

    Free-threaded Python (PEP 703) sets the ``Py_GIL_DISABLED`` sysconfig
    variable to ``1``.  This function caches the result after the first call.

    The check matters because, on GIL-enabled builds, the GIL itself already
    serialises bytecode execution, so explicit locking is technically
    redundant (though still harmless).  On free-threaded builds, however,
    there is *no* GIL, and without explicit synchronisation, concurrent
    dict mutations or cache accesses could lead to data corruption.

    Returns:
        bool: ``True`` when running on a no-GIL / free-threaded interpreter,
              ``False`` otherwise.
    """
    global _FREE_THREADED
    # This check-then-cache pattern is benign even without a lock: on
    # GIL-enabled builds the GIL serialises access; on free-threaded builds
    # the worst case is two threads both computing the same boolean and
    # writing it — the result is idempotent so no corruption can occur.
    if _FREE_THREADED is None:
        # sysconfig.get_config_var("Py_GIL_DISABLED") returns 1 (truthy)
        # on free-threaded builds and None or 0 on standard builds.
        _FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    return _FREE_THREADED


# ---------------------------------------------------------------------------
# ThreadSafeDict
# ---------------------------------------------------------------------------

# TypeVars for generic dict signatures.  _KT/_VT mirror the conventional key/
# value type parameters of Mapping; _DT is the default-value type for methods
# like .get() and .pop() that accept an optional fallback argument.
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_DT = TypeVar("_DT")


class ThreadSafeDict(dict):
    """A ``dict`` subclass that protects mutations with a ``threading.RLock``.

    **Why subclass ``dict`` rather than wrap one?**
    Subclassing ensures that ``isinstance(obj, dict)`` remains ``True``, so
    the object is accepted anywhere the standard library or third-party code
    expects a plain ``dict`` (e.g. ``json.dumps``, Pydantic validators,
    type-checking guards).  The trade-off is that we must override *every*
    public method to ensure the lock is always acquired — if any inherited
    ``dict`` method were left un-overridden, it would bypass the lock and
    break thread safety on free-threaded Python.

    **Why an ``RLock`` (re-entrant lock) instead of a ``Lock``?**
    Some dict methods may internally call other dict methods (e.g.
    ``setdefault`` may call ``__getitem__`` then ``__setitem__``).  An
    ``RLock`` allows the *same* thread to acquire it multiple times without
    deadlocking, which is essential when method calls are nested.

    All mutating operations (``__setitem__``, ``__delitem__``, ``pop``,
    ``update``, ``setdefault``, ``clear``) acquire the lock before
    modifying internal state.  Read operations (``__getitem__``,
    ``__contains__``, ``get``, ``keys``, ``values``, ``items``, ``__len__``,
    ``__iter__``) also acquire the lock to guarantee a consistent snapshot
    on free-threaded builds.

    On GIL-enabled Python the lock is still acquired but effectively
    uncontended, so the overhead is negligible.

    The class inherits from ``dict`` so it is a drop-in replacement
    everywhere a plain ``dict`` is expected (``isinstance`` checks,
    ``json.dumps``, etc.).

    Example::

        actions: ThreadSafeDict[str, Callable] = ThreadSafeDict()
        actions["greet"] = greet_action
        assert "greet" in actions
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The lock is created *before* calling super().__init__ so that
        # any insertion triggered by the initial data is already guarded.
        #
        # RLock (re-entrant) rather than Lock: several dict methods are
        # compound operations that call other overridden methods on self.
        # For example, __ior__ calls update(), which itself acquires the
        # lock.  A plain Lock would deadlock in this scenario because the
        # same thread would attempt to acquire it twice.  RLock permits
        # the owning thread to re-enter without blocking.
        #
        # On ARM64 (Apple Silicon), RLock.acquire()/release() issue the
        # necessary memory barriers (dmb ish) through pthread_mutex, so
        # stores made inside the critical section are visible to any
        # thread that subsequently acquires the same lock — no additional
        # fencing is required.
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)

    # -- mutating operations ------------------------------------------------
    # Each mutating method acquires the lock for the duration of the
    # underlying C-level dict mutation, preventing torn writes when two
    # threads modify the dict concurrently on free-threaded Python.

    def __setitem__(self, key: Any, value: Any) -> None:
        # Even single-slot writes need the lock on free-threaded Python:
        # CPython's dict implementation may resize the internal hash table
        # during insertion, and a concurrent read during resize can segfault.
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)

    def pop(self, key: Any, *args: Any) -> Any:  # type: ignore[override]
        # *args captures the optional default; we cannot declare it as a
        # keyword argument because dict.pop() uses a positional-only default
        # and we must preserve the "raises KeyError if absent and no default"
        # behaviour by forwarding the argument tuple verbatim.
        with self._lock:
            return super().pop(key, *args)

    def update(self, __m: Any = (), **kwargs: Any) -> None:  # type: ignore[override]
        # Holding the lock for the entire update ensures that a batch of
        # insertions appears atomically to other threads.
        with self._lock:
            super().update(__m, **kwargs)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        # setdefault is a compound read-then-write operation.  Holding the
        # lock across both steps prevents a TOCTOU race where another thread
        # inserts the key between the check and the insertion.
        with self._lock:
            return super().setdefault(key, default)

    def clear(self) -> None:
        with self._lock:
            super().clear()

    def popitem(self) -> Tuple[Any, Any]:
        with self._lock:
            return super().popitem()

    # -- read operations (consistent snapshot) ------------------------------
    # Even read operations acquire the lock.  On free-threaded Python,
    # reading a dict while another thread mutates it can yield garbage or
    # raise RuntimeError.  Guarding reads ensures a consistent view.

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return super().__contains__(key)

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        with self._lock:
            return super().get(key, default)

    def __len__(self) -> int:
        with self._lock:
            return super().__len__()

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            # Iterating directly over a dict while another thread mutates it
            # would raise RuntimeError ("dictionary changed size during
            # iteration") or, worse, silently yield inconsistent results on
            # free-threaded builds.
            #
            # To avoid this, we materialise the keys into a *snapshot* list
            # while holding the lock, then return an iterator over that
            # snapshot.  Callers can safely iterate without holding the lock
            # themselves, at the cost of a one-time O(n) copy.
            return iter(list(super().keys()))

    def keys(self) -> Any:  # type: ignore[override]
        # Returns a plain list (snapshot), not a live dict_keys view, so
        # that the caller cannot observe concurrent mutations after release.
        with self._lock:
            return list(super().keys())

    def values(self) -> Any:  # type: ignore[override]
        # Same snapshot strategy as keys() — see comment above.
        with self._lock:
            return list(super().values())

    def items(self) -> Any:  # type: ignore[override]
        # Same snapshot strategy as keys() — see comment above.
        with self._lock:
            return list(super().items())

    def copy(self) -> "ThreadSafeDict":  # type: ignore[override]
        # Returns a *new* ThreadSafeDict with its own independent lock, so
        # mutations on the copy do not affect the original (and vice versa).
        with self._lock:
            return ThreadSafeDict(super().copy())

    def __repr__(self) -> str:
        with self._lock:
            return f"{self.__class__.__name__}({super().__repr__()})"

    def __eq__(self, other: object) -> bool:
        # Lock is acquired to prevent the dict contents from changing while
        # the element-wise comparison is in progress.
        #
        # Note: if `other` is also a ThreadSafeDict, only *this* instance's
        # lock is acquired.  Acquiring both locks would risk deadlock if
        # two threads compare a == b and b == a simultaneously (lock
        # ordering violation).  The risk of `other` mutating during the
        # comparison is accepted as a pragmatic trade-off; callers needing
        # a fully atomic comparison of two ThreadSafeDicts should hold an
        # external lock.
        with self._lock:
            return super().__eq__(other)

    def __bool__(self) -> bool:
        # Uses __len__ rather than the default truth-check to ensure the
        # lock is acquired (the inherited dict.__bool__ would bypass it).
        with self._lock:
            return super().__len__() > 0

    # -- PEP 584 merge operators (Python 3.9+) -----------------------------
    # ``dict | dict``, ``dict |= dict``, and the reflected ``__ror__``
    # must be overridden so that merged results are also ThreadSafeDict
    # instances and the merge operation itself is guarded by the lock.

    def __or__(self, other: Any) -> "ThreadSafeDict":
        # self | other — the result is a *new* ThreadSafeDict with its own lock.
        with self._lock:
            merged = ThreadSafeDict(super().__or__(other))
        return merged

    def __ior__(self, other: Any) -> "ThreadSafeDict":
        # self |= other — mutates in place; must hold the lock to prevent
        # concurrent readers from seeing a partially merged state.
        with self._lock:
            super().__ior__(other)
        return self

    def __ror__(self, other: Any) -> "ThreadSafeDict":
        with self._lock:
            # other | self — `other` is a plain dict (otherwise Python would
            # have dispatched to other.__or__).  We copy it into a new
            # ThreadSafeDict then merge self's contents on top.
            merged = ThreadSafeDict(other)
            merged.update(self)
        return merged


# ---------------------------------------------------------------------------
# ThreadSafeCache (bounded LRU)
# ---------------------------------------------------------------------------


class ThreadSafeCache:
    """A bounded LRU cache protected by a ``threading.RLock``.

    **How it works:**
    Internally, an ``OrderedDict`` maintains insertion/access order.  On
    every ``get`` hit the accessed key is moved to the *end* (most recently
    used position).  On every ``put``, if the cache exceeds ``maxsize``,
    the entry at the *front* (least recently used) is evicted.  Both
    operations are O(1) amortised.

    **Thread safety:**
    Every public method acquires an ``RLock`` before touching ``_data``,
    ensuring that concurrent ``get`` / ``put`` / ``invalidate`` calls
    cannot corrupt the internal ``OrderedDict``.  The ``RLock`` (rather
    than a plain ``Lock``) is used so that, should any future method call
    another public method on the same instance, the thread will not
    deadlock.

    Args:
        maxsize: Maximum number of entries.  ``0`` means unlimited.

    Example::

        cache: ThreadSafeCache = ThreadSafeCache(maxsize=1024)
        cache.put("key", expensive_result)
        hit = cache.get("key")  # returns expensive_result
    """

    def __init__(self, maxsize: int = 1024) -> None:
        # maxsize of 0 is treated as unbounded (no eviction).
        self._maxsize = maxsize
        # OrderedDict is used because it supports O(1) move_to_end() and
        # popitem(last=False), which are the two operations needed for an
        # efficient LRU eviction policy.  A plain dict preserves insertion
        # order since Python 3.7, but does not offer move_to_end().
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        # RLock rather than Lock for the same re-entrancy safety rationale
        # as ThreadSafeDict — future methods may compose public calls.
        self._lock = threading.RLock()
        # Simple hit/miss counters for observability.  These are updated
        # under the lock so they remain consistent even on free-threaded
        # builds where plain integer increments (x += 1) are *not* atomic
        # — the read-modify-write sequence can be interleaved by another
        # thread without the GIL's implicit serialisation.
        self._hits: int = 0
        self._misses: int = 0

    # -- public API ---------------------------------------------------------

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Retrieve *key* from the cache, moving it to the MRU position.

        On a cache hit, ``move_to_end`` promotes the entry so that it
        becomes the *last* to be evicted.  This is the core of the LRU
        behaviour: frequently accessed entries survive longer.

        Args:
            key: Cache key.
            default: Value returned when *key* is absent.

        Returns:
            The cached value, or *default*.
        """
        with self._lock:
            if key in self._data:
                # Promote to most-recently-used position (end of OrderedDict).
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return default

    def put(self, key: Hashable, value: Any) -> None:
        """Insert or update *key* in the cache.

        If the cache exceeds ``maxsize``, the least-recently-used entry
        is evicted.

        Args:
            key: Cache key.
            value: Value to store.
        """
        with self._lock:
            if key in self._data:
                # Key already present — promote it to MRU and update value.
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                # New key — insert it (automatically at the end / MRU).
                self._data[key] = value
                # Evict the oldest (least recently used) entry if we have
                # exceeded the capacity.  A maxsize of 0 means unbounded.
                if self._maxsize > 0 and len(self._data) > self._maxsize:
                    # popitem(last=False) removes the *first* item, i.e.
                    # the least recently used one.
                    self._data.popitem(last=False)

    def __contains__(self, key: Hashable) -> bool:
        # Note: this is a pure membership test and intentionally does *not*
        # promote the key (unlike get), so it does not affect eviction order.
        # This is a deliberate design choice — callers who merely check
        # existence should not perturb the LRU ordering, otherwise
        # monitoring or health-check code would inadvertently keep entries
        # alive and defeat the eviction policy.
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def invalidate(self, key: Hashable) -> None:
        """Remove a single entry.  No-op if *key* is absent."""
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        # Both the data and the counters are reset atomically under the
        # same lock acquisition, so a concurrent stats() call will never
        # observe a state where the counters are reset but old entries
        # remain (or vice versa).
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of cache statistics.

        The snapshot is taken under the lock so all four values are
        mutually consistent.

        Returns:
            dict: ``{"size": ..., "maxsize": ..., "hits": ..., "misses": ...}``
        """
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
            }


# ---------------------------------------------------------------------------
# atomic_init decorator
# ---------------------------------------------------------------------------

# TypeVar for the return type of the wrapped callable in atomic_init.
_T = TypeVar("_T")


class _AtomicInitWrapper:
    """Internal descriptor that ensures a callable executes exactly once.

    This implements the **double-checked locking** pattern using a
    ``threading.Event`` as a publication barrier:

    1. **Fast path (no lock):** Check ``_done.is_set()``.  If ``True``,
       return the cached result immediately.  ``Event.is_set()`` provides
       an acquire-fence on weakly-ordered architectures (e.g. ARM64 /
       Apple Silicon), guaranteeing that the ``_result`` / ``_exc`` stores
       that preceded ``_done.set()`` are visible to the caller.

    2. **Slow path (lock held):** If the event is not set, acquire the lock
       and check the event *again* inside the critical section.  A second
       thread may have completed initialisation between the first check and
       the lock acquisition (the "double check").

    3. **Publish:** Call ``_done.set()`` only *after* storing the result.
       ``Event.set()`` provides a release-fence, so any thread that
       subsequently observes ``is_set() == True`` also sees the fully
       constructed ``_result``.

    After the first successful call, subsequent calls return the cached
    result immediately via the fast path.

    If the wrapped callable raises an exception, that exception is stored and
    re-raised on every subsequent call — the initialisation is considered
    permanently failed (no retry), which prevents repeated expensive failures.
    """

    def __init__(self, fn: Callable[[], _T]) -> None:
        self._fn = fn
        # A plain Lock (not RLock) suffices here because the wrapped
        # function is not expected to call back into __call__.  Using a
        # plain Lock is also marginally cheaper — no ownership tracking.
        self._lock = threading.Lock()
        # ``threading.Event`` acts as a publication barrier.
        #
        # Memory ordering on ARM64 / Apple Silicon:
        # ------------------------------------------
        # ARM64 has a weakly-ordered memory model.  Without a barrier, a
        # thread could observe ``_done`` as True (via a speculative load)
        # whilst still seeing stale values for ``_result``.  Event
        # internally uses a Condition (which wraps a pthread_mutex), and
        # pthread_mutex_unlock issues a ``dmb ish`` (data memory barrier,
        # inner shareable), which acts as a full release-fence.
        # Event.is_set() acquires the internal lock momentarily, providing
        # the corresponding acquire-fence.  Together, these guarantee
        # happens-before ordering: any store preceding set() is visible
        # to any thread that subsequently observes is_set() == True.
        #
        # This replaces a previous bare ``bool`` flag which lacked these
        # barrier semantics and could silently return uninitialised data
        # on weakly-ordered hardware.
        self._done = threading.Event()
        self._result: Any = None
        # If the initialisation raises, the exception is captured here and
        # re-raised on every subsequent call, preserving fail-fast semantics.
        # No retry is attempted — the rationale is that if initialisation
        # of an expensive resource (model loading, index building) fails,
        # it is likely a permanent or configuration error and retrying
        # would waste resources.
        self._exc: Optional[BaseException] = None
        # Preserve the original function's __name__, __doc__, etc., so
        # that introspection, logging, and debuggers show the wrapped
        # function's identity rather than "_AtomicInitWrapper.__call__".
        functools.update_wrapper(self, fn)

    def __call__(self) -> Any:
        """Execute the wrapped function exactly once and return its result.

        This is a once-only initialiser, not a general-purpose memoiser.
        The wrapped function must accept zero arguments.  Subsequent calls
        return the cached result from the first invocation.
        """
        # --- Fast path (lock-free) -----------------------------------------
        # This is the common-case hot path after initialisation is complete.
        # ``Event.is_set()`` provides an acquire-fence on ARM64, so when it
        # returns ``True`` we are guaranteed to see the fully written
        # ``_result`` / ``_exc`` values that were stored before ``set()``.
        #
        # On x86-64, Total Store Order (TSO) makes this fence implicit, but
        # the Event still provides correctness on weaker architectures
        # (ARM64, RISC-V) without any code changes.
        if self._done.is_set():
            if self._exc is not None:
                raise self._exc
            return self._result

        # --- Slow path (lock held) -----------------------------------------
        # Only contended during the very first call(s); once _done is set,
        # all subsequent invocations take the fast path above.
        with self._lock:
            # Double-checked locking: another thread may have won the race
            # and completed initialisation between our fast-path check above
            # and acquiring the lock here.  Without this second check we
            # would execute the function twice — defeating the "exactly once"
            # guarantee.
            #
            # The pattern is:
            #   1. Check without lock  (fast, racy but safe — only skips work)
            #   2. Acquire lock
            #   3. Check again         (authoritative — no race possible now)
            #   4. Do the work if still needed
            if self._done.is_set():
                if self._exc is not None:
                    raise self._exc
                return self._result

            try:
                self._result = self._fn()
            except BaseException as exc:
                # Store the exception so future callers receive the same
                # error without re-running the (possibly expensive) function.
                self._exc = exc
                # Signal completion *after* storing the exception, so
                # that the fast path above can see a consistent state.
                # Ordering matters: _exc must be written before set(), and
                # set()'s release-fence publishes both stores atomically
                # from the perspective of any reader that observes is_set().
                self._done.set()
                raise
            else:
                # Signal completion *after* storing the result.
                # ``Event.set()`` provides a release-fence, ensuring any
                # thread that subsequently observes ``is_set() == True``
                # also sees the fully written ``_result`` value.
                #
                # Crucially, the store to _result *must* precede set().
                # If these were reordered (possible on ARM64 without a
                # barrier), a reader could see is_set()==True but read a
                # stale _result of None.  Event.set()'s internal mutex
                # release prevents this reordering.
                self._done.set()
                return self._result

    def reset(self) -> None:
        """Reset the wrapper so the function can be called again.

        Intended for testing purposes only.  Acquires the lock to ensure
        that a concurrent __call__ does not observe a half-reset state.
        """
        with self._lock:
            # Order matters: clear the event *first* so that any thread
            # entering __call__ after this point takes the slow path and
            # blocks on the lock.  If we cleared _result first, a
            # concurrent fast-path reader could see is_set()==True but
            # read _result=None.
            self._done.clear()
            self._result = None
            self._exc = None


def atomic_init(fn: Callable[[], _T]) -> _AtomicInitWrapper:
    """Decorator ensuring *fn* is executed exactly once (thread-safe).

    On the first invocation the wrapped function runs under a
    ``threading.Lock``.  All subsequent invocations return the cached
    result without acquiring the lock (double-checked locking pattern).

    **Why is this needed on free-threaded Python?**
    Without the GIL, multiple threads can genuinely execute Python bytecode
    in parallel.  If two threads simultaneously call a lazy-initialisation
    function, both would create separate (and potentially expensive)
    instances.  ``atomic_init`` serialises the first call so exactly one
    thread performs the work, and every other thread receives the same
    cached result.

    On GIL-enabled Python, the decorator is still correct — the lock is
    simply uncontended — so callers need not check ``is_free_threaded()``
    before using it.

    This is useful for lazy, one-time initialisation of expensive
    resources (models, indices, etc.) that may be triggered from
    multiple threads simultaneously.

    Args:
        fn: The callable to wrap.

    Returns:
        An ``_AtomicInitWrapper`` instance that behaves like the original
        callable but guarantees single execution.

    Example::

        @atomic_init
        def load_model():
            return HeavyModel()

        # Both threads get the same instance, load_model() runs once.
        model = load_model()
    """
    return _AtomicInitWrapper(fn)
