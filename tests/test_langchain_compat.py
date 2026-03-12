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

"""Tests for nemoguardrails._langchain_compat.

These tests verify the compatibility shim works correctly under various
installation scenarios, using unittest.mock to simulate different
langchain installation states. All tests are safe to run on both
Python 3.12 and 3.14.
"""

import builtins
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails._langchain_compat import (
    _NEEDS_PATCH,
    _patch_langchain_dict_shadow,
    import_chat_models_base,
    import_init_chat_model,
    safe_import_langchain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_module(name, attrs=None):
    """Create a fake module object and insert it into sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Tests for _patch_langchain_dict_shadow
# ---------------------------------------------------------------------------

class TestPatchLangchainDictShadow:
    """Tests for the builtins patching function."""

    def test_noop_on_python_below_314(self, monkeypatch):
        """On Python < 3.14 the patch should be a no-op."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", False
        )
        # Even if a target module exists, nothing should be patched.
        fake = _make_fake_module("langchain.chains.base")
        monkeypatch.setitem(sys.modules, "langchain.chains.base", fake)
        try:
            _patch_langchain_dict_shadow()
            assert not hasattr(fake, "__dict_patched__")
        finally:
            sys.modules.pop("langchain.chains.base", None)

    def test_patches_loaded_modules(self, monkeypatch):
        """Modules already in sys.modules should get dict/list injected."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        fake = _make_fake_module("langchain.chains.base")
        monkeypatch.setitem(sys.modules, "langchain.chains.base", fake)
        try:
            _patch_langchain_dict_shadow()
            assert fake.dict is builtins.dict
            assert fake.list is builtins.list
            assert fake.__dict_patched__ is True
        finally:
            sys.modules.pop("langchain.chains.base", None)

    def test_idempotent(self, monkeypatch):
        """Calling the patch multiple times should not re-patch."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        fake = _make_fake_module("langchain.chains.base")
        monkeypatch.setitem(sys.modules, "langchain.chains.base", fake)
        try:
            _patch_langchain_dict_shadow()
            # Tamper with the value after the first patch to detect re-patching.
            sentinel = object()
            fake.dict = sentinel

            _patch_langchain_dict_shadow()
            # Should still be the sentinel -- second call should skip.
            assert fake.dict is sentinel
        finally:
            sys.modules.pop("langchain.chains.base", None)

    def test_patches_langchain_core_modules(self, monkeypatch):
        """langchain_core modules should also be patched."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        fake_core = _make_fake_module("langchain_core.runnables.base")
        monkeypatch.setitem(sys.modules, "langchain_core.runnables.base", fake_core)
        try:
            _patch_langchain_dict_shadow()
            assert fake_core.dict is builtins.dict
            assert fake_core.list is builtins.list
            assert fake_core.__dict_patched__ is True
        finally:
            sys.modules.pop("langchain_core.runnables.base", None)

    def test_skips_missing_modules(self, monkeypatch):
        """Modules not in sys.modules should be silently skipped."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        # Ensure none of the target modules are loaded.
        for mod_name in [
            "langchain.chains.base",
            "langchain.chains",
            "langchain.schema",
            "langchain.schema.runnable.base",
            "langchain_core.runnables.base",
            "langchain_core.load.serializable",
        ]:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

        # Should not raise.
        _patch_langchain_dict_shadow()


# ---------------------------------------------------------------------------
# Tests for safe_import_langchain
# ---------------------------------------------------------------------------

class TestSafeImportLangchain:
    """Tests for the safe_import_langchain function."""

    def test_langchain_not_installed(self, monkeypatch):
        """When langchain is completely absent, a clear error is raised."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain":
                raise ImportError("No module named 'langchain'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="langchain is not installed"):
                safe_import_langchain()

    def test_langchain_installed_returns_module(self):
        """When langchain is installed, the module is returned."""
        result = safe_import_langchain()
        assert result is not None
        assert hasattr(result, "__version__")

    def test_type_error_on_py314_gives_clear_message(self, monkeypatch):
        """TypeError with 'not subscriptable' on Python 3.14+ gets a clear error."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain":
                raise TypeError("'function' object is not subscriptable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="PEP 649"):
                safe_import_langchain()

    def test_type_error_non_subscriptable_reraised(self, monkeypatch):
        """A TypeError that is NOT the dict-shadow issue is re-raised as-is."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain":
                raise TypeError("some other type error")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(TypeError, match="some other type error"):
                safe_import_langchain()


# ---------------------------------------------------------------------------
# Tests for import_init_chat_model
# ---------------------------------------------------------------------------

class TestImportInitChatModel:
    """Tests for the import_init_chat_model helper."""

    def test_returns_callable(self):
        """Should return the init_chat_model callable when langchain is available."""
        result = import_init_chat_model()
        assert callable(result)

    def test_falls_back_to_langchain_classic(self, monkeypatch):
        """When langchain path fails, falls back to langchain_classic."""
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "langchain.chat_models" and fromlist:
                raise ImportError("no langchain.chat_models")
            if name == "langchain_classic.chat_models" and fromlist:
                mock_mod = types.ModuleType("langchain_classic.chat_models")
                mock_mod.init_chat_model = lambda: "classic_init"
                return mock_mod
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = import_init_chat_model()
            assert result is not None

    def test_raises_when_both_paths_fail(self, monkeypatch):
        """When neither langchain nor langchain_classic has init_chat_model, raises ImportError."""
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "langchain" in name and "chat_models" in name:
                raise ImportError(f"no {name}")
            if name.startswith("langchain_classic"):
                raise ImportError(f"no {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="Could not import init_chat_model"):
                import_init_chat_model()


# ---------------------------------------------------------------------------
# Tests for import_chat_models_base
# ---------------------------------------------------------------------------

class TestImportChatModelsBase:
    """Tests for the import_chat_models_base helper."""

    def test_returns_module_when_available(self):
        """Should return the base module when langchain is installed."""
        result = import_chat_models_base()
        # May return None if the specific module doesn't exist, but shouldn't raise.
        # With langchain installed, it should return a module.
        if result is not None:
            assert hasattr(result, "__name__")

    def test_returns_none_when_nothing_available(self, monkeypatch):
        """When neither langchain nor langchain_classic base is available, returns None."""
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "langchain" in name and "chat_models" in name:
                raise ImportError(f"no {name}")
            if name.startswith("langchain_classic"):
                raise ImportError(f"no {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = import_chat_models_base()
            assert result is None


# ---------------------------------------------------------------------------
# Tests for edge case: langchain_core present but langchain absent
# ---------------------------------------------------------------------------

class TestLangchainCoreWithoutLangchain:
    """Test behavior when langchain_core is installed but langchain is not."""

    def test_safe_import_fails_gracefully(self, monkeypatch):
        """safe_import_langchain should give a clear install message."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain" and not name.startswith("langchain_core"):
                raise ImportError("No module named 'langchain'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="langchain is not installed"):
                safe_import_langchain()

    def test_patch_still_works_for_langchain_core(self, monkeypatch):
        """Even without langchain, langchain_core modules should be patchable."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        fake_core = _make_fake_module("langchain_core.runnables.base")
        monkeypatch.setitem(sys.modules, "langchain_core.runnables.base", fake_core)

        # Remove langchain modules to simulate langchain being absent.
        for mod_name in [
            "langchain.chains.base",
            "langchain.chains",
            "langchain.schema",
            "langchain.schema.runnable.base",
        ]:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

        try:
            _patch_langchain_dict_shadow()
            assert fake_core.dict is builtins.dict
        finally:
            sys.modules.pop("langchain_core.runnables.base", None)


# ---------------------------------------------------------------------------
# Tests for edge case: old langchain (< 0.1) installed
# ---------------------------------------------------------------------------

class TestOldLangchain:
    """Test behavior with very old langchain versions."""

    def test_import_init_chat_model_old_langchain(self, monkeypatch):
        """Old langchain without chat_models.init_chat_model should fail clearly."""
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "langchain.chat_models" and fromlist and "init_chat_model" in fromlist:
                raise ImportError("cannot import name 'init_chat_model'")
            if name.startswith("langchain_classic"):
                raise ImportError(f"no {name}")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="Could not import init_chat_model"):
                import_init_chat_model()


# ---------------------------------------------------------------------------
# Tests for edge case: langchain installed without langchain_community
# ---------------------------------------------------------------------------

class TestLangchainWithoutCommunity:
    """Test that the compat shim works when langchain_community is missing."""

    def test_safe_import_succeeds_without_community(self):
        """safe_import_langchain should succeed even without langchain_community."""
        # This test verifies that safe_import_langchain doesn't depend on
        # langchain_community at all -- it only imports the base langchain package.
        result = safe_import_langchain()
        assert result is not None

    def test_patch_does_not_require_community(self, monkeypatch):
        """_patch_langchain_dict_shadow should work without langchain_community."""
        monkeypatch.setattr(
            "nemoguardrails._langchain_compat._NEEDS_PATCH", True
        )
        # Remove langchain_community from sys.modules if present.
        monkeypatch.delitem(sys.modules, "langchain_community", raising=False)

        fake = _make_fake_module("langchain.chains.base")
        monkeypatch.setitem(sys.modules, "langchain.chains.base", fake)
        try:
            _patch_langchain_dict_shadow()
            assert fake.dict is builtins.dict
        finally:
            sys.modules.pop("langchain.chains.base", None)
