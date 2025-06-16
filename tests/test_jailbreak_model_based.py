import sys
import types
import pytest
from unittest import mock

# Test 1: Lazy import behavior

def test_lazy_import_does_not_require_heavy_deps():
    """
    Importing the checks module should not require torch, transformers, or sklearn unless model-based classifier is used.
    """
    with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "sklearn": None}):
        import importlib
        import nemoguardrails.library.jailbreak_detection.model_based.checks as checks
        # Just importing and calling unrelated functions should not raise ImportError
        assert hasattr(checks, "initialize_model")

# Test 2: Model-based classifier instantiation requires dependencies

def test_model_based_classifier_imports(monkeypatch):
    """
    Instantiating JailbreakClassifier should require sklearn and pickle, and use SnowflakeEmbed which requires torch/transformers.
    """
    # Mock dependencies
    fake_rf = mock.MagicMock()
    fake_embed = mock.MagicMock(return_value=[0.0])
    fake_pickle = types.SimpleNamespace(load=mock.MagicMock(return_value=fake_rf))
    fake_snowflake = mock.MagicMock(return_value=fake_embed)

    monkeypatch.setitem(sys.modules, "sklearn.ensemble", types.SimpleNamespace(RandomForestClassifier=mock.MagicMock()))
    monkeypatch.setitem(sys.modules, "pickle", fake_pickle)
    monkeypatch.setitem(sys.modules, "torch", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", mock.MagicMock())

    # Patch SnowflakeEmbed to avoid real model loading
    import nemoguardrails.library.jailbreak_detection.model_based.models as models
    monkeypatch.setattr(models, "SnowflakeEmbed", fake_snowflake)

    # Create a fake model file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        # Should not raise
        classifier = models.JailbreakClassifier(tmp.name)
        assert classifier is not None
        # Should be callable
        result = classifier("test")
        assert isinstance(result, tuple)

# Test 3: Error if dependencies missing when instantiating model-based classifier

def test_model_based_classifier_missing_deps(monkeypatch):
    """
    If sklearn is missing, instantiating JailbreakClassifier should raise ImportError.
    """
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    import nemoguardrails.library.jailbreak_detection.model_based.models as models
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        with pytest.raises(ImportError):
            models.JailbreakClassifier(tmp.name)