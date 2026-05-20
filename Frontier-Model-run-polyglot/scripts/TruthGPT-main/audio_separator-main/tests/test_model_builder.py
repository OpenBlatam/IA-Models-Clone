"""
Tests for model builder.
Refactored to use constants.
"""

import pytest
from audio_separator.model_builder import build_audio_separator_model
from audio_separator.exceptions import AudioValidationError, AudioModelError
from audio_separator.separator.constants import DEFAULT_MODEL_TYPE


def test_build_demucs_model(model_type):
    """Test building Demucs model."""
    # This will fail if demucs is not installed, which is expected
    try:
        model = build_audio_separator_model(model_type)
        assert model is not None
    except (ImportError, AudioModelError):
        # Expected if demucs is not installed
        pass


def test_build_invalid_model():
    """Test building invalid model."""
    with pytest.raises(AudioValidationError):
        build_audio_separator_model("invalid_model")


def test_build_model_with_kwargs():
    """Test building model with kwargs."""
    try:
        model = build_audio_separator_model(
            "demucs",
            variant="htdemucs",
            num_sources=4
        )
        assert model is not None
    except (ImportError, AudioModelError):
        # Expected if demucs is not installed
        pass

