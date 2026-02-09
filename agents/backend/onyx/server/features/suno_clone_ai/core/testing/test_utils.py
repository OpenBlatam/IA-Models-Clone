"""
Test Utilities

Utilities for testing.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


def create_test_fixture(
    fixture_type: str,
    **kwargs
) -> Any:
    """
    Create test fixture.
    
    Args:
        fixture_type: Type of fixture
        **kwargs: Fixture parameters
        
    Returns:
        Test fixture
    """
    if fixture_type == "model":
        return mock_model(**kwargs)
    elif fixture_type == "audio":
        return mock_audio(**kwargs)
    else:
        raise ValueError(f"Unknown fixture type: {fixture_type}")


def mock_model(
    model_type: str = "musicgen",
    **kwargs
) -> MagicMock:
    """
    Create mock model.
    
    Args:
        model_type: Model type
        **kwargs: Model parameters
        
    Returns:
        Mock model
    """
    model = MagicMock()
    model.model_type = model_type
    
    # Mock generate method
    def mock_generate(prompt, **gen_kwargs):
        return mock_audio(**kwargs)
    
    model.generate = MagicMock(side_effect=mock_generate)
    
    return model


def mock_audio(
    duration: float = 5.0,
    sample_rate: int = 44100,
    channels: int = 2
) -> np.ndarray:
    """
    Create mock audio.
    
    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate
        channels: Number of channels
        
    Returns:
        Mock audio array
    """
    samples = int(duration * sample_rate)
    audio = np.random.randn(channels, samples).astype(np.float32)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio


def assert_audio_valid(
    audio: np.ndarray,
    sample_rate: int = 44100,
    min_duration: float = 1.0,
    max_duration: float = 60.0
) -> None:
    """
    Assert audio is valid.
    
    Args:
        audio: Audio array
        sample_rate: Expected sample rate
        min_duration: Minimum duration
        max_duration: Maximum duration
        
    Raises:
        AssertionError: If audio is invalid
    """
    assert audio is not None, "Audio is None"
    assert isinstance(audio, np.ndarray), "Audio must be numpy array"
    assert audio.ndim >= 1, "Audio must have at least 1 dimension"
    
    # Check duration
    duration = audio.shape[-1] / sample_rate
    assert min_duration <= duration <= max_duration, \
        f"Duration {duration} not in range [{min_duration}, {max_duration}]"
    
    # Check values
    assert np.all(np.isfinite(audio)), "Audio contains NaN or Inf"
    assert np.max(np.abs(audio)) <= 1.0, "Audio values exceed [-1, 1] range"



