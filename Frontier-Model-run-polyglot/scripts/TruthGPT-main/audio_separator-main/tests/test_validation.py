"""
Tests for validation utilities.
Refactored to use constants and fixtures.
"""

import pytest
import numpy as np
from pathlib import Path
from audio_separator.utils.validation_utils import (
    validate_audio_file,
    validate_output_format,
    validate_sample_rate,
    validate_num_sources,
    validate_audio_array,
    validate_output_dir
)
from audio_separator.exceptions import (
    AudioValidationError,
    AudioFormatError
)
from audio_separator.separator.constants import DEFAULT_SAMPLE_RATE


def test_validate_sample_rate(valid_sample_rates, invalid_sample_rates):
    """Test sample rate validation."""
    # Valid rates
    for rate in valid_sample_rates:
        validate_sample_rate(rate)
    
    # Invalid rates
    for rate in invalid_sample_rates:
        with pytest.raises(AudioValidationError):
            validate_sample_rate(rate)


def test_validate_num_sources(valid_num_sources, invalid_num_sources):
    """Test number of sources validation."""
    # Valid
    for num in valid_num_sources:
        validate_num_sources(num)
    
    # Invalid
    for num in invalid_num_sources:
        with pytest.raises(AudioValidationError):
            validate_num_sources(num)


def test_validate_audio_array():
    """Test audio array validation."""
    # Valid arrays
    validate_audio_array(np.array([1, 2, 3]))
    validate_audio_array(np.array([[1, 2], [3, 4]]))
    
    # Invalid arrays
    with pytest.raises(AudioValidationError):
        validate_audio_array([])
    
    with pytest.raises(AudioValidationError):
        validate_audio_array(np.array([]))
    
    with pytest.raises(AudioValidationError):
        validate_audio_array(np.array([np.nan, 1, 2]))
    
    with pytest.raises(AudioValidationError):
        validate_audio_array(np.array([np.inf, 1, 2]))


def test_validate_output_format(valid_output_formats, invalid_output_formats):
    """Test output format validation."""
    # Valid formats
    for fmt in valid_output_formats:
        validate_output_format(fmt)
    
    # Invalid formats
    for fmt in invalid_output_formats:
        with pytest.raises(AudioFormatError):
            validate_output_format(fmt)


def test_validate_output_dir(test_output_dir):
    """Test output directory validation."""
    # Should create directory
    validate_output_dir(test_output_dir, create=True)
    assert test_output_dir.exists()
    assert test_output_dir.is_dir()
    
    # Should not raise if exists
    validate_output_dir(test_output_dir, create=True)

