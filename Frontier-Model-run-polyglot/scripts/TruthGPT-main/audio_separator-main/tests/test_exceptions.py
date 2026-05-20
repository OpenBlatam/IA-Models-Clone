"""
Tests for exception handling.
Refactored to use constants.
"""

import pytest
from audio_separator.exceptions import (
    AudioSeparatorError,
    AudioProcessingError,
    AudioFormatError,
    AudioModelError,
    AudioValidationError,
    AudioIOError,
    AudioInitializationError,
    AudioConfigurationError
)
from tests.conftest import TEST_COMPONENT_NAME, TEST_ERROR_CODE, TEST_AUDIO_FILE


def test_base_exception():
    """Test base exception."""
    error = AudioSeparatorError(
        "Test error",
        component=TEST_COMPONENT_NAME,
        error_code=TEST_ERROR_CODE
    )
    expected_str = f"Test error | Component: {TEST_COMPONENT_NAME} | Code: {TEST_ERROR_CODE}"
    assert str(error) == expected_str
    assert error.component == TEST_COMPONENT_NAME
    assert error.error_code == TEST_ERROR_CODE


def test_exception_hierarchy():
    """Test exception hierarchy."""
    assert issubclass(AudioProcessingError, AudioSeparatorError)
    assert issubclass(AudioFormatError, AudioSeparatorError)
    assert issubclass(AudioModelError, AudioSeparatorError)
    assert issubclass(AudioValidationError, AudioSeparatorError)
    assert issubclass(AudioIOError, AudioSeparatorError)
    assert issubclass(AudioInitializationError, AudioSeparatorError)
    assert issubclass(AudioConfigurationError, AudioSeparatorError)


def test_exception_with_details():
    """Test exception with details."""
    error = AudioProcessingError(
        "Processing failed",
        component=TEST_COMPONENT_NAME,
        error_code="PROCESS_FAILED",
        details={"file": TEST_AUDIO_FILE, "reason": "invalid format"}
    )
    assert error.details == {"file": TEST_AUDIO_FILE, "reason": "invalid format"}

