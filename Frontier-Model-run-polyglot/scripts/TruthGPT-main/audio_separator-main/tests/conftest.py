"""
Pytest configuration and shared fixtures.
Refactored to use constants and improve test organization.
"""

import pytest
import numpy as np
from pathlib import Path
from audio_separator.separator.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_OUTPUT_FORMAT
)
from audio_separator.utils.constants import SUPPORTED_AUDIO_FORMATS


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

TEST_COMPONENT_NAME = "TestComponent"
TEST_ERROR_CODE = "TEST_ERROR"
TEST_AUDIO_FILE = "test.wav"
TEST_AUDIO_DURATION = 1.0  # seconds


# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_rate():
    """Fixture for default sample rate."""
    return DEFAULT_SAMPLE_RATE


@pytest.fixture
def model_type():
    """Fixture for default model type."""
    return DEFAULT_MODEL_TYPE


@pytest.fixture
def output_format():
    """Fixture for default output format."""
    return DEFAULT_OUTPUT_FORMAT


@pytest.fixture
def test_audio_array(sample_rate):
    """Fixture for test audio array."""
    duration = TEST_AUDIO_DURATION
    samples = int(sample_rate * duration)
    return np.random.randn(samples).astype(np.float32)


@pytest.fixture
def test_audio_file(tmp_path):
    """Fixture for test audio file path."""
    return tmp_path / TEST_AUDIO_FILE


@pytest.fixture
def test_output_dir(tmp_path):
    """Fixture for test output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def valid_sample_rates():
    """Fixture for valid sample rates."""
    return [8000, 16000, 22050, DEFAULT_SAMPLE_RATE, 48000, 96000]


@pytest.fixture
def invalid_sample_rates():
    """Fixture for invalid sample rates."""
    return [0, -1, -100, 100, 1000, 50000]


@pytest.fixture
def valid_num_sources():
    """Fixture for valid number of sources."""
    return [1, 2, 4, 8, 10]


@pytest.fixture
def invalid_num_sources():
    """Fixture for invalid number of sources."""
    return [0, -1, -10, 11, 100]


@pytest.fixture
def valid_output_formats():
    """Fixture for valid output formats."""
    return ["wav", "mp3", "flac", "ogg"]


@pytest.fixture
def invalid_output_formats():
    """Fixture for invalid output formats."""
    return ["invalid", "txt", "pdf", "xyz"]


@pytest.fixture
def supported_audio_extensions():
    """Fixture for supported audio extensions."""
    return list(SUPPORTED_AUDIO_FORMATS)

