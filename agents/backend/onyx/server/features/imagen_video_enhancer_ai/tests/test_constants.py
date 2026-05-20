"""
Tests for Constants
==================

Tests for application constants.
"""

import pytest
from ..core.constants import (
    DEFAULT_MAX_PARALLEL_TASKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_TTL_HOURS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    OUTPUT_DIRECTORIES
)


def test_default_values():
    """Test default constant values."""
    assert DEFAULT_MAX_PARALLEL_TASKS == 5
    assert DEFAULT_OUTPUT_DIR == "enhancer_output"
    assert DEFAULT_CACHE_TTL_HOURS == 24


def test_supported_formats():
    """Test supported file formats."""
    assert ".jpg" in SUPPORTED_IMAGE_FORMATS
    assert ".png" in SUPPORTED_IMAGE_FORMATS
    assert ".mp4" in SUPPORTED_VIDEO_FORMATS
    assert ".mov" in SUPPORTED_VIDEO_FORMATS


def test_output_directories():
    """Test output directory names."""
    assert "results" in OUTPUT_DIRECTORIES
    assert "cache" in OUTPUT_DIRECTORIES
    assert "uploads" in OUTPUT_DIRECTORIES




