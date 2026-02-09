"""
Tests for validators
"""

import pytest
from utils.validators import (
    validate_image_url,
    validate_prompt,
    validate_guidance_scale,
    validate_num_steps,
    validate_seed
)


def test_validate_image_url_valid():
    """Test valid image URL"""
    is_valid, error = validate_image_url("https://example.com/image.png")
    assert is_valid is True
    assert error is None


def test_validate_image_url_invalid():
    """Test invalid image URL"""
    is_valid, error = validate_image_url("")
    assert is_valid is False
    assert error is not None


def test_validate_prompt_valid():
    """Test valid prompt"""
    is_valid, error = validate_prompt("a red dress")
    assert is_valid is True
    assert error is None


def test_validate_prompt_empty():
    """Test empty prompt"""
    is_valid, error = validate_prompt("")
    assert is_valid is False
    assert error is not None


def test_validate_guidance_scale_valid():
    """Test valid guidance scale"""
    is_valid, error = validate_guidance_scale(50.0)
    assert is_valid is True
    assert error is None


def test_validate_guidance_scale_invalid():
    """Test invalid guidance scale"""
    is_valid, error = validate_guidance_scale(150.0)
    assert is_valid is False
    assert error is not None


def test_validate_num_steps_valid():
    """Test valid num steps"""
    is_valid, error = validate_num_steps(12)
    assert is_valid is True
    assert error is None


def test_validate_num_steps_invalid():
    """Test invalid num steps"""
    is_valid, error = validate_num_steps(150)
    assert is_valid is False
    assert error is not None


def test_validate_seed_valid():
    """Test valid seed"""
    is_valid, error = validate_seed(12345)
    assert is_valid is True
    assert error is None


def test_validate_seed_none():
    """Test None seed (valid)"""
    is_valid, error = validate_seed(None)
    assert is_valid is True
    assert error is None

