"""
Tests for Validators
====================
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from imagen_video_enhancer_ai.utils.validators import (
    FileValidator,
    ParameterValidator,
    ValidationError
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def test_validate_image_file_success(temp_dir):
    """Test successful image file validation."""
    test_file = temp_dir / "test.jpg"
    test_file.write_bytes(b"fake image data")
    
    # Should not raise
    FileValidator.validate_image_file(str(test_file), max_size_mb=10)


def test_validate_image_file_not_found(temp_dir):
    """Test image file validation with missing file."""
    with pytest.raises(ValidationError, match="not found"):
        FileValidator.validate_image_file(str(temp_dir / "nonexistent.jpg"))


def test_validate_image_file_wrong_format(temp_dir):
    """Test image file validation with wrong format."""
    test_file = temp_dir / "test.txt"
    test_file.write_bytes(b"not an image")
    
    with pytest.raises(ValidationError, match="Unsupported image format"):
        FileValidator.validate_image_file(str(test_file))


def test_validate_video_file_success(temp_dir):
    """Test successful video file validation."""
    test_file = temp_dir / "test.mp4"
    test_file.write_bytes(b"fake video data")
    
    # Should not raise
    FileValidator.validate_video_file(str(test_file), max_size_mb=100)


def test_get_file_type():
    """Test file type detection."""
    assert FileValidator.get_file_type("test.jpg") == "image"
    assert FileValidator.get_file_type("test.png") == "image"
    assert FileValidator.get_file_type("test.mp4") == "video"
    assert FileValidator.get_file_type("test.avi") == "video"
    
    with pytest.raises(ValidationError):
        FileValidator.get_file_type("test.unknown")


def test_validate_enhancement_type():
    """Test enhancement type validation."""
    # Valid types
    for valid_type in ["general", "sharpness", "colors", "denoise", "upscale", "restore"]:
        ParameterValidator.validate_enhancement_type(valid_type)
    
    # Invalid type
    with pytest.raises(ValidationError):
        ParameterValidator.validate_enhancement_type("invalid")


def test_validate_scale_factor():
    """Test scale factor validation."""
    # Valid factors
    for factor in [1, 2, 4, 8]:
        ParameterValidator.validate_scale_factor(factor)
    
    # Invalid factors
    with pytest.raises(ValidationError):
        ParameterValidator.validate_scale_factor(0)
    
    with pytest.raises(ValidationError):
        ParameterValidator.validate_scale_factor(10)


def test_validate_priority():
    """Test priority validation."""
    # Valid priorities
    ParameterValidator.validate_priority(0)
    ParameterValidator.validate_priority(5)
    ParameterValidator.validate_priority(10)
    
    # Invalid priorities
    with pytest.raises(ValidationError):
        ParameterValidator.validate_priority(-1)
    
    with pytest.raises(ValidationError):
        ParameterValidator.validate_priority("not a number")




