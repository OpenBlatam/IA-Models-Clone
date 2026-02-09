"""
Tests for validators.
"""

import pytest
from pathlib import Path
import tempfile
import os

from piel_mejorador_ai_sam3.core.validators import (
    ParameterValidator,
    FileValidator,
    ValidationError
)


class TestParameterValidator:
    """Tests for ParameterValidator."""
    
    def test_validate_enhancement_level_valid(self):
        """Test valid enhancement levels."""
        for level in ["low", "medium", "high", "ultra"]:
            ParameterValidator.validate_enhancement_level(level)
    
    def test_validate_enhancement_level_invalid(self):
        """Test invalid enhancement levels."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_enhancement_level("invalid")
    
    def test_validate_realism_level_valid(self):
        """Test valid realism levels."""
        ParameterValidator.validate_realism_level(0.0)
        ParameterValidator.validate_realism_level(0.5)
        ParameterValidator.validate_realism_level(1.0)
        ParameterValidator.validate_realism_level(None)
    
    def test_validate_realism_level_invalid(self):
        """Test invalid realism levels."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_realism_level(1.5)
        
        with pytest.raises(ValidationError):
            ParameterValidator.validate_realism_level(-0.1)
    
    def test_validate_file_type_valid(self):
        """Test valid file types."""
        ParameterValidator.validate_file_type("image")
        ParameterValidator.validate_file_type("video")
    
    def test_validate_file_type_invalid(self):
        """Test invalid file types."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_file_type("invalid")
    
    def test_validate_file_path_exists(self):
        """Test file path validation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            ParameterValidator.validate_file_path(temp_path, must_exist=True)
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_path_not_exists(self):
        """Test file path validation when file doesn't exist."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_file_path("/nonexistent/file.jpg", must_exist=True)
    
    def test_validate_priority_valid(self):
        """Test valid priorities."""
        ParameterValidator.validate_priority(0)
        ParameterValidator.validate_priority(10)
        ParameterValidator.validate_priority(100)
    
    def test_validate_priority_invalid(self):
        """Test invalid priorities."""
        with pytest.raises(ValidationError):
            ParameterValidator.validate_priority(-1)
        
        with pytest.raises(ValidationError):
            ParameterValidator.validate_priority("invalid")


class TestFileValidator:
    """Tests for FileValidator."""
    
    def test_validate_image_file_valid(self):
        """Test valid image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name
        
        try:
            FileValidator.validate_image_file(temp_path, max_size_mb=1)
        finally:
            os.unlink(temp_path)
    
    def test_validate_image_file_invalid_extension(self):
        """Test invalid image extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                FileValidator.validate_image_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_video_file_valid(self):
        """Test valid video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            temp_path = f.name
        
        try:
            FileValidator.validate_video_file(temp_path, max_size_mb=1)
        finally:
            os.unlink(temp_path)




