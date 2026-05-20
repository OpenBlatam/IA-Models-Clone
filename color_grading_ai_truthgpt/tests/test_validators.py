"""
Tests for validators.
"""

import pytest
from pathlib import Path
import tempfile

from ..core.validators import (
    ParameterValidator,
    MediaValidator,
    ConfigValidator
)
from ..core.exceptions import InvalidParametersError, MediaNotFoundError


class TestParameterValidator:
    """Tests for ParameterValidator."""
    
    def test_validate_brightness_valid(self):
        """Test valid brightness."""
        params = {"brightness": 0.5}
        result = ParameterValidator.validate_color_params(params)
        assert result["brightness"] == 0.5
    
    def test_validate_brightness_invalid_high(self):
        """Test invalid brightness (too high)."""
        params = {"brightness": 2.0}
        with pytest.raises(InvalidParametersError):
            ParameterValidator.validate_color_params(params)
    
    def test_validate_brightness_invalid_low(self):
        """Test invalid brightness (too low)."""
        params = {"brightness": -2.0}
        with pytest.raises(InvalidParametersError):
            ParameterValidator.validate_color_params(params)
    
    def test_validate_contrast_valid(self):
        """Test valid contrast."""
        params = {"contrast": 1.5}
        result = ParameterValidator.validate_color_params(params)
        assert result["contrast"] == 1.5
    
    def test_validate_contrast_invalid(self):
        """Test invalid contrast."""
        params = {"contrast": 5.0}
        with pytest.raises(InvalidParametersError):
            ParameterValidator.validate_color_params(params)
    
    def test_validate_color_balance_valid(self):
        """Test valid color balance."""
        params = {"color_balance": {"r": 0.1, "g": 0.0, "b": -0.1}}
        result = ParameterValidator.validate_color_params(params)
        assert result["color_balance"]["r"] == 0.1
        assert result["color_balance"]["g"] == 0.0
        assert result["color_balance"]["b"] == -0.1
    
    def test_validate_color_balance_invalid(self):
        """Test invalid color balance."""
        params = {"color_balance": {"r": 1.0}}  # Too high
        with pytest.raises(InvalidParametersError):
            ParameterValidator.validate_color_params(params)


class TestMediaValidator:
    """Tests for MediaValidator."""
    
    def test_validate_file_path_exists(self):
        """Test validating existing file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name
        
        try:
            result = MediaValidator.validate_file_path(temp_path)
            assert isinstance(result, Path)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_file_path_not_exists(self):
        """Test validating non-existent file."""
        with pytest.raises(MediaNotFoundError):
            MediaValidator.validate_file_path("/nonexistent/file.txt")
    
    def test_validate_image_file_valid(self):
        """Test validating valid image file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            temp_path = f.name
        
        try:
            result = MediaValidator.validate_image_file(temp_path)
            assert isinstance(result, Path)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_image_file_invalid_format(self):
        """Test validating invalid image format."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            temp_path = f.name
        
        try:
            with pytest.raises(MediaNotFoundError):
                MediaValidator.validate_image_file(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_get_file_size_mb(self):
        """Test getting file size in MB."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024 * 1024)  # 1 MB
            temp_path = f.name
        
        try:
            size = MediaValidator.get_file_size_mb(temp_path)
            assert 0.9 < size < 1.1  # Allow small margin
        finally:
            Path(temp_path).unlink()
    
    def test_validate_file_size_valid(self):
        """Test validating file size (valid)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = MediaValidator.validate_file_size(temp_path, max_size_mb=100.0)
            assert result is True
        finally:
            Path(temp_path).unlink()
    
    def test_validate_file_size_too_large(self):
        """Test validating file size (too large)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 10 * 1024 * 1024)  # 10 MB
            temp_path = f.name
        
        try:
            with pytest.raises(InvalidParametersError):
                MediaValidator.validate_file_size(temp_path, max_size_mb=1.0)
        finally:
            Path(temp_path).unlink()




