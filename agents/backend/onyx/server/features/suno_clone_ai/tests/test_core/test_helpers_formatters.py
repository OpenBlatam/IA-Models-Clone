"""
Tests para helper formatters
"""

import pytest

from core.helpers.formatters import (
    format_number,
    format_duration,
    format_size,
    format_percentage
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestFormatNumber(BaseServiceTestCase, StandardTestMixin):
    """Tests para format_number"""
    
    @pytest.mark.parametrize("number,expected", [
        (1000, "1,000"),
        (1000000, "1,000,000"),
        (1234567, "1,234,567"),
        (0, "0"),
        (-1000, "-1,000")
    ])
    def test_format_number(self, number, expected):
        """Test de formateo de números"""
        result = format_number(number)
        
        assert isinstance(result, str)
        # Verificar que tiene comas para números grandes
        if abs(number) >= 1000:
            assert ',' in result


class TestFormatDuration(BaseServiceTestCase, StandardTestMixin):
    """Tests para format_duration"""
    
    @pytest.mark.parametrize("seconds,expected_contains", [
        (0, "0"),
        (30, "30"),
        (60, "1"),
        (120, "2"),
        (3661, "1")
    ])
    def test_format_duration(self, seconds, expected_contains):
        """Test de formateo de duración"""
        result = format_duration(seconds)
        
        assert isinstance(result, str)
        assert expected_contains in result or str(seconds) in result


class TestFormatSize(BaseServiceTestCase, StandardTestMixin):
    """Tests para format_size"""
    
    @pytest.mark.parametrize("bytes_size,expected_contains", [
        (0, "0"),
        (1024, "KB"),
        (1048576, "MB"),
        (1073741824, "GB")
    ])
    def test_format_size(self, bytes_size, expected_contains):
        """Test de formateo de tamaño"""
        result = format_size(bytes_size)
        
        assert isinstance(result, str)
        assert expected_contains in result


class TestFormatPercentage(BaseServiceTestCase, StandardTestMixin):
    """Tests para format_percentage"""
    
    @pytest.mark.parametrize("value,total,expected_contains", [
        (0, 100, "0"),
        (50, 100, "50"),
        (100, 100, "100"),
        (25, 100, "25")
    ])
    def test_format_percentage(self, value, total, expected_contains):
        """Test de formateo de porcentaje"""
        result = format_percentage(value, total)
        
        assert isinstance(result, str)
        assert expected_contains in result
        assert "%" in result



