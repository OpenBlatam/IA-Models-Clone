"""
Tests refactorizados para helper formatters
Usando clases base y helpers para eliminar duplicación
"""

import pytest

from core.helpers.formatters import (
    format_number,
    format_duration,
    format_size,
    format_percentage
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestFormatNumberRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para format_number"""
    
    @pytest.mark.parametrize("number,expected_contains", [
        (1000, "1"),
        (1000000, "1"),
        (1234567, "1"),
        (0, "0"),
        (-1000, "-")
    ])
    def test_format_number(self, number, expected_contains):
        """Test de formateo de números con diferentes valores"""
        result = format_number(number)
        
        assert isinstance(result, str)
        assert expected_contains in result


class TestFormatDurationRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para format_duration"""
    
    @pytest.mark.parametrize("seconds,expected_contains", [
        (0, "0"),
        (30, "30"),
        (60, "1"),
        (120, "2"),
        (3661, "1")
    ])
    def test_format_duration(self, seconds, expected_contains):
        """Test de formateo de duración con diferentes valores"""
        result = format_duration(seconds)
        
        assert isinstance(result, str)
        assert expected_contains in result or str(seconds) in result


class TestFormatSizeRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para format_size"""
    
    @pytest.mark.parametrize("bytes_size,expected_contains", [
        (0, "0"),
        (1024, "KB"),
        (1048576, "MB"),
        (1073741824, "GB")
    ])
    def test_format_size(self, bytes_size, expected_contains):
        """Test de formateo de tamaño con diferentes valores"""
        result = format_size(bytes_size)
        
        assert isinstance(result, str)
        assert expected_contains in result


class TestFormatPercentageRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para format_percentage"""
    
    @pytest.mark.parametrize("value,total,expected_contains", [
        (0, 100, "0"),
        (50, 100, "50"),
        (100, 100, "100"),
        (25, 100, "25")
    ])
    def test_format_percentage(self, value, total, expected_contains):
        """Test de formateo de porcentaje con diferentes valores"""
        result = format_percentage(value, total)
        
        assert isinstance(result, str)
        assert expected_contains in result
        assert "%" in result



