"""
Tests refactorizados para helper validators
Usando clases base y helpers para eliminar duplicación
"""

import pytest

from core.helpers.validators import (
    validate_range,
    validate_type,
    validate_not_none
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestValidateRangeRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_range"""
    
    @pytest.mark.parametrize("value,min_val,max_val,should_pass", [
        (5, 0, 10, True),
        (0, 0, 10, True),
        (10, 0, 10, True),
        (-1, 0, 10, False),
        (11, 0, 10, False),
        (5, None, 10, True),
        (5, 0, None, True)
    ])
    def test_validate_range(self, value, min_val, max_val, should_pass):
        """Test de validación de rango con diferentes configuraciones"""
        is_valid, error = validate_range(value, min_val, max_val)
        
        if should_pass:
            assert is_valid is True
            assert error is None
        else:
            assert is_valid is False
            assert error is not None


class TestValidateTypeRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_type"""
    
    @pytest.mark.parametrize("value,expected_type,should_pass", [
        (5, int, True),
        ("test", str, True),
        ([1, 2], list, True),
        (5, str, False),
        ("test", int, False)
    ])
    def test_validate_type(self, value, expected_type, should_pass):
        """Test de validación de tipo con diferentes tipos"""
        is_valid, error = validate_type(value, expected_type)
        
        if should_pass:
            assert is_valid is True
            assert error is None
        else:
            assert is_valid is False
            assert error is not None


class TestValidateNotNoneRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_not_none"""
    
    @pytest.mark.parametrize("value,should_pass", [
        (0, True),
        ("", True),
        ([], True),
        (False, True),
        (None, False)
    ])
    def test_validate_not_none(self, value, should_pass):
        """Test de validación de no None con diferentes valores"""
        is_valid, error = validate_not_none(value)
        
        if should_pass:
            assert is_valid is True
            assert error is None
        else:
            assert is_valid is False
            assert error is not None



