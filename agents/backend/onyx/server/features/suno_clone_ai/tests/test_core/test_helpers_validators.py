"""
Tests para helper validators
"""

import pytest

from core.helpers.validators import (
    validate_range,
    validate_type,
    validate_not_none
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestValidateRange(BaseServiceTestCase, StandardTestMixin):
    """Tests para validate_range"""
    
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
        """Test de validación de rango"""
        if should_pass:
            result = validate_range(value, min_val, max_val)
            assert result is True
        else:
            with pytest.raises(ValueError):
                validate_range(value, min_val, max_val)


class TestValidateType(BaseServiceTestCase, StandardTestMixin):
    """Tests para validate_type"""
    
    @pytest.mark.parametrize("value,expected_type,should_pass", [
        (5, int, True),
        ("test", str, True),
        ([1, 2], list, True),
        (5, str, False),
        ("test", int, False)
    ])
    def test_validate_type(self, value, expected_type, should_pass):
        """Test de validación de tipo"""
        if should_pass:
            result = validate_type(value, expected_type)
            assert result is True
        else:
            with pytest.raises(TypeError):
                validate_type(value, expected_type)


class TestValidateNotNone(BaseServiceTestCase, StandardTestMixin):
    """Tests para validate_not_none"""
    
    @pytest.mark.parametrize("value,should_pass", [
        (0, True),
        ("", True),
        ([], True),
        (False, True),
        (None, False)
    ])
    def test_validate_not_none(self, value, should_pass):
        """Test de validación de no None"""
        if should_pass:
            result = validate_not_none(value)
            assert result is True
        else:
            with pytest.raises(ValueError):
                validate_not_none(value)



