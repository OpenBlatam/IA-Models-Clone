"""
Tests for validation utilities
"""

import pytest
from utils.validation import (
    sanitize_string,
    validate_email,
    validate_phone,
    validate_port_code,
    validate_positive_number,
    validate_transportation_mode,
    validate_country_code,
)
from utils.exceptions import ValidationError


def test_sanitize_string():
    """Test string sanitization"""
    assert sanitize_string("  test  ") == "test"
    assert sanitize_string("test\x00string") == "teststring"
    
    with pytest.raises(ValidationError):
        sanitize_string("test", max_length=3)


def test_validate_email():
    """Test email validation"""
    assert validate_email("test@example.com") == "test@example.com"
    assert validate_email("TEST@EXAMPLE.COM") == "test@example.com"
    
    with pytest.raises(ValidationError):
        validate_email("invalid-email")
    
    with pytest.raises(ValidationError):
        validate_email("")


def test_validate_phone():
    """Test phone validation"""
    assert validate_phone("+1234567890") == "+1234567890"
    assert validate_phone("(123) 456-7890") == "(123) 456-7890"
    
    with pytest.raises(ValidationError):
        validate_phone("123")
    
    with pytest.raises(ValidationError):
        validate_phone("")


def test_validate_port_code():
    """Test port code validation"""
    assert validate_port_code("mxver") == "MXVER"
    assert validate_port_code("USNYC") == "USNYC"
    
    with pytest.raises(ValidationError):
        validate_port_code("INVALID")
    
    with pytest.raises(ValidationError):
        validate_port_code("")


def test_validate_positive_number():
    """Test positive number validation"""
    assert validate_positive_number(100) == 100.0
    assert validate_positive_number(100.5) == 100.5
    
    with pytest.raises(ValidationError):
        validate_positive_number(-10)
    
    with pytest.raises(ValidationError):
        validate_positive_number(50, min_value=100)
    
    with pytest.raises(ValidationError):
        validate_positive_number(200, max_value=100)


def test_validate_transportation_mode():
    """Test transportation mode validation"""
    assert validate_transportation_mode("air") == "air"
    assert validate_transportation_mode("MARITIME") == "maritime"
    
    with pytest.raises(ValidationError):
        validate_transportation_mode("invalid")
    
    with pytest.raises(ValidationError):
        validate_transportation_mode("")


def test_validate_country_code():
    """Test country code validation"""
    assert validate_country_code("mx") == "MX"
    assert validate_country_code("USA") == "USA"
    
    with pytest.raises(ValidationError):
        validate_country_code("INVALIDCODE")
    
    with pytest.raises(ValidationError):
        validate_country_code("")

