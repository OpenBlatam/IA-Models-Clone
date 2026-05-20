"""
Tests for Validation Engine
============================
"""

import pytest
from ..core.validation import ValidationEngine, ValidationRule, ValidationError


@pytest.fixture
def validation_engine():
    """Create validation engine for testing."""
    return ValidationEngine()


def test_add_validation_rule(validation_engine):
    """Test adding a validation rule."""
    def min_length_validator(value):
        return len(value) >= 5
    
    rule = ValidationRule(
        rule_id="min_length",
        validator=min_length_validator,
        error_message="Value must be at least 5 characters"
    )
    
    validation_engine.add_rule("test_field", rule)
    
    assert "test_field" in validation_engine.rules
    assert len(validation_engine.rules["test_field"]) == 1


def test_validate_success(validation_engine):
    """Test successful validation."""
    def min_length_validator(value):
        return len(value) >= 5
    
    rule = ValidationRule(
        rule_id="min_length",
        validator=min_length_validator,
        error_message="Too short"
    )
    
    validation_engine.add_rule("username", rule)
    
    result = validation_engine.validate("username", "testuser")
    
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_failure(validation_engine):
    """Test validation failure."""
    def min_length_validator(value):
        return len(value) >= 5
    
    rule = ValidationRule(
        rule_id="min_length",
        validator=min_length_validator,
        error_message="Too short"
    )
    
    validation_engine.add_rule("username", rule)
    
    result = validation_engine.validate("username", "test")
    
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_validate_multiple_fields(validation_engine):
    """Test validating multiple fields."""
    def email_validator(value):
        return "@" in value
    
    def min_length_validator(value):
        return len(value) >= 3
    
    validation_engine.add_rule("email", ValidationRule("email", email_validator, "Invalid email"))
    validation_engine.add_rule("name", ValidationRule("min_length", min_length_validator, "Too short"))
    
    data = {
        "email": "test@example.com",
        "name": "John"
    }
    
    result = validation_engine.validate_all(data)
    
    assert result.is_valid is True or len(result.errors) == 0


