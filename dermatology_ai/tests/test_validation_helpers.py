"""
Validation Testing Helpers
Specialized helpers for validation testing
"""

from typing import Any, Dict, List, Optional, Callable, Type
from unittest.mock import Mock
import re
from datetime import datetime


class ValidationTestHelpers:
    """Helpers for validation testing"""
    
    @staticmethod
    def create_mock_validator(
        validation_rules: Optional[Dict[str, Callable]] = None,
        default_valid: bool = True
    ) -> Mock:
        """Create mock validator with custom rules"""
        rules = validation_rules or {}
        validator = Mock()
        
        def validate_side_effect(data: Dict[str, Any]) -> tuple[bool, List[str]]:
            errors = []
            for field, rule in rules.items():
                if field in data:
                    try:
                        if not rule(data[field]):
                            errors.append(f"{field} validation failed")
                    except Exception as e:
                        errors.append(f"{field} validation error: {str(e)}")
            
            if default_valid and len(errors) == 0:
                return True, []
            return len(errors) == 0, errors
        
        validator.validate = Mock(side_effect=validate_side_effect)
        validator.validate_field = Mock(return_value=(True, []))
        return validator
    
    @staticmethod
    def assert_validation_passes(
        validator: Mock,
        data: Dict[str, Any],
        expected_errors: Optional[List[str]] = None
    ):
        """Assert validation passes"""
        result = validator.validate(data)
        
        if isinstance(result, tuple):
            is_valid, errors = result
            assert is_valid, f"Validation failed with errors: {errors}"
            if expected_errors:
                assert set(errors) == set(expected_errors), \
                    f"Errors {errors} do not match expected {expected_errors}"
        else:
            assert result, "Validation failed"
    
    @staticmethod
    def assert_validation_fails(
        validator: Mock,
        data: Dict[str, Any],
        expected_errors: Optional[List[str]] = None
    ):
        """Assert validation fails"""
        result = validator.validate(data)
        
        if isinstance(result, tuple):
            is_valid, errors = result
            assert not is_valid, "Validation should have failed"
            if expected_errors:
                assert any(
                    any(expected in error for expected in expected_errors)
                    for error in errors
                ), f"Expected errors {expected_errors} not found in {errors}"
        else:
            assert not result, "Validation should have failed"


class FieldValidationHelpers:
    """Helpers for field-level validation"""
    
    @staticmethod
    def create_email_validator() -> Callable:
        """Create email validator"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return lambda email: bool(re.match(email_pattern, email))
    
    @staticmethod
    def create_url_validator() -> Callable:
        """Create URL validator"""
        url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        return lambda url: bool(re.match(url_pattern, url))
    
    @staticmethod
    def create_phone_validator() -> Callable:
        """Create phone validator"""
        phone_pattern = r'^\+?[\d\s\-()]+$'
        return lambda phone: bool(re.match(phone_pattern, phone)) and len(re.sub(r'[\s\-()]', '', phone)) >= 10
    
    @staticmethod
    def create_date_validator(date_format: str = "%Y-%m-%d") -> Callable:
        """Create date validator"""
        def validate_date(date_str: str) -> bool:
            try:
                datetime.strptime(date_str, date_format)
                return True
            except ValueError:
                return False
        return validate_date
    
    @staticmethod
    def create_range_validator(min_value: float, max_value: float) -> Callable:
        """Create range validator"""
        return lambda value: min_value <= value <= max_value
    
    @staticmethod
    def create_length_validator(min_length: int, max_length: int) -> Callable:
        """Create length validator"""
        return lambda value: min_length <= len(str(value)) <= max_length


class SchemaValidationHelpers:
    """Helpers for schema validation"""
    
    @staticmethod
    def assert_schema_valid(
        data: Dict[str, Any],
        schema: Dict[str, Type],
        required_fields: Optional[List[str]] = None
    ):
        """Assert data matches schema"""
        required_fields = required_fields or []
        
        # Check required fields
        for field in required_fields:
            assert field in data, f"Required field {field} is missing"
        
        # Check types
        for field, expected_type in schema.items():
            if field in data:
                assert isinstance(data[field], expected_type), \
                    f"Field {field} has type {type(data[field])}, expected {expected_type}"
    
    @staticmethod
    def assert_schema_invalid(
        data: Dict[str, Any],
        schema: Dict[str, Type],
        required_fields: Optional[List[str]] = None
    ):
        """Assert data does not match schema"""
        required_fields = required_fields or []
        
        # Check if any required field is missing
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return  # Invalid due to missing fields
        
        # Check if any type is wrong
        for field, expected_type in schema.items():
            if field in data and not isinstance(data[field], expected_type):
                return  # Invalid due to wrong type
        
        # If we get here, schema is valid, so assertion should fail
        assert False, "Schema should be invalid but appears valid"


class CustomValidationHelpers:
    """Helpers for custom validation scenarios"""
    
    @staticmethod
    def create_conditional_validator(
        condition: Callable[[Dict[str, Any]], bool],
        validator: Callable[[Dict[str, Any]], tuple[bool, List[str]]]
    ) -> Callable:
        """Create validator that applies conditionally"""
        def validate(data: Dict[str, Any]) -> tuple[bool, List[str]]:
            if condition(data):
                return validator(data)
            return True, []
        return validate
    
    @staticmethod
    def create_cross_field_validator(
        fields: List[str],
        validator: Callable[[Dict[str, Any]], bool]
    ) -> Callable:
        """Create validator that validates across multiple fields"""
        def validate(data: Dict[str, Any]) -> tuple[bool, List[str]]:
            if all(field in data for field in fields):
                if validator(data):
                    return True, []
                return False, [f"Cross-field validation failed for {fields}"]
            return False, [f"Missing fields for cross-field validation: {fields}"]
        return validate


# Convenience exports
create_mock_validator = ValidationTestHelpers.create_mock_validator
assert_validation_passes = ValidationTestHelpers.assert_validation_passes
assert_validation_fails = ValidationTestHelpers.assert_validation_fails

create_email_validator = FieldValidationHelpers.create_email_validator
create_url_validator = FieldValidationHelpers.create_url_validator
create_phone_validator = FieldValidationHelpers.create_phone_validator
create_date_validator = FieldValidationHelpers.create_date_validator
create_range_validator = FieldValidationHelpers.create_range_validator
create_length_validator = FieldValidationHelpers.create_length_validator

assert_schema_valid = SchemaValidationHelpers.assert_schema_valid
assert_schema_invalid = SchemaValidationHelpers.assert_schema_invalid

create_conditional_validator = CustomValidationHelpers.create_conditional_validator
create_cross_field_validator = CustomValidationHelpers.create_cross_field_validator



