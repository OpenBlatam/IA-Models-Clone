"""
Request Validator
=================

Advanced request validation system.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error exception."""
    pass


class ValidationRule:
    """Single validation rule."""
    
    def __init__(
        self,
        field_name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        required: bool = False
    ):
        """
        Initialize validation rule.
        
        Args:
            field_name: Field name to validate
            validator: Validation function
            error_message: Error message on failure
            required: Whether field is required
        """
        self.field_name = field_name
        self.validator = validator
        self.error_message = error_message
        self.required = required
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate field in data.
        
        Args:
            data: Data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        value = data.get(self.field_name)
        
        # Check required
        if self.required and value is None:
            return False, f"Field '{self.field_name}' is required"
        
        # Skip validation if not required and value is None
        if not self.required and value is None:
            return True, None
        
        # Validate value
        try:
            if self.validator(value):
                return True, None
            return False, self.error_message
        except Exception as e:
            return False, f"Validation error for '{self.field_name}': {str(e)}"


class RequestValidator:
    """Request validator with multiple rules."""
    
    def __init__(self, name: str = "RequestValidator"):
        """
        Initialize request validator.
        
        Args:
            name: Validator name
        """
        self.name = name
        self.rules: List[ValidationRule] = []
    
    def add_rule(
        self,
        field_name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        required: bool = False
    ) -> "RequestValidator":
        """
        Add validation rule.
        
        Args:
            field_name: Field name
            validator: Validation function
            error_message: Error message
            required: Whether field is required
            
        Returns:
            Self for chaining
        """
        rule = ValidationRule(field_name, validator, error_message, required)
        self.rules.append(rule)
        return self
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against all rules.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for rule in self.rules:
            is_valid, error = rule.validate(data)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def validate_or_raise(self, data: Dict[str, Any]):
        """
        Validate data and raise exception if invalid.
        
        Args:
            data: Data to validate
            
        Raises:
            ValidationError: If validation fails
        """
        is_valid, errors = self.validate(data)
        if not is_valid:
            raise ValidationError(f"Validation failed: {', '.join(errors)}")


# Common validators
def is_not_empty(value: Any) -> bool:
    """Check if value is not empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def is_positive_number(value: Any) -> bool:
    """Check if value is a positive number."""
    try:
        num = float(value)
        return num > 0
    except (ValueError, TypeError):
        return False


def is_valid_enum(value: Any, enum_class: Type[Enum]) -> bool:
    """Check if value is a valid enum value."""
    try:
        if isinstance(value, enum_class):
            return True
        if isinstance(value, str):
            return value in [e.value for e in enum_class]
        return False
    except Exception:
        return False


def is_in_range(value: Any, min_val: float, max_val: float) -> bool:
    """Check if value is in range."""
    try:
        num = float(value)
        return min_val <= num <= max_val
    except (ValueError, TypeError):
        return False




