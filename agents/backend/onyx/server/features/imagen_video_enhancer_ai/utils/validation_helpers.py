"""
Validation Helpers
==================

Common validation utilities and patterns.
"""

from typing import Any, Callable, List, Optional
from pathlib import Path


class ValidationRule:
    """Single validation rule."""
    
    def __init__(
        self,
        validator: Callable[[Any], bool],
        error_message: str
    ):
        """
        Initialize validation rule.
        
        Args:
            validator: Validation function
            error_message: Error message on failure
        """
        self.validator = validator
        self.error_message = error_message
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate value.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.validator(value):
            return True, None
        return False, self.error_message


class ValidationChain:
    """Chain of validation rules."""
    
    def __init__(self):
        """Initialize validation chain."""
        self.rules: List[ValidationRule] = []
    
    def add_rule(
        self,
        validator: Callable[[Any], bool],
        error_message: str
    ) -> "ValidationChain":
        """
        Add validation rule.
        
        Args:
            validator: Validation function
            error_message: Error message
            
        Returns:
            Self for chaining
        """
        self.rules.append(ValidationRule(validator, error_message))
        return self
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate value against all rules.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for rule in self.rules:
            is_valid, error = rule.validate(value)
            if not is_valid:
                return False, error
        return True, None


# Common validators
def is_positive(value: Any) -> bool:
    """Check if value is positive."""
    return isinstance(value, (int, float)) and value > 0


def is_non_negative(value: Any) -> bool:
    """Check if value is non-negative."""
    return isinstance(value, (int, float)) and value >= 0


def is_in_range(value: Any, min_val: float, max_val: float) -> bool:
    """Check if value is in range."""
    return isinstance(value, (int, float)) and min_val <= value <= max_val


def is_valid_path(value: Any) -> bool:
    """Check if value is a valid path."""
    try:
        path = Path(value)
        return True
    except Exception:
        return False


def is_valid_email(value: Any) -> bool:
    """Check if value is a valid email."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return isinstance(value, str) and bool(re.match(pattern, value))


def is_valid_url(value: Any) -> bool:
    """Check if value is a valid URL."""
    import re
    pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return isinstance(value, str) and bool(pattern.match(value))


def has_min_length(value: Any, min_length: int) -> bool:
    """Check if string has minimum length."""
    return isinstance(value, str) and len(value) >= min_length


def has_max_length(value: Any, max_length: int) -> bool:
    """Check if string has maximum length."""
    return isinstance(value, str) and len(value) <= max_length


def is_one_of(value: Any, allowed_values: List[Any]) -> bool:
    """Check if value is one of allowed values."""
    return value in allowed_values




