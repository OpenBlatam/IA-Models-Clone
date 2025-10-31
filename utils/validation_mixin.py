from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
import time
import re
from .base_types import VALIDATION_TIMEOUT
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Validation Mixin - Onyx Integration
Validation functionality for models.
"""

class ValidationMixin:
    """Mixin for validation functionality."""
    
    _lazy_validation: bool = True
    _validation_timeout: float = VALIDATION_TIMEOUT
    
    def validate_with_timeout(self, validation_func: callable, *args, **kwargs) -> List[str]:
        """Run validation with timeout."""
        start_time = time.time()
        errors = []
        
        try:
            result = validation_func(*args, **kwargs)
            if time.time() - start_time > self._validation_timeout:
                return errors
            return result
        except Exception as e:
            errors.append(str(e))
            return errors
    
    def validate_required(self, field_name: str, value: Any) -> List[str]:
        """Validate required field."""
        errors = []
        if value is None:
            errors.append(f"{field_name} is required")
        return errors
    
    def validate_type(self, field_name: str, value: Any, expected_type: Type) -> List[str]:
        """Validate field type."""
        errors = []
        if value is not None and not isinstance(value, expected_type):
            errors.append(f"{field_name} must be of type {expected_type.__name__}")
        return errors
    
    def validate_length(self, field_name: str, value: str, min_length: Optional[int] = None, max_length: Optional[int] = None) -> List[str]:
        """Validate string length."""
        errors = []
        if value is not None:
            if min_length is not None and len(value) < min_length:
                errors.append(f"{field_name} must be at least {min_length} characters")
            if max_length is not None and len(value) > max_length:
                errors.append(f"{field_name} must be at most {max_length} characters")
        return errors
    
    def validate_range(self, field_name: str, value: Any, min_value: Optional[Any] = None, max_value: Optional[Any] = None) -> List[str]:
        """Validate numeric range."""
        errors = []
        if value is not None:
            if min_value is not None and value < min_value:
                errors.append(f"{field_name} must be greater than or equal to {min_value}")
            if max_value is not None and value > max_value:
                errors.append(f"{field_name} must be less than or equal to {max_value}")
        return errors
    
    def validate_pattern(self, field_name: str, value: str, pattern: str) -> List[str]:
        """Validate string pattern."""
        errors = []
        if value is not None and not re.match(pattern, value):
            errors.append(f"{field_name} format is invalid")
        return errors
    
    def validate_choices(self, field_name: str, value: Any, choices: List[Any]) -> List[str]:
        """Validate value against choices."""
        errors = []
        if value is not None and value not in choices:
            errors.append(f"{field_name} must be one of {choices}")
        return errors
    
    def validate_format(self, field_name: str, value: str, format_type: str) -> List[str]:
        """Validate string format."""
        errors = []
        if value is not None:
            if format_type == "email" and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value):
                errors.append(f"{field_name} must be a valid email address")
            elif format_type == "url" and not re.match(r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$", value):
                errors.append(f"{field_name} must be a valid URL")
            elif format_type == "date" and not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                errors.append(f"{field_name} must be a valid date (YYYY-MM-DD)")
            elif format_type == "datetime" and not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z?$", value):
                errors.append(f"{field_name} must be a valid datetime (ISO 8601)")
        return errors
    
    def validate_relationship(self, field_name: str, value: Any) -> List[str]:
        """Validate relationship field."""
        errors = []
        if value is not None:
            if not isinstance(value, list):
                errors.append(f"{field_name} must be a list of IDs")
            else:
                for item in value:
                    if not isinstance(item, (str, int)):
                        errors.append(f"Invalid ID type in {field_name}")
        return errors 