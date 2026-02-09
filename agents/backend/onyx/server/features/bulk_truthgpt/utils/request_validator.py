"""
Request Validator
================

Advanced request validation with type checking, sanitization, and security.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RequestValidator:
    """Advanced request validator with security checks."""
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        self.sanitizers: Dict[str, Callable] = {}
        self.max_string_length = 10000
        self.max_array_length = 1000
    
    def register_validator(self, field: str, validator: Callable):
        """Register validator for a field."""
        if field not in self.validators:
            self.validators[field] = []
        self.validators[field].append(validator)
    
    def register_sanitizer(self, field: str, sanitizer: Callable):
        """Register sanitizer for a field."""
        self.sanitizers[field] = sanitizer
    
    def validate_string(
        self,
        value: Any,
        min_length: int = 1,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allow_empty: bool = False
    ) -> tuple[bool, Optional[str]]:
        """Validate string value."""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if not allow_empty and not value.strip():
            return False, "String cannot be empty"
        
        if len(value) < min_length:
            return False, f"String must be at least {min_length} characters"
        
        max_len = max_length or self.max_string_length
        if len(value) > max_len:
            return False, f"String must be at most {max_len} characters"
        
        if pattern and not re.match(pattern, value):
            return False, f"String does not match required pattern"
        
        return True, None
    
    def validate_integer(
        self,
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate integer value."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return False, "Value must be an integer"
        
        if min_value is not None and value < min_value:
            return False, f"Value must be at least {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value must be at most {max_value}"
        
        return True, None
    
    def validate_float(
        self,
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate float value."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False, "Value must be a number"
        
        if min_value is not None and value < min_value:
            return False, f"Value must be at least {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value must be at most {max_value}"
        
        return True, None
    
    def validate_array(
        self,
        value: Any,
        min_length: int = 0,
        max_length: Optional[int] = None,
        item_validator: Optional[Callable] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate array value."""
        if not isinstance(value, (list, tuple)):
            return False, "Value must be an array"
        
        if len(value) < min_length:
            return False, f"Array must have at least {min_length} items"
        
        max_len = max_length or self.max_array_length
        if len(value) > max_len:
            return False, f"Array must have at most {max_len} items"
        
        if item_validator:
            for i, item in enumerate(value):
                is_valid, error = item_validator(item)
                if not is_valid:
                    return False, f"Item {i} invalid: {error}"
        
        return True, None
    
    def sanitize_string(self, value: str, max_length: Optional[int] = None) -> str:
        """Sanitize string value."""
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Limit length
        if max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def sanitize_json(self, value: Any) -> Any:
        """Sanitize JSON value recursively."""
        if isinstance(value, dict):
            return {k: self.sanitize_json(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.sanitize_json(item) for item in value]
        elif isinstance(value, str):
            return self.sanitize_string(value)
        else:
            return value
    
    def validate_request(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]]
    ) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate request data against schema.
        
        Returns:
            (is_valid, error_message, sanitized_data)
        """
        sanitized = {}
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Check required
            if rules.get("required", False) and value is None:
                return False, f"Field '{field}' is required", {}
            
            if value is None:
                continue
            
            # Type validation
            field_type = rules.get("type", "string")
            
            if field_type == "string":
                is_valid, error = self.validate_string(
                    value,
                    min_length=rules.get("min_length", 1),
                    max_length=rules.get("max_length"),
                    pattern=rules.get("pattern"),
                    allow_empty=rules.get("allow_empty", False)
                )
                if not is_valid:
                    return False, f"Field '{field}': {error}", {}
                
                # Sanitize
                sanitized[field] = self.sanitize_string(value, rules.get("max_length"))
            
            elif field_type == "integer":
                is_valid, error = self.validate_integer(
                    value,
                    min_value=rules.get("min_value"),
                    max_value=rules.get("max_value")
                )
                if not is_valid:
                    return False, f"Field '{field}': {error}", {}
                sanitized[field] = int(value)
            
            elif field_type == "float":
                is_valid, error = self.validate_float(
                    value,
                    min_value=rules.get("min_value"),
                    max_value=rules.get("max_value")
                )
                if not is_valid:
                    return False, f"Field '{field}': {error}", {}
                sanitized[field] = float(value)
            
            elif field_type == "array":
                is_valid, error = self.validate_array(
                    value,
                    min_length=rules.get("min_length", 0),
                    max_length=rules.get("max_length"),
                    item_validator=rules.get("item_validator")
                )
                if not is_valid:
                    return False, f"Field '{field}': {error}", {}
                sanitized[field] = list(value)
            
            elif field_type == "object":
                if not isinstance(value, dict):
                    return False, f"Field '{field}' must be an object", {}
                sanitized[field] = self.sanitize_json(value)
            
            else:
                sanitized[field] = value
        
        return True, None, sanitized

# Global instance
request_validator = RequestValidator()
































