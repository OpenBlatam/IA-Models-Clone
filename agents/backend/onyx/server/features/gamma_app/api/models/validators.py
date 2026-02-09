"""
Common Validators
Reusable field validators for models
"""

from typing import Optional, List
from pydantic import field_validator

def sanitize_string_field(v: str) -> str:
    """Sanitize string field by stripping whitespace"""
    if not isinstance(v, str):
        return v
    sanitized = v.strip()
    if not sanitized and v:
        raise ValueError('Field cannot contain only whitespace')
    return sanitized

def sanitize_optional_string_field(v: Optional[str]) -> Optional[str]:
    """Sanitize optional string field"""
    if v is None:
        return None
    sanitized = v.strip()
    return sanitized if sanitized else None

class StringFieldValidators:
    """Mixin class with common string field validators"""
    
    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Sanitize name field"""
        return sanitize_string_field(v)
    
    @field_validator('name', 'description')
    @classmethod
    def sanitize_string_fields(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize string fields"""
        return sanitize_optional_string_field(v)







