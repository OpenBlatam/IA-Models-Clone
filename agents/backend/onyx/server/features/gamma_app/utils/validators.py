"""
Pydantic Validators and Custom Validators
"""

from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime
import re

from .service import UtilityService


class BaseValidator(BaseModel):
    """Base validator with common functionality"""
    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True
    )


class EmailValidator(BaseValidator):
    """Email validator"""
    email: str = Field(..., description="Email address")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if not UtilityService.validate_email(v):
            raise ValueError('Invalid email format')
        return v.lower().strip()


class URLValidator(BaseValidator):
    """URL validator"""
    url: str = Field(..., description="URL")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not UtilityService.validate_url(v):
            raise ValueError('Invalid URL format')
        return v


class PasswordValidator(BaseValidator):
    """Password validator"""
    password: str = Field(..., min_length=8, max_length=128)
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v


class DateTimeRangeValidator(BaseValidator):
    """Date time range validator"""
    start: datetime
    end: datetime
    
    @model_validator(mode='after')
    def validate_range(self) -> 'DateTimeRangeValidator':
        if self.start and self.end and self.start >= self.end:
            raise ValueError('Start time must be before end time')
        return self


class PaginationValidator(BaseValidator):
    """Pagination validator"""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset"""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit"""
        return self.page_size


class SortValidator(BaseValidator):
    """Sort validator"""
    sort_by: Optional[str] = None
    sort_order: str = Field("asc", pattern="^(asc|desc)$")
    
    @field_validator('sort_by')
    @classmethod
    def validate_sort_by(cls, v: Optional[str]) -> Optional[str]:
        if v:
            # Sanitize sort field to prevent SQL injection
            v = UtilityService.sanitize_string(v)
            # Only allow alphanumeric and underscore
            if not re.match(r'^[a-zA-Z0-9_]+$', v):
                raise ValueError('Invalid sort field')
        return v


class SearchQueryValidator(BaseValidator):
    """Search query validator"""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(10, ge=1, le=100)
    
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return UtilityService.sanitize_string(v)


class IDValidator(BaseValidator):
    """ID validator"""
    id: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        # UUID or alphanumeric ID
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Invalid ID format')
        return v


class ListValidator(BaseValidator):
    """List validator with min/max items"""
    items: List[Any] = Field(..., min_length=1)
    max_items: Optional[int] = None
    
    @model_validator(mode='after')
    def validate_items_count(self) -> 'ListValidator':
        if self.max_items and len(self.items) > self.max_items:
            raise ValueError(f'List must have at most {self.max_items} items')
        return self


# Custom validators as functions
def validate_uuid(value: str) -> str:
    """Validate UUID format"""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    if not uuid_pattern.match(value):
        raise ValueError('Invalid UUID format')
    return value


def validate_phone(value: str) -> str:
    """Validate phone number"""
    # Basic phone validation (international format)
    phone_pattern = re.compile(r'^\+?[1-9]\d{1,14}$')
    if not phone_pattern.match(value.replace(' ', '').replace('-', '')):
        raise ValueError('Invalid phone number format')
    return value


def validate_slug(value: str) -> str:
    """Validate URL slug"""
    slug_pattern = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
    if not slug_pattern.match(value):
        raise ValueError('Invalid slug format (use lowercase letters, numbers, and hyphens)')
    return value.lower()


def validate_json_string(value: str) -> dict:
    """Validate and parse JSON string"""
    import json
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON: {e}')


def validate_enum(value: Any, enum_class: type) -> Any:
    """Validate enum value"""
    try:
        return enum_class(value)
    except ValueError:
        raise ValueError(f'Invalid enum value. Must be one of: {[e.value for e in enum_class]}')
