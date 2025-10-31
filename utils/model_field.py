from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import time
from .base_types import CACHE_TTL, VALIDATION_TIMEOUT
from .validation_mixin import ValidationMixin
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Field - Onyx Integration
Field definition and validation for models.
"""

T = TypeVar('T')

@dataclass
class FieldConfig:
    """Field configuration."""
    required: bool = False
    default: Optional[Any] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    format_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelField(ValidationMixin):
    """Model field with validation and caching."""
    
    def __init__(
        self,
        field_type: Type[T],
        config: Optional[FieldConfig] = None,
        name: Optional[str] = None
    ):
        """Initialize field."""
        self.field_type = field_type
        self.config = config or FieldConfig()
        self.name = name
        self._cache = {}
        self._cache_timestamps = {}
    
    def validate(self, value: Any) -> List[str]:
        """Validate field value."""
        errors = []
        
        # Check required
        if self.config.required and value is None:
            errors.append(f"{self.name} is required")
            return errors
        
        # Skip validation if value is None and not required
        if value is None:
            return errors
        
        # Check type
        if not isinstance(value, self.field_type):
            errors.append(f"{self.name} must be of type {self.field_type.__name__}")
            return errors
        
        # Validate with timeout
        start_time = time.time()
        
        # Check length for strings
        if isinstance(value, str):
            if self.config.min_length is not None and len(value) < self.config.min_length:
                errors.append(f"{self.name} must be at least {self.config.min_length} characters")
            if self.config.max_length is not None and len(value) > self.config.max_length:
                errors.append(f"{self.name} must be at most {self.config.max_length} characters")
        
        # Check range for numbers
        if isinstance(value, (int, float)):
            if self.config.min_value is not None and value < self.config.min_value:
                errors.append(f"{self.name} must be greater than or equal to {self.config.min_value}")
            if self.config.max_value is not None and value > self.config.max_value:
                errors.append(f"{self.name} must be less than or equal to {self.config.max_value}")
        
        # Check pattern
        if self.config.pattern and isinstance(value, str):
            if not self.validate_pattern(self.name, value, self.config.pattern):
                errors.append(f"{self.name} format is invalid")
        
        # Check choices
        if self.config.choices:
            if not self.validate_choices(self.name, value, self.config.choices):
                errors.append(f"{self.name} must be one of {self.config.choices}")
        
        # Check format
        if self.config.format_type:
            if not self.validate_format(self.name, value, self.config.format_type):
                errors.append(f"{self.name} format is invalid")
        
        # Check timeout
        if time.time() - start_time > VALIDATION_TIMEOUT:
            errors.append(f"Validation timeout for {self.name}")
        
        return errors
    
    def get_default(self) -> Optional[Dict[str, Any]]:
        """Get default value."""
        if self.config.default is not None:
            return self.config.default
        
        if self.field_type == str:
            return ""
        elif self.field_type == int:
            return 0
        elif self.field_type == float:
            return 0.0
        elif self.field_type == bool:
            return False
        elif self.field_type == list:
            return []
        elif self.field_type == dict:
            return {}
        elif self.field_type == datetime:
            return datetime.utcnow()
        
        return None
    
    @lru_cache(maxsize=128)
    def to_dict(self, value: Any) -> Dict[str, Any]:
        """Convert field value to dictionary."""
        if value is None:
            return None
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, datetime):
            return value.isoformat()
        
        if isinstance(value, (list, tuple)):
            return [self.to_dict(item) for item in value]
        
        if isinstance(value, dict):
            return {k: self.to_dict(v) for k, v in value.items()}
        
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        
        return str(value)
    
    @lru_cache(maxsize=128)
    def from_dict(self, value: Any) -> Any:
        """Convert dictionary to field value."""
        if value is None:
            return None
        
        if self.field_type == datetime and isinstance(value, str):
            return datetime.fromisoformat(value)
        
        if self.field_type == list and isinstance(value, (list, tuple)):
            return [self.from_dict(item) for item in value]
        
        if self.field_type == dict and isinstance(value, dict):
            return {k: self.from_dict(v) for k, v in value.items()}
        
        return value
    
    def clear_cache(self) -> None:
        """Clear field cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.to_dict.cache_clear()
        self.from_dict.cache_clear() 