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
from .model_field import ModelField, FieldConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Model Schema - Onyx Integration
Schema definition and validation for models.
"""

T = TypeVar('T')

@dataclass
class SchemaConfig:
    """Schema configuration."""
    strict: bool = True
    allow_extra: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelSchema(ValidationMixin):
    """Model schema with validation and caching."""
    
    def __init__(
        self,
        fields: Dict[str, ModelField],
        config: Optional[SchemaConfig] = None,
        name: Optional[str] = None
    ):
        """Initialize schema."""
        self.fields = fields
        self.config = config or SchemaConfig()
        self.name = name
        self._cache = {}
        self._cache_timestamps = {}
    
    def validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate schema data."""
        errors = []
        
        # Validate with timeout
        start_time = time.time()
        
        # Check required fields
        for field_name, field in self.fields.items():
            if field.config.required and field_name not in data:
                errors.append(f"{field_name} is required")
        
        # Validate fields
        for field_name, value in data.items():
            if field_name not in self.fields:
                if not self.config.allow_extra:
                    errors.append(f"Unknown field: {field_name}")
                continue
            
            field = self.fields[field_name]
            field_errors = field.validate(value)
            errors.extend(field_errors)
        
        # Check timeout
        if time.time() - start_time > VALIDATION_TIMEOUT:
            errors.append(f"Validation timeout for schema {self.name}")
        
        return errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all fields."""
        return {
            field_name: field.get_default()
            for field_name, field in self.fields.items()
        }
    
    @lru_cache(maxsize=128)
    def to_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert schema data to dictionary."""
        result = {}
        
        for field_name, value in data.items():
            if field_name in self.fields:
                field = self.fields[field_name]
                result[field_name] = field.to_dict(value)
            elif self.config.allow_extra:
                result[field_name] = value
        
        return result
    
    @lru_cache(maxsize=128)
    def from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary to schema data."""
        result = {}
        
        for field_name, value in data.items():
            if field_name in self.fields:
                field = self.fields[field_name]
                result[field_name] = field.from_dict(value)
            elif self.config.allow_extra:
                result[field_name] = value
        
        return result
    
    def get_field(self, field_name: str) -> Optional[ModelField]:
        """Get field by name."""
        return self.fields.get(field_name)
    
    def get_fields(self) -> Dict[str, ModelField]:
        """Get all fields."""
        return self.fields.copy()
    
    def add_field(self, field_name: str, field: ModelField) -> None:
        """Add a field."""
        if field_name in self.fields:
            raise ValueError(f"Field {field_name} already exists")
        
        self.fields[field_name] = field
        self.clear_cache()
    
    def remove_field(self, field_name: str) -> None:
        """Remove a field."""
        if field_name not in self.fields:
            raise ValueError(f"Field {field_name} does not exist")
        
        del self.fields[field_name]
        self.clear_cache()
    
    def clear_cache(self) -> None:
        """Clear schema cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.to_dict.cache_clear()
        self.from_dict.cache_clear() 