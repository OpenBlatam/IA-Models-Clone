from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Literal, Tuple, ClassVar
from dataclasses import dataclass, field
from datetime import datetime
from ..base import ModelField, ValidationMixin, CacheMixin, EventMixin, IndexMixin, PermissionMixin, StatusMixin
from ..base_types import (
from ..model_schema import ModelSchema, SchemaConfig
from .color import BrandKitColor
from .typography import BrandKitTypography
from .voice import BrandKitVoice
import math
from functools import lru_cache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Brand Kit Schema Component - Onyx Integration
Optimized component for managing brand kit schema with advanced features.
"""
    CACHE_TTL, VALIDATION_TIMEOUT,
    ModelId, ModelKey, ModelValue,
    ValidationType, CacheType, EventType,
    StatusType, CategoryType, PermissionType
)

T = TypeVar('T')

@dataclass
class BrandKitSchema:
    """Brand Kit Schema Component with advanced features and optimizations"""
    name: str
    description: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'timeout': 0.3,  # Reduced timeout
        'rules': {
            'required': ['name', 'colors', 'typography', 'voice'],
            'optional': ['description', 'logo', 'mission', 'vision', 'values', 'target_audience']
        }
    })
    cache: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'ttl': 1800,  # Reduced TTL
        'prefix': 'brand_kit:schema',
        'strategy': 'lru'  # Using LRU cache strategy
    })
    events: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'types': ['schema_created', 'schema_updated', 'schema_deleted'],
        'notify': True,
        'batch': True  # Enable event batching
    })
    index: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'fields': ['name', 'fields'],
        'type': 'hash',
        'strategy': 'lazy'  # Lazy indexing
    })
    permissions: Dict[str, Any] = field(default_factory=lambda: {
        'roles': ['admin', 'designer'],
        'actions': ['create', 'read', 'update', 'delete'],
        'cache': True  # Cache permissions
    })
    status: Dict[str, Any] = field(default_factory=lambda: {
        'active': True,
        'archived': False
    })
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = '1.0.0'
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Class-level caches
    _schema_cache: ClassVar[Dict[str, Dict[str, Any]]] = {}
    _validation_cache: ClassVar[Dict[str, bool]] = {}
    _field_cache: ClassVar[Dict[str, Dict[str, Any]]] = {}

    def __post_init__(self) -> Any:
        """Initialize schema field with optimized validation and caching"""
        self.schema_field = ModelField(
            name=self.name,
            value=self.fields,
            required=True,
            validation=self.validation,
            cache=self.cache,
            events=self.events,
            index=self.index,
            permissions=self.permissions,
            status=self.status
        )

    @lru_cache(maxsize=128)
    def get_data(self) -> Dict[str, Any]:
        """Get schema data with optimized caching"""
        cache_key = f"brand_kit:schema:{self.name}"
        cached_data = self.schema_field.get_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        data = {
            'name': self.name,
            'description': self.description,
            'fields': self.fields,
            'validation': self.validation,
            'cache': self.cache,
            'events': self.events,
            'index': self.index,
            'permissions': self.permissions,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'metadata': self.metadata
        }
        
        self.schema_field.set_cache(cache_key, data)
        return data

    def update(self, **kwargs) -> None:
        """Update schema properties with optimized cache clearing"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.utcnow()
        self.schema_field.clear_cache(f"brand_kit:schema:{self.name}")
        self.get_data.cache_clear()  # Clear LRU cache

    @lru_cache(maxsize=32)
    def validate_colors(self, colors: List[Dict[str, Any]]) -> bool:
        """Validate colors with optimized caching"""
        cache_key = f"colors:{hash(str(colors))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        required_fields = ['name', 'hex', 'category']
        valid = all(
            all(field in color for field in required_fields)
            for color in colors
        )
        
        self._validation_cache[cache_key] = valid
        return valid

    @lru_cache(maxsize=32)
    def validate_typography(self, typography: Dict[str, Any]) -> bool:
        """Validate typography with optimized caching"""
        cache_key = f"typography:{hash(str(typography))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        required_fields = ['name', 'font_family', 'style', 'category']
        valid = all(field in typography for field in required_fields)
        
        self._validation_cache[cache_key] = valid
        return valid

    @lru_cache(maxsize=32)
    def validate_voice(self, voice: Dict[str, Any]) -> bool:
        """Validate voice with optimized caching"""
        cache_key = f"voice:{hash(str(voice))}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]

        required_fields = ['name', 'tone', 'style', 'personality_traits']
        valid = all(field in voice for field in required_fields)
        
        self._validation_cache[cache_key] = valid
        return valid

    @lru_cache(maxsize=32)
    def get_field_schema(self, field_name: str) -> Dict[str, Any]:
        """Get field schema with optimized caching"""
        cache_key = f"field:{field_name}"
        if cache_key in self._field_cache:
            return self._field_cache[cache_key]

        field_schema = self.fields.get(field_name, {})
        self._field_cache[cache_key] = field_schema
        return field_schema

    @lru_cache(maxsize=32)
    def get_required_fields(self) -> List[str]:
        """Get required fields with optimized caching"""
        return self.validation.get('rules', {}).get('required', [])

    @lru_cache(maxsize=32)
    def get_optional_fields(self) -> List[str]:
        """Get optional fields with optimized caching"""
        return self.validation.get('rules', {}).get('optional', [])

    @lru_cache(maxsize=32)
    def get_field_validation_rules(self, field_name: str) -> Dict[str, Any]:
        """Get field validation rules with optimized caching"""
        field_schema = self.get_field_schema(field_name)
        return field_schema.get('validation', {})

    @lru_cache(maxsize=32)
    def get_field_cache_settings(self, field_name: str) -> Dict[str, Any]:
        """Get field cache settings with optimized caching"""
        field_schema = self.get_field_schema(field_name)
        return field_schema.get('cache', {})

    @lru_cache(maxsize=32)
    def get_field_permissions(self, field_name: str) -> Dict[str, Any]:
        """Get field permissions with optimized caching"""
        field_schema = self.get_field_schema(field_name)
        return field_schema.get('permissions', {})

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> 'BrandKitSchema':
        """Create schema from data with optimized initialization"""
        return cls(**data)

    def __del__(self) -> Any:
        """Cleanup caches on deletion"""
        self.get_data.cache_clear()
        self.validate_colors.cache_clear()
        self.validate_typography.cache_clear()
        self.validate_voice.cache_clear()
        self.get_field_schema.cache_clear()
        self.get_required_fields.cache_clear()
        self.get_optional_fields.cache_clear()
        self.get_field_validation_rules.cache_clear()
        self.get_field_cache_settings.cache_clear()
        self.get_field_permissions.cache_clear()

class BrandKitSchema(ModelSchema):
    """Schema for brand kit with Onyx integration."""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        cache: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        index: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None
    ):
        """Initialize brand kit schema."""
        super().__init__(
            name=name,
            description=description or "Brand kit schema",
            fields=fields or {},
            validation=validation or {
                "type": ValidationType.SCHEMA,
                "timeout": VALIDATION_TIMEOUT,
                "rules": {
                    "colors": "required",
                    "typography": "required",
                    "voice": "required",
                    "message": "Brand kit must have colors, typography, and voice"
                }
            },
            cache=cache or {
                "type": CacheType.SCHEMA,
                "ttl": CACHE_TTL,
                "key_prefix": "brand_kit"
            },
            events=events or {
                "type": EventType.SCHEMA,
                "on_change": True,
                "notify": True
            },
            index=index or {
                "type": "schema",
                "searchable": True,
                "categories": ["brand", "design", "content"]
            },
            permissions=permissions or {
                "type": PermissionType.SCHEMA,
                "read": True,
                "write": True,
                "roles": ["admin", "designer", "content"]
            },
            status=status or {
                "type": StatusType.SCHEMA,
                "active": True
            }
        )
    
    def validate_colors(self, colors: List[BrandKitColor]) -> bool:
        """Validate brand kit colors."""
        if not colors:
            return False
        
        for color in colors:
            if not isinstance(color, BrandKitColor):
                return False
            if not color.validate():
                return False
        
        return True
    
    def validate_typography(self, typography: Dict[str, BrandKitTypography]) -> bool:
        """Validate brand kit typography."""
        if not typography:
            return False
        
        for name, typo in typography.items():
            if not isinstance(typo, BrandKitTypography):
                return False
            if not typo.validate():
                return False
        
        return True
    
    def validate_voice(self, voice: BrandKitVoice) -> bool:
        """Validate brand kit voice."""
        if not voice:
            return False
        
        if not isinstance(voice, BrandKitVoice):
            return False
        
        return voice.validate()
    
    def get_data(self) -> Dict[str, Any]:
        """Get schema data."""
        return {
            "name": self.name,
            "description": self.description,
            "fields": self.fields,
            "validation": self.validation,
            "cache": self.cache,
            "events": self.events,
            "index": self.index,
            "permissions": self.permissions,
            "status": self.status
        }
    
    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> BrandKitSchema:
        """Create schema from data."""
        return cls(
            name=data["name"],
            description=data.get("description"),
            fields=data.get("fields", {}),
            validation=data.get("validation"),
            cache=data.get("cache"),
            events=data.get("events"),
            index=data.get("index"),
            permissions=data.get("permissions"),
            status=data.get("status")
        ) 