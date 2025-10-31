from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union, ClassVar
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import uuid
import re
import time
from functools import lru_cache
from datetime import datetime
from typing import List, Optional
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Types - Onyx Integration
Type definitions for model operations.
"""

# Type variables
T = TypeVar('T', bound='OnyxBaseModel')
M = TypeVar('M', bound='OnyxBaseModel')

# Cache configuration
CACHE_TTL = 2  # seconds
CACHE_SIZE = 1000

# Basic types
JsonDict = Dict[str, Any]
JsonList = List[Any]
JsonValue = Union[str, int, float, bool, None, JsonDict, JsonList]

# Field types
FieldType = Union[str, int, float, bool, datetime, None]
FieldValue = Union[FieldType, List[FieldType], Dict[str, FieldType]]

# Model types
ModelId = Union[str, int]
ModelKey = str
ModelValue = Any
ModelData = Dict[str, Any]
ModelList = List["T"]
ModelDict = Dict[ModelKey, "T"]

# Index types
IndexField = str
IndexValue = Any
IndexKey = str
IndexData = Dict[IndexField, IndexValue]
IndexList = List[IndexData]
IndexDict = Dict[IndexKey, IndexData]

# Cache types
CacheKey = str
CacheValue = Any
CacheData = Dict[CacheKey, CacheValue]
CacheList = List[CacheData]
CacheDict = Dict[CacheKey, CacheData]

# Validation types
ValidationRule = Dict[str, Any]
ValidationRules = Dict[str, ValidationRule]
ValidationError = str
ValidationErrors = List[ValidationError]

# Event types
EventName = str
EventData = Dict[str, Any]
EventHandler = Any
EventHandlers = Dict[EventName, List[EventHandler]]

# Status types
class ModelStatus(str, Enum):
    """Model status types."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"
    ARCHIVED = "archived"
    DRAFT = "draft"
    PUBLISHED = "published"
    PENDING = "pending"
    REJECTED = "rejected"
    APPROVED = "approved"

# Category types
class ModelCategory(str, Enum):
    """Model category types."""
    USER = "user"
    PRODUCT = "product"
    ORDER = "order"
    CUSTOMER = "customer"
    INVENTORY = "inventory"
    PAYMENT = "payment"
    SHIPPING = "shipping"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SYSTEM = "system"

# Permission types
class ModelPermission(str, Enum):
    """Model permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"
    VIEWER = "viewer"
    EDITOR = "editor"
    MANAGER = "manager"

# Model registry types
class ModelRegistry:
    """Registry for model classes."""
    _models: Dict[str, Type['OnyxBaseModel']] = {}
    
    @classmethod
    def register(cls, model_class: Type['OnyxBaseModel']) -> Type['OnyxBaseModel']:
        """Register a model class."""
        cls._models[model_class.__name__.lower()] = model_class
        return model_class
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Type['OnyxBaseModel']]:
        """Get a model class by name."""
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())

# Base model types
class OnyxBaseModel(BaseModel):
    """Base model for Onyx with performance optimizations."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    # Core fields with optimized defaults
    id: Optional[ModelId] = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    status: ModelStatus = Field(default=ModelStatus.ACTIVE)
    category: ModelCategory = Field(default=ModelCategory.SYSTEM)
    permission: ModelPermission = Field(default=ModelPermission.VIEWER)
    version: str = Field(default="1.0.0")
    metadata: Optional[JsonDict] = Field(default_factory=dict)
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    previous_version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    validation_rules: Optional[Dict[str, Any]] = None
    cache_keys: List[str] = Field(default_factory=list)
    index_fields: List[str] = Field(default_factory=list)
    
    # Performance optimized fields
    _validation_cache: ClassVar[Dict[str, Any]] = {}
    _cache_timestamp: ClassVar[Dict[str, float]] = {}
    _lazy_validation: bool = True
    _validation_timeout: float = 2.0  # seconds

    @property
    def is_active(self) -> bool:
        """Check if model is active (cached)."""
        cache_key = f"{self.id}_is_active"
        if cache_key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < CACHE_TTL:
                return self._validation_cache[cache_key]
        
        result = self.status == ModelStatus.ACTIVE and not self.is_deleted
        self._validation_cache[cache_key] = result
        self._cache_timestamp[cache_key] = time.time()
        return result

    def validate(self) -> List[str]:
        """Validate model data with timeout and caching."""
        start_time = time.time()
        cache_key = f"{self.id}_validation"
        
        # Check cache first
        if cache_key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < CACHE_TTL:
                return self._validation_cache[cache_key]
        
        errors = []
        try:
            # Basic model validation with timeout
            if self._lazy_validation:
                # Only validate required fields first
                for field_name, field in self.model_fields.items():
                    if field.is_required():
                        value = getattr(self, field_name)
                        if value is None:
                            errors.append(f"{field_name} is required")
                            if time.time() - start_time > self._validation_timeout:
                                return errors
            else:
                # Full validation
                self.model_validate(self.model_dump())
            
            # Custom validation rules with timeout
            if self.validation_rules:
                for field, rules in self.validation_rules.items():
                    if time.time() - start_time > self._validation_timeout:
                        return errors
                    
                    value = getattr(self, field, None)
                    if rules.get('required') and value is None:
                        errors.append(f"{field} is required")
                    if value is not None:
                        if rules.get('min_length') and len(str(value)) < rules['min_length']:
                            errors.append(f"{field} must be at least {rules['min_length']} characters")
                        if rules.get('max_length') and len(str(value)) > rules['max_length']:
                            errors.append(f"{field} must be at most {rules['max_length']} characters")
                        if rules.get('pattern') and not re.match(rules['pattern'], str(value)):
                            errors.append(f"{field} format is invalid")
                        if rules.get('min') and value < rules['min']:
                            errors.append(f"{field} must be greater than {rules['min']}")
                        if rules.get('max') and value > rules['max']:
                            errors.append(f"{field} must be less than {rules['max']}")
            
            # Cache results
            self._validation_cache[cache_key] = errors
            self._cache_timestamp[cache_key] = time.time()
            
        except Exception as e:
            errors.append(str(e))
        
        return errors

    def is_valid(self) -> bool:
        """Check if model is valid (cached)."""
        cache_key = f"{self.id}_is_valid"
        if cache_key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < CACHE_TTL:
                return self._validation_cache[cache_key]
        
        result = len(self.validate()) == 0
        self._validation_cache[cache_key] = result
        self._cache_timestamp[cache_key] = time.time()
        return result

    @lru_cache(maxsize=CACHE_SIZE)
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary with caching."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def to_json(self) -> str:
        """Convert model to JSON string with caching."""
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

    def update(self, data: Dict[str, Any]) -> None:
        """Update model fields with new data and clear cache."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Clear model cache."""
        self._validation_cache.clear()
        self._cache_timestamp.clear()
        self.to_dict.cache_clear()
        self.to_json.cache_clear()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnyxBaseModel':
        """Create model instance from dictionary with validation timeout."""
        start_time = time.time()
        try:
            instance = cls.model_validate(data)
            if time.time() - start_time > cls._validation_timeout:
                # If validation takes too long, return a basic instance
                return cls(**{k: v for k, v in data.items() if k in cls.model_fields})
            return instance
        except Exception as e:
            # If validation fails, return a basic instance
            return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    @classmethod
    def from_json(cls, json_str: str) -> 'OnyxBaseModel':
        """Create model instance from JSON string with validation timeout."""
        start_time = time.time()
        try:
            instance = cls.model_validate_json(json_str)
            if time.time() - start_time > cls._validation_timeout:
                # If validation takes too long, return a basic instance
                data = cls.model_validate_json(json_str)
                return cls(**{k: v for k, v in data.items() if k in cls.model_fields})
            return instance
        except Exception as e:
            # If validation fails, return a basic instance
            data = cls.model_validate_json(json_str)
            return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    def soft_delete(self) -> None:
        """Soft delete the model."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.status = ModelStatus.DELETED

    def restore(self) -> None:
        """Restore a soft-deleted model."""
        self.is_deleted = False
        self.deleted_at = None
        self.status = ModelStatus.ACTIVE

    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_relationship(self, rel_name: str, rel_id: str) -> None:
        """Add a relationship to the model."""
        if rel_name not in self.relationships:
            self.relationships[rel_name] = []
        if rel_id not in self.relationships[rel_name]:
            self.relationships[rel_name].append(rel_id)

    def remove_relationship(self, rel_name: str, rel_id: str) -> None:
        """Remove a relationship from the model."""
        if rel_name in self.relationships and rel_id in self.relationships[rel_name]:
            self.relationships[rel_name].remove(rel_id)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute on the model."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get a custom attribute from the model."""
        return self.attributes.get(key, default)

    def remove_attribute(self, key: str) -> None:
        """Remove a custom attribute from the model."""
        self.attributes.pop(key, None)

    def add_cache_key(self, key: str) -> None:
        """Add a cache key to the model."""
        if key not in self.cache_keys:
            self.cache_keys.append(key)

    def add_index_field(self, field: str) -> None:
        """Add an index field to the model."""
        if field not in self.index_fields:
            self.index_fields.append(field)

# Model field types
class ModelField(BaseModel):
    """Model field definition with performance optimizations."""
    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (string, integer, float, boolean, etc.)")
    required: bool = Field(default=False, description="Whether the field is required")
    unique: bool = Field(default=False, description="Whether the field value must be unique")
    description: Optional[str] = Field(default=None, description="Field description")
    default: Any = Field(default=None, description="Default value for the field")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules for the field")
    choices: Optional[List[Any]] = Field(default=None, description="List of allowed values")
    min_length: Optional[int] = Field(default=None, description="Minimum length for string fields")
    max_length: Optional[int] = Field(default=None, description="Maximum length for string fields")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Minimum value for numeric fields")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Maximum value for numeric fields")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for string validation")
    format: Optional[str] = Field(default=None, description="Format specification (email, url, date, etc.)")
    index: bool = Field(default=False, description="Whether to create an index for this field")
    cache: bool = Field(default=False, description="Whether to cache this field")
    readonly: bool = Field(default=False, description="Whether the field is read-only")
    hidden: bool = Field(default=False, description="Whether to hide the field in responses")
    computed: bool = Field(default=False, description="Whether the field is computed")
    computed_expression: Optional[str] = Field(default=None, description="Expression to compute the field value")
    depends_on: Optional[List[str]] = Field(default=None, description="Fields this field depends on")
    transform: Optional[str] = Field(default=None, description="Transformation to apply to the field value")
    encrypt: bool = Field(default=False, description="Whether to encrypt the field value")
    sensitive: bool = Field(default=False, description="Whether the field contains sensitive data")
    deprecated: bool = Field(default=False, description="Whether the field is deprecated")
    deprecated_message: Optional[str] = Field(default=None, description="Message explaining why the field is deprecated")
    version_added: Optional[str] = Field(default=None, description="Version when the field was added")
    version_removed: Optional[str] = Field(default=None, description="Version when the field was removed")
    examples: Optional[List[Any]] = Field(default=None, description="Example values for the field")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the field")

    # Performance optimized fields
    _validation_cache: ClassVar[Dict[str, Any]] = {}
    _cache_timestamp: ClassVar[Dict[str, float]] = {}
    _validation_timeout: float = 2.0  # seconds
    _lazy_validation: bool = True

    @lru_cache(maxsize=CACHE_SIZE)
    def validate_field(self, value: Any) -> List[str]:
        """Validate a field value against its rules with timeout and caching."""
        start_time = time.time()
        cache_key = f"{self.name}_{str(value)}"
        
        # Check cache first
        if cache_key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < CACHE_TTL:
                return self._validation_cache[cache_key]
        
        errors = []
        
        # Type validation with timeout
        if value is not None:
            try:
                if self.type == "string" and not isinstance(value, str):
                    errors.append(f"{self.name} must be a string")
                elif self.type == "integer" and not isinstance(value, int):
                    errors.append(f"{self.name} must be an integer")
                elif self.type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"{self.name} must be a number")
                elif self.type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{self.name} must be a boolean")
                elif self.type == "array" and not isinstance(value, list):
                    errors.append(f"{self.name} must be an array")
                elif self.type == "object" and not isinstance(value, dict):
                    errors.append(f"{self.name} must be an object")
                
                if time.time() - start_time > self._validation_timeout:
                    return errors
            except Exception as e:
                errors.append(f"Type validation error for {self.name}: {str(e)}")
                return errors

        # Required validation
        if self.required and value is None:
            errors.append(f"{self.name} is required")
            return errors

        # Lazy validation for non-required fields
        if self._lazy_validation and not self.required and value is None:
            return errors

        # Length validation with timeout
        if value is not None and isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                errors.append(f"{self.name} must be at least {self.min_length} characters")
            if self.max_length is not None and len(value) > self.max_length:
                errors.append(f"{self.name} must be at most {self.max_length} characters")
            
            if time.time() - start_time > self._validation_timeout:
                return errors

        # Value range validation with timeout
        if value is not None and isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"{self.name} must be greater than or equal to {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"{self.name} must be less than or equal to {self.max_value}")
            
            if time.time() - start_time > self._validation_timeout:
                return errors

        # Pattern validation with timeout
        if value is not None and isinstance(value, str) and self.pattern:
            if not re.match(self.pattern, value):
                errors.append(f"{self.name} format is invalid")
            
            if time.time() - start_time > self._validation_timeout:
                return errors

        # Choices validation with timeout
        if value is not None and self.choices and value not in self.choices:
            errors.append(f"{self.name} must be one of {self.choices}")
            
            if time.time() - start_time > self._validation_timeout:
                return errors

        # Format validation with timeout
        if value is not None and isinstance(value, str) and self.format:
            if self.format == "email" and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", value):
                errors.append(f"{self.name} must be a valid email address")
            elif self.format == "url" and not re.match(r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$", value):
                errors.append(f"{self.name} must be a valid URL")
            elif self.format == "date" and not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                errors.append(f"{self.name} must be a valid date (YYYY-MM-DD)")
            elif self.format == "datetime" and not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z?$", value):
                errors.append(f"{self.name} must be a valid datetime (ISO 8601)")
            
            if time.time() - start_time > self._validation_timeout:
                return errors

        # Cache results
        self._validation_cache[cache_key] = errors
        self._cache_timestamp[cache_key] = time.time()
        
        return errors

    @lru_cache(maxsize=CACHE_SIZE)
    def get_default_value(self) -> Optional[Dict[str, Any]]:
        """Get the default value for the field with caching."""
        if self.default is not None:
            return self.default
        if self.type == "string":
            return ""
        if self.type == "integer":
            return 0
        if self.type == "float":
            return 0.0
        if self.type == "boolean":
            return False
        if self.type == "array":
            return []
        if self.type == "object":
            return {}
        return None

    def _clear_cache(self) -> None:
        """Clear field cache."""
        self._validation_cache.clear()
        self._cache_timestamp.clear()
        self.validate_field.cache_clear()
        self.get_default_value.cache_clear()

    @lru_cache(maxsize=CACHE_SIZE)
    def to_dict(self) -> Dict[str, Any]:
        """Convert field to dictionary with caching."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def to_json(self) -> str:
        """Convert field to JSON string with caching."""
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

# Model schema types
class ModelSchema(BaseModel):
    """Model schema definition with performance optimizations."""
    name: str = Field(..., description="Schema name")
    fields: Dict[str, ModelField] = Field(..., description="Field definitions")
    indexes: Optional[List[str]] = Field(default=None, description="List of fields to index")
    cache: Optional[List[str]] = Field(default=None, description="List of fields to cache")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Global validation rules")
    version: str = Field(default="1.0.0", description="Schema version")
    description: Optional[str] = Field(default=None, description="Schema description")
    category: Optional[str] = Field(default=None, description="Schema category")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    permissions: Optional[Dict[str, List[str]]] = Field(default=None, description="Permission definitions")
    events: Optional[Dict[str, List[str]]] = Field(default=None, description="Event definitions")
    hooks: Optional[Dict[str, List[str]]] = Field(default=None, description="Hook definitions")
    computed_fields: Optional[Dict[str, str]] = Field(default=None, description="Computed field definitions")
    relationships: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Relationship definitions")
    constraints: Optional[List[Dict[str, Any]]] = Field(default=None, description="Schema constraints")
    migrations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Schema migrations")
    deprecated: bool = Field(default=False, description="Whether the schema is deprecated")
    deprecated_message: Optional[str] = Field(default=None, description="Message explaining why the schema is deprecated")
    version_added: Optional[str] = Field(default=None, description="Version when the schema was added")
    version_removed: Optional[str] = Field(default=None, description="Version when the schema was removed")
    examples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Example data")

    # Performance optimized fields
    _validation_cache: ClassVar[Dict[str, Any]] = {}
    _cache_timestamp: ClassVar[Dict[str, float]] = {}
    _validation_timeout: float = 2.0  # seconds
    _lazy_validation: bool = True

    @lru_cache(maxsize=CACHE_SIZE)
    def validate_schema(self) -> List[str]:
        """Validate the schema definition with timeout and caching."""
        start_time = time.time()
        cache_key = f"{self.name}_validation"
        
        # Check cache first
        if cache_key in self._validation_cache:
            if time.time() - self._cache_timestamp.get(cache_key, 0) < CACHE_TTL:
                return self._validation_cache[cache_key]
        
        errors = []

        # Validate field names with timeout
        for field_name, field in self.fields.items():
            if time.time() - start_time > self._validation_timeout:
                return errors
                
            if not isinstance(field_name, str):
                errors.append(f"Field name must be a string: {field_name}")
            if not isinstance(field, ModelField):
                errors.append(f"Field definition must be a ModelField: {field_name}")

        # Validate indexes with timeout
        if self.indexes:
            for index in self.indexes:
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if index not in self.fields:
                    errors.append(f"Index field not found: {index}")

        # Validate cache fields with timeout
        if self.cache:
            for cache_field in self.cache:
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if cache_field not in self.fields:
                    errors.append(f"Cache field not found: {cache_field}")

        # Validate computed fields with timeout
        if self.computed_fields:
            for field_name, expression in self.computed_fields.items():
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if field_name not in self.fields:
                    errors.append(f"Computed field not found: {field_name}")
                if not isinstance(expression, str):
                    errors.append(f"Computed field expression must be a string: {field_name}")

        # Validate relationships with timeout
        if self.relationships:
            for rel_name, rel_def in self.relationships.items():
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if not isinstance(rel_def, dict):
                    errors.append(f"Relationship definition must be a dictionary: {rel_name}")
                if "type" not in rel_def:
                    errors.append(f"Relationship type not specified: {rel_name}")
                if "model" not in rel_def:
                    errors.append(f"Relationship model not specified: {rel_name}")

        # Validate constraints with timeout
        if self.constraints:
            for constraint in self.constraints:
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if not isinstance(constraint, dict):
                    errors.append("Constraint must be a dictionary")
                if "type" not in constraint:
                    errors.append("Constraint type not specified")
                if "fields" not in constraint:
                    errors.append("Constraint fields not specified")

        # Validate migrations with timeout
        if self.migrations:
            for migration in self.migrations:
                if time.time() - start_time > self._validation_timeout:
                    return errors
                    
                if not isinstance(migration, dict):
                    errors.append("Migration must be a dictionary")
                if "version" not in migration:
                    errors.append("Migration version not specified")
                if "changes" not in migration:
                    errors.append("Migration changes not specified")

        # Cache results
        self._validation_cache[cache_key] = errors
        self._cache_timestamp[cache_key] = time.time()
        
        return errors

    @lru_cache(maxsize=CACHE_SIZE)
    def get_field(self, field_name: str) -> Optional[ModelField]:
        """Get a field definition by name with caching."""
        return self.fields.get(field_name)

    def add_field(self, field: ModelField) -> None:
        """Add a field to the schema and clear cache."""
        self.fields[field.name] = field
        self._clear_cache()

    def remove_field(self, field_name: str) -> None:
        """Remove a field from the schema and clear cache."""
        self.fields.pop(field_name, None)
        self._clear_cache()

    def add_index(self, field_name: str) -> None:
        """Add an index field and clear cache."""
        if field_name in self.fields and field_name not in (self.indexes or []):
            if self.indexes is None:
                self.indexes = []
            self.indexes.append(field_name)
            self._clear_cache()

    def remove_index(self, field_name: str) -> None:
        """Remove an index field and clear cache."""
        if self.indexes and field_name in self.indexes:
            self.indexes.remove(field_name)
            self._clear_cache()

    def add_cache_field(self, field_name: str) -> None:
        """Add a cache field and clear cache."""
        if field_name in self.fields and field_name not in (self.cache or []):
            if self.cache is None:
                self.cache = []
            self.cache.append(field_name)
            self._clear_cache()

    def remove_cache_field(self, field_name: str) -> None:
        """Remove a cache field and clear cache."""
        if self.cache and field_name in self.cache:
            self.cache.remove(field_name)
            self._clear_cache()

    def add_computed_field(self, field_name: str, expression: str) -> None:
        """Add a computed field and clear cache."""
        if field_name in self.fields:
            if self.computed_fields is None:
                self.computed_fields = {}
            self.computed_fields[field_name] = expression
            self._clear_cache()

    def remove_computed_field(self, field_name: str) -> None:
        """Remove a computed field and clear cache."""
        if self.computed_fields and field_name in self.computed_fields:
            self.computed_fields.pop(field_name)
            self._clear_cache()

    def add_relationship(self, name: str, rel_type: str, model: str, **kwargs) -> None:
        """Add a relationship definition and clear cache."""
        if self.relationships is None:
            self.relationships = {}
        self.relationships[name] = {
            "type": rel_type,
            "model": model,
            **kwargs
        }
        self._clear_cache()

    def remove_relationship(self, name: str) -> None:
        """Remove a relationship definition and clear cache."""
        if self.relationships and name in self.relationships:
            self.relationships.pop(name)
            self._clear_cache()

    def add_constraint(self, constraint_type: str, fields: List[str], **kwargs) -> None:
        """Add a schema constraint and clear cache."""
        if self.constraints is None:
            self.constraints = []
        self.constraints.append({
            "type": constraint_type,
            "fields": fields,
            **kwargs
        })
        self._clear_cache()

    def remove_constraint(self, constraint_type: str, fields: List[str]) -> None:
        """Remove a schema constraint and clear cache."""
        if self.constraints:
            self.constraints = [
                c for c in self.constraints
                if not (c["type"] == constraint_type and c["fields"] == fields)
            ]
            self._clear_cache()

    def add_migration(self, version: str, changes: List[Dict[str, Any]]) -> None:
        """Add a schema migration and clear cache."""
        if self.migrations is None:
            self.migrations = []
        self.migrations.append({
            "version": version,
            "changes": changes
        })
        self._clear_cache()

    def remove_migration(self, version: str) -> None:
        """Remove a schema migration and clear cache."""
        if self.migrations:
            self.migrations = [m for m in self.migrations if m["version"] != version]
            self._clear_cache()

    def _clear_cache(self) -> None:
        """Clear schema cache."""
        self._validation_cache.clear()
        self._cache_timestamp.clear()
        self.validate_schema.cache_clear()
        self.get_field.cache_clear()
        self.to_dict.cache_clear()
        self.to_json.cache_clear()

    @lru_cache(maxsize=CACHE_SIZE)
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary with caching."""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def to_json(self) -> str:
        """Convert schema to JSON string with caching."""
        return self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSchema':
        """Create schema from dictionary with validation timeout."""
        start_time = time.time()
        try:
            instance = cls.model_validate(data)
            if time.time() - start_time > cls._validation_timeout:
                # If validation takes too long, return a basic instance
                return cls(**{k: v for k, v in data.items() if k in cls.model_fields})
            return instance
        except Exception as e:
            # If validation fails, return a basic instance
            return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelSchema':
        """Create schema from JSON string with validation timeout."""
        start_time = time.time()
        try:
            instance = cls.model_validate_json(json_str)
            if time.time() - start_time > cls._validation_timeout:
                # If validation takes too long, return a basic instance
                data = cls.model_validate_json(json_str)
                return cls(**{k: v for k, v in data.items() if k in cls.model_fields})
            return instance
        except Exception as e:
            # If validation fails, return a basic instance
            data = cls.model_validate_json(json_str)
            return cls(**{k: v for k, v in data.items() if k in cls.model_fields})

# Model cache types
class ModelCache(BaseModel):
    """Model cache entry."""
    key: str
    value: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# Model index types
class ModelIndex(BaseModel):
    """Model index entry."""
    field: str
    value: Any
    model_id: str
    created_at: datetime
    updated_at: datetime

# Model event types
class ModelEvent(BaseModel):
    """Model event."""
    name: str
    data: Dict[str, Any]
    model_id: str
    created_at: datetime = datetime.utcnow()

# Model validation types
class ModelValidation(BaseModel):
    """Model validation result."""
    is_valid: bool
    errors: List[str]

# Model factory types
class ModelFactory(BaseModel):
    """Model factory."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    registry: ClassVar[ModelRegistry] = ModelRegistry()
    
    @classmethod
    def create(cls, model_name: str, **data: Any) -> 'OnyxBaseModel':
        """Create a model instance."""
        model_class = cls.registry.get_model(model_name)
        if model_class is None:
            raise ValueError(f"Model {model_name} not found")
        return model_class(**data)
    
    @classmethod
    def validate(cls, model_name: str, **data: Any) -> ModelValidation:
        """Validate model data."""
        model_class = cls.registry.get_model(model_name)
        if model_class is None:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            model_class(**data)
            return ModelValidation(is_valid=True, errors=[])
        except Exception as e:
            return ModelValidation(is_valid=False, errors=[str(e)])

# Example usage:
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create model with types
class UserModel(OnyxBaseModel):
    name: str
    email: str
    age: Optional[int] = None
    
    # Define schema
    schema = ModelSchema(
        name="user",
        fields={
            "name": ModelField(
                name="name",
                type="string",
                required=True,
                description="User's full name"
            ),
            "email": ModelField(
                name="email",
                type="string",
                required=True,
                unique=True,
                description="User's email address"
            ),
            "age": ModelField(
                name="age",
                type="integer",
                required=False,
                description="User's age"
            )
        },
        indexes=["email"],
        cache=["id", "email"],
        validation={
            "email": {
                "type": "string",
                "format": "email",
                "required": True
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            }
        }
    )
    
    def validate(self) -> ModelValidation:
        validation = ModelValidation(rules=self.schema.validation)
        
        # Validate email
        if not self.email or "@" not in self.email:
            validation.errors.append("Invalid email format")
            validation.is_valid = False
        
        # Validate age
        if self.age is not None and (self.age < 0 or self.age > 150):
            validation.errors.append("Age must be between 0 and 150")
            validation.is_valid = False
        
        return validation
    
    def get_indexes(self) -> List[ModelIndex]:
        indexes = []
        for field in self.schema.indexes:
            value = getattr(self, field)
            if value is not None:
                indexes.append(
                    ModelIndex(
                        field=field,
                        value=value,
                        model_id=self.id
                    )
                )
        return indexes
    
    def get_cache(self) -> List[ModelCache]:
        cache = []
        for key in self.schema.cache:
            value = getattr(self, key)
            if value is not None:
                cache.append(
                    ModelCache(
                        key=str(value),
                        value=self.model_dump()
                    )
                )
        return cache

# Create and use model with types
try:
    user = UserModel(
        name="John",
        email="john@example.com",
        age=30
    )
    
    # Validate
    validation = user.validate()
    if not validation.is_valid:
        logger.error(f"Validation errors: {validation.errors}")
    
    # Get indexes
    indexes = user.get_indexes()
    logger.info(f"Indexes: {indexes}")
    
    # Get cache
    cache = user.get_cache()
    logger.info(f"Cache: {cache}")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
""" 