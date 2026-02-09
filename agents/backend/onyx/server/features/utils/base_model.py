from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from datetime import datetime
import uuid
import logging
from .model_field import ModelField, FieldConfig
from .model_schema import ModelSchema, SchemaConfig
from .model_exceptions import OnyxModelError, ValidationError, VersionError
from .model_mixins import (
from .model_decorators import validate_model, cache_model, log_operations
from pydantic import BaseModel, ConfigDict
import orjson
from .batch_utils import BatchMethodsMixin
from .value_objects import Money, Dimensions, SEOData
from .enums import ProductStatus, ProductType, PriceType, InventoryTracking
from .validators import not_empty_string, list_or_empty, dict_or_empty
        from .model_repository import ModelRepository
        from .model_service import ModelService
from datetime import datetime, timedelta
from typing import List, Optional
from typing import Any, List, Dict, Optional
import asyncio
"""
Base Model - Onyx Integration
Production-ready base model class with repository, audit context, hooks, and robust logging.
"""
    TimestampMixin, SoftDeleteMixin, VersionMixin, AuditMixin, ValidationMixin, LoggingMixin
)

T = TypeVar('T', bound='OnyxBaseModel')

class OnyxBaseModel(
    TimestampMixin,
    SoftDeleteMixin,
    VersionMixin,
    AuditMixin,
    ValidationMixin,
    LoggingMixin,
    BatchMethodsMixin
):
    """
    Production-ready base model class with repository, audit context, hooks, robust logging, batch methods, and value object/enums integration.
    Inherits timestamp, soft delete, versioning, audit, validation, logging, and batch behaviors.
    All persistence and business logic is delegated to ModelRepository and ModelService.
    Uses orjson for fastest serialization with Pydantic v2.
    Strict field validation and OpenAPI documentation.
    Value objects and enums centralizados para máxima reutilización.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_loads=orjson.loads,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        strict=True,
        title="OnyxBaseModel",
        description="Onyx base model with strict validation, audit, and repository integration."
    )
    _pre_hooks: Dict[str, List[Callable]] = {"create": [], "update": [], "delete": [], "restore": []}
    _post_hooks: Dict[str, List[Callable]] = {"create": [], "update": [], "delete": [], "restore": []}

    def __init__(
        self,
        schema: Optional[ModelSchema] = None,
        data: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        version: str = "1.0.0",
        user_context: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
super().__init__()
        self._schema = schema
        self._id = id or str(uuid.uuid4())
        self.version = version
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.is_deleted = False
        self.deleted_at: Optional[datetime] = None
        self._audit_log: List[Dict[str, Any]] = []
        self._user_context: Optional[Dict[str, Any]] = user_context
        if data:
            self.set_data(data)
        self.log_info(f"Model {self._id} created.")

    @property
    def id(self) -> str:
        """Get model ID."""
        return self._id

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self._audit_log.copy()

    def set_user_context(self, user: Optional[str] = None, ip: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Set user and context for audit logging."""
        self._user_context = {"user": user, "ip": ip, **(extra or {})}

    @classmethod
    def add_pre_hook(cls, action: str, hook: Callable) -> None:
        if action in cls._pre_hooks:
            cls._pre_hooks[action].append(hook)

    @classmethod
    def add_post_hook(cls, action: str, hook: Callable) -> None:
        if action in cls._post_hooks:
            cls._post_hooks[action].append(hook)

    def _run_hooks(self, action: str, pre: bool = True) -> None:
        hooks = self._pre_hooks if pre else self._post_hooks
        for hook in hooks.get(action, []):
            try:
                hook(self)
            except Exception as e:
                self.log_error(f"Error in {'pre' if pre else 'post'}-{action} hook: {e}")

    def set_data(self, data: Dict[str, Any]) -> None:
        """Set model data with granular validation."""
        errors = []
        if self._schema:
            for field_name, field in self._schema.fields.items():
                value = data.get(field_name)
                errors.extend(self.validate_required(field_name, value) if field.required else [])
                if field.type == "string":
                    errors.extend(self.validate_type(field_name, value, str))
                elif field.type == "integer":
                    errors.extend(self.validate_type(field_name, value, int))
                elif field.type == "array":
                    errors.extend(self.validate_type(field_name, value, list))
                if isinstance(value, str):
                    min_length = field.validation.get("min_length") if field.validation else None
                    max_length = field.validation.get("max_length") if field.validation else None
                    errors.extend(self.validate_length(field_name, value, min_length, max_length))
                if isinstance(value, int):
                    min_value = field.validation.get("min") if field.validation else None
                    max_value = field.validation.get("max") if field.validation else None
                    errors.extend(self.validate_range(field_name, value, min_value, max_value))
                if field.validation and "pattern" in field.validation and isinstance(value, str):
                    errors.extend(self.validate_pattern(field_name, value, field.validation["pattern"]))
                if field.validation and "choices" in field.validation:
                    errors.extend(self.validate_choices(field_name, value, field.validation["choices"]))
                if field.validation and "format" in field.validation and isinstance(value, str):
                    errors.extend(self.validate_format(field_name, value, field.validation["format"]))
                if field.type == "relationship":
                    errors.extend(self.validate_relationship(field_name, value))
        if errors:
            self.log_error(f"Validation errors on set_data: {errors}")
            raise ValidationError(f"Invalid data: {', '.join(errors)}", errors)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        self._log_audit("update", {"data": data})
        self.log_info(f"Model {self._id} updated.")

    def get_data(self) -> Dict[str, Any]:
        """Get model data."""
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                data[key] = value
        return data

    def validate(self) -> List[str]:
        """Validate model data using granular field-level validation."""
        errors = []
        if self._schema:
            for field_name, field in self._schema.fields.items():
                value = getattr(self, field_name, None)
                errors.extend(self.validate_required(field_name, value) if field.required else [])
                if field.type == "string":
                    errors.extend(self.validate_type(field_name, value, str))
                elif field.type == "integer":
                    errors.extend(self.validate_type(field_name, value, int))
                elif field.type == "array":
                    errors.extend(self.validate_type(field_name, value, list))
                if isinstance(value, str):
                    min_length = field.validation.get("min_length") if field.validation else None
                    max_length = field.validation.get("max_length") if field.validation else None
                    errors.extend(self.validate_length(field_name, value, min_length, max_length))
                if isinstance(value, int):
                    min_value = field.validation.get("min") if field.validation else None
                    max_value = field.validation.get("max") if field.validation else None
                    errors.extend(self.validate_range(field_name, value, min_value, max_value))
                if field.validation and "pattern" in field.validation and isinstance(value, str):
                    errors.extend(self.validate_pattern(field_name, value, field.validation["pattern"]))
                if field.validation and "choices" in field.validation:
                    errors.extend(self.validate_choices(field_name, value, field.validation["choices"]))
                if field.validation and "format" in field.validation and isinstance(value, str):
                    errors.extend(self.validate_format(field_name, value, field.validation["format"]))
                if field.type == "relationship":
                    errors.extend(self.validate_relationship(field_name, value))
        return errors

    def is_valid(self) -> bool:
        """Check if model is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        data = self.get_data()
        data.update({
            'id': self._id,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_deleted': self.is_deleted,
            'deleted_at': self.deleted_at.isoformat() if self.deleted_at else None,
            'audit_log': self._audit_log,
            'user_context': self._user_context
        })
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model from dictionary."""
        model = cls()
        model._id = data.pop('id', str(uuid.uuid4()))
        model.version = data.pop('version', "1.0.0")
        model.created_at = datetime.fromisoformat(data.pop('created_at'))
        model.updated_at = datetime.fromisoformat(data.pop('updated_at'))
        model.is_deleted = data.pop('is_deleted', False)
        deleted_at = data.pop('deleted_at', None)
        model.deleted_at = datetime.fromisoformat(deleted_at) if deleted_at else None
        model._audit_log = data.pop('audit_log', [])
        model._user_context = data.pop('user_context', None)
        for key, value in data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        return model

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def save(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        """Save the model using the repository and service, with audit and hooks."""
        self._run_hooks("create", pre=True)
        self._service.create_model(self.__class__.__name__, self.to_dict(), self._id)
        self._repository._service = self._service
        self._repository._service.register_model(self.__class__.__name__, self.__class__)
        self._repository._service.register_schema(self.__class__.__name__, self._schema)
        self._repository._service.create_model(self.__class__.__name__, self.to_dict(), self._id)
        self._log_audit("create", {"data": self.to_dict(), "user_context": user_context or self._user_context})
        self._run_hooks("create", pre=False)
        self.log_info(f"Model {self._id} saved.")

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def delete(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        """Soft delete the model using the repository and service, with audit and hooks."""
        self._run_hooks("delete", pre=True)
        self.soft_delete()
        self._service.update_model(self.__class__.__name__, self._id, {"is_deleted": True, "deleted_at": datetime.utcnow().isoformat()})
        self._log_audit("delete", {"user_context": user_context or self._user_context})
        self._run_hooks("delete", pre=False)
        self.log_info(f"Model {self._id} soft deleted.")

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def restore(self, user_context: Optional[Dict[str, Any]] = None) -> None:
        """Restore a soft-deleted model using the repository and service, with audit and hooks."""
        self._run_hooks("restore", pre=True)
        self.restore()
        self._service.update_model(self.__class__.__name__, self._id, {"is_deleted": False, "deleted_at": None})
        self._log_audit("restore", {"user_context": user_context or self._user_context})
        self._run_hooks("restore", pre=False)
        self.log_info(f"Model {self._id} restored.")

    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="id")
    @log_operations(logging.getLogger(__name__))
    def update_version(self, new_version: str, user_context: Optional[Dict[str, Any]] = None) -> None:
        """Update the model version using the repository and service, with audit."""
        old_version = self.version
        self.update_version(new_version)
        self._service.update_model(self.__class__.__name__, self._id, {"version": new_version})
        self._log_audit("update_version", {"old_version": old_version, "new_version": new_version, "user_context": user_context or self._user_context})
        self.log_info(f"Model {self._id} version updated from {old_version} to {new_version}.")

    @classmethod
    def all(cls: Type[T]) -> List[T]:
        """Return all models from the repository."""
        models = cls._repository._service.get_models(cls.__name__)
        return [cls.from_dict(model.to_dict()) for model in models.values()]

    @classmethod
    def load(cls: Type[T], model_id: str) -> Optional[T]:
        """Load a model by ID from the repository."""
        model = cls._repository._service.get_model(cls.__name__, model_id)
        if model:
            return cls.from_dict(model.to_dict())
        return None

    def _log_audit(self, action: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an audit entry for the model."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "model_id": self._id,
            "user_context": self._user_context,
            "data": data or {}
        }
        self._audit_log.append(entry)
        self.log_info(f"Audit log: {entry}")

    @property
    def _repository(self) -> Any:
        if not hasattr(self, '__repository'):
            self.__repository = ModelRepository()
        return self.__repository

    @property
    def _service(self) -> Any:
        if not hasattr(self, '__service'):
            self.__service = ModelService()
        return self.__service

# Generic model with type parameter
class OnyxGenericModel(GenericModel, Generic[T]):
    """Generic model for Onyx with type parameters."""
    
    data: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def create(cls, data: T, **metadata) -> "OnyxGenericModel[T]":
        """Create a generic model instance."""
        return cls(data=data, metadata=metadata)
    
    def update(self, data: T) -> None:
        """Update the model data."""
        self.data = data
        self.metadata["updated_at"] = datetime.utcnow()

# Example usage:
"""

# Create a model with specific mixins
class UserModel(
    OnyxBaseModel,
    TimestampMixin,
    IdentifierMixin,
    StatusMixin
):
    name: str
    email: str
    age: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    
    # Configure indexing
    index_fields = ["id", "email"]
    search_fields = ["name", "tags"]
    cache_ttl = 3600  # 1 hour
    
    @validator("email")
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator("age")
    def validate_age(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 150):
            raise ValueError("Invalid age")
        return v

# Create a model with all mixins
class ProductModel(OnyxBaseModel):
    name: str
    price: float
    category: str
    in_stock: bool = True
    
    # Configure indexing
    index_fields = ["id", "category"]
    search_fields = ["name"]
    
    @validator("price")
    def validate_price(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

# Create a generic model
class UserData:
    def __init__(self, name: str, email: str):
        
    """__init__ function."""
self.name = name
        self.email = email

user_data = UserData("John", "john@example.com")
generic_model = OnyxGenericModel.create(
    data=user_data,
    created_at=datetime.utcnow()
)

# Create and use models
user = UserModel(
    name="John Doe",
    email="john@example.com",
    age=30,
    tags=["premium", "verified"]
)

product = ProductModel(
    name="Laptop",
    price=999.99,
    category="Electronics"
)

# Use mixin functionality
user.activate()
user.increment_version()
user.index(RedisIndexer())

product.deactivate()
product.update_index(RedisIndexer())

# Serialize models
user_dict = user.to_dict()
user_json = user.to_json()

# Validate models
user_errors = user.validate_fields()
is_valid = user.is_valid()

# Use generic model
generic_model.update(UserData("John Updated", "john.updated@example.com"))
""" 