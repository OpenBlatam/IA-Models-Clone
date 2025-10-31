from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime
import json
import logging
from pydantic import BaseModel, ValidationError
from .model_types import OnyxBaseModel, ModelCache, ModelIndex
from datetime import datetime
from typing import List, Optional
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Mixins - Onyx Integration
Mixins for model operations and validations.
"""

T = TypeVar('T', bound=OnyxBaseModel)

class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    
    def soft_delete(self) -> None:
        """Soft delete the model."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore the model."""
        self.is_deleted = False
        self.deleted_at = None

class VersionMixin:
    """Mixin for version control."""
    version: str
    previous_version: Optional[str] = None
    
    def update_version(self, new_version: str) -> None:
        """Update the model version."""
        self.previous_version = self.version
        self.version = new_version

class AuditMixin:
    """Mixin for audit fields."""
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    def set_audit_fields(self, user_id: str) -> None:
        """Set audit fields."""
        if not self.created_by:
            self.created_by = user_id
        self.updated_by = user_id

class ValidationMixin:
    """Mixin for validation methods."""
    def validate(self) -> List[str]:
        """Validate the model."""
        errors = []
        try:
            self.model_validate(self.model_dump())
        except ValidationError as e:
            errors.extend([f"{field}: {error}" for field, error in e.errors()])
        return errors
    
    def is_valid(self) -> bool:
        """Check if the model is valid."""
        return len(self.validate()) == 0

class CacheMixin:
    """Mixin for caching methods."""
    def cache(self, key_field: str) -> None:
        """Cache the model."""
        key = getattr(self, key_field)
        cache = ModelCache(
            key=str(key),
            value=self.model_dump(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        # Store cache in Redis or other storage
        # This is a placeholder - implement actual storage logic
    
    def uncache(self, key_field: str) -> None:
        """Remove the model from cache."""
        key = getattr(self, key_field)
        # Remove from Redis or other storage
        # This is a placeholder - implement actual removal logic

class SerializationMixin:
    """Mixin for serialization methods."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create model from JSON string."""
        return cls.model_validate_json(json_str)

class IndexingMixin:
    """Mixin for indexing methods."""
    def index(self, indexer: Any) -> None:
        """Index the model."""
        if hasattr(self, 'index_fields'):
            for field in self.index_fields:
                value = getattr(self, field)
                if value is not None:
                    index = ModelIndex(
                        field=field,
                        value=value,
                        model_id=self.id,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    # Store index in Redis or other storage
                    # This is a placeholder - implement actual storage logic
    
    def unindex(self, indexer: Any) -> None:
        """Remove model from index."""
        if hasattr(self, 'index_fields'):
            for field in self.index_fields:
                value = getattr(self, field)
                if value is not None:
                    # Remove from Redis or other storage
                    # This is a placeholder - implement actual removal logic
                    pass

class LoggingMixin:
    """Mixin for logging methods."""
    def __init__(self, **data) -> Any:
        super().__init__(**data)
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        return self._logger
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

# Example usage:
"""

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create model with mixins
class UserModel(
    OnyxBaseModel,
    TimestampMixin,
    SoftDeleteMixin,
    VersionMixin,
    AuditMixin,
    ValidationMixin,
    CacheMixin,
    SerializationMixin,
    IndexingMixin,
    LoggingMixin
):
    name: str
    email: str
    age: Optional[int] = None
    index_fields = ["email"]
    
    def __init__(self, **data) -> Any:
        super().__init__(**data)
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.version = "1.0.0"

# Create and use model
user = UserModel(
    name="John",
    email="john@example.com",
    age=30
)

# Use mixin methods
user.set_audit_fields("user123")
user.cache("email")
user.log_info("User created successfully")

# Validate model
if user.is_valid():
    print("User is valid")
else:
    print("Validation errors:", user.validate())

# Serialize model
user_dict = user.to_dict()
user_json = user.to_json()

# Create from serialized data
new_user = UserModel.from_dict(user_dict)
new_user_from_json = UserModel.from_json(user_json)

# Soft delete
user.soft_delete()
print("Is deleted:", user.is_deleted)
print("Deleted at:", user.deleted_at)

# Restore
user.restore()
print("Is deleted:", user.is_deleted)
print("Deleted at:", user.deleted_at)

# Update version
user.update_version("1.1.0")
print("Current version:", user.version)
print("Previous version:", user.previous_version)
""" 