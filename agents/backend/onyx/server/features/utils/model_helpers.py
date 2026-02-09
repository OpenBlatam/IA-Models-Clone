from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from datetime import datetime
import json
import logging
import re
from .model_types import (
from datetime import datetime
from typing import List, Optional
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Helpers - Onyx Integration
Helper functions for model operations.
"""
    JsonDict, JsonList, JsonValue, FieldType, FieldValue,
    ModelId, ModelKey, ModelValue, ModelData, ModelList, ModelDict,
    IndexField, IndexValue, IndexKey, IndexData, IndexList, IndexDict,
    CacheKey, CacheValue, CacheData, CacheList, CacheDict,
    ValidationRule, ValidationRules, ValidationError, ValidationErrors,
    EventName, EventData, EventHandler, EventHandlers,
    ModelStatus, ModelCategory, ModelPermission,
    OnyxBaseModel, ModelField, ModelSchema, ModelRegistry,
    ModelCache, ModelIndex, ModelEvent, ModelValidation, ModelFactory
)

T = TypeVar('T', bound=OnyxBaseModel)

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))

def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
    return bool(re.match(pattern, url))

def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r"^\+?1?\d{9,15}$"
    return bool(re.match(pattern, phone))

def validate_date(date: str) -> bool:
    """Validate date format."""
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    return bool(re.match(pattern, date))

def validate_datetime(dt: str) -> bool:
    """Validate datetime format."""
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z?$"
    return bool(re.match(pattern, dt))

def validate_field_type(value: Any, field_type: str) -> bool:
    """Validate field type."""
    type_map = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "date": str,
        "datetime": str,
        "email": str,
        "url": str,
        "phone": str
    }
    
    if field_type not in type_map:
        return False
    
    try:
        if field_type == "email":
            return validate_email(str(value))
        elif field_type == "url":
            return validate_url(str(value))
        elif field_type == "phone":
            return validate_phone(str(value))
        elif field_type == "date":
            return validate_date(str(value))
        elif field_type == "datetime":
            return validate_datetime(str(value))
        else:
            return isinstance(value, type_map[field_type])
    except Exception:
        return False

def validate_field_value(value: Any, rules: ValidationRule) -> List[ValidationError]:
    """Validate field value against rules."""
    errors = []
    
    # Check required
    if rules.get("required", False) and value is None:
        errors.append("Field is required")
        return errors
    
    # Skip validation if value is None and not required
    if value is None:
        return errors
    
    # Check type
    if "type" in rules and not validate_field_type(value, rules["type"]):
        errors.append(f"Invalid type: expected {rules['type']}")
    
    # Check minimum
    if "minimum" in rules and value < rules["minimum"]:
        errors.append(f"Value must be greater than or equal to {rules['minimum']}")
    
    # Check maximum
    if "maximum" in rules and value > rules["maximum"]:
        errors.append(f"Value must be less than or equal to {rules['maximum']}")
    
    # Check min length
    if "min_length" in rules and len(str(value)) < rules["min_length"]:
        errors.append(f"Length must be greater than or equal to {rules['min_length']}")
    
    # Check max length
    if "max_length" in rules and len(str(value)) > rules["max_length"]:
        errors.append(f"Length must be less than or equal to {rules['max_length']}")
    
    # Check pattern
    if "pattern" in rules and not re.match(rules["pattern"], str(value)):
        errors.append("Value does not match pattern")
    
    # Check enum
    if "enum" in rules and value not in rules["enum"]:
        errors.append(f"Value must be one of {rules['enum']}")
    
    return errors

def validate_model_fields(model: T, schema: ModelSchema) -> ModelValidation:
    """Validate model fields against schema."""
    validation = ModelValidation(rules=schema.validation or {})
    
    for field_name, field in schema.fields.items():
        value = getattr(model, field_name, None)
        field_rules = schema.validation.get(field_name, {}) if schema.validation else {}
        
        # Validate field
        errors = validate_field_value(value, field_rules)
        validation.errors.extend(errors)
    
    validation.is_valid = len(validation.errors) == 0
    return validation

def create_model_index(model: T, field: IndexField, value: IndexValue) -> ModelIndex:
    """Create model index."""
    return ModelIndex(
        field=field,
        value=value,
        model_id=model.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

def create_model_cache(model: T, key: CacheKey, value: CacheValue) -> ModelCache:
    """Create model cache."""
    return ModelCache(
        key=key,
        value=value,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

def create_model_event(model: T, name: EventName, data: EventData) -> ModelEvent:
    """Create model event."""
    return ModelEvent(
        name=name,
        data=data,
        model_id=model.id,
        created_at=datetime.utcnow()
    )

def serialize_model(model: T) -> JsonDict:
    """Serialize model to JSON dictionary."""
    return model.model_dump()

def deserialize_model(model_class: Type[T], data: JsonDict) -> T:
    """Deserialize JSON dictionary to model."""
    return model_class(**data)

def get_model_indexes(model: T, schema: ModelSchema) -> List[ModelIndex]:
    """Get model indexes from schema."""
    indexes = []
    
    if schema.indexes:
        for field in schema.indexes:
            value = getattr(model, field, None)
            if value is not None:
                indexes.append(create_model_index(model, field, value))
    
    return indexes

def get_model_cache(model: T, schema: ModelSchema) -> List[ModelCache]:
    """Get model cache from schema."""
    cache = []
    
    if schema.cache:
        for key in schema.cache:
            value = getattr(model, key, None)
            if value is not None:
                cache.append(create_model_cache(model, str(value), serialize_model(model)))
    
    return cache

def get_model_events(model: T, schema: ModelSchema, event_name: EventName) -> List[ModelEvent]:
    """Get model events from schema."""
    events = []
    
    if schema.events and event_name in schema.events:
        for handler in schema.events[event_name]:
            event_data = {
                "model": serialize_model(model),
                "handler": handler.__name__ if hasattr(handler, "__name__") else str(handler)
            }
            events.append(create_model_event(model, event_name, event_data))
    
    return events

def update_model_timestamps(model: T) -> None:
    """Update model timestamps."""
    now = datetime.utcnow()
    
    if not model.created_at:
        model.created_at = now
    
    model.updated_at = now

def update_model_status(model: T, status: ModelStatus) -> None:
    """Update model status."""
    model.status = status
    update_model_timestamps(model)

def update_model_version(model: T, version: str) -> None:
    """Update model version."""
    model.version = version
    update_model_timestamps(model)

def update_model_metadata(model: T, metadata: JsonDict) -> None:
    """Update model metadata."""
    if model.metadata is None:
        model.metadata = {}
    
    model.metadata.update(metadata)
    update_model_timestamps(model)

# Example usage:
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create model with helpers
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
        return validate_model_fields(self, self.schema)
    
    def get_indexes(self) -> List[ModelIndex]:
        return get_model_indexes(self, self.schema)
    
    def get_cache(self) -> List[ModelCache]:
        return get_model_cache(self, self.schema)
    
    def get_events(self, event_name: EventName) -> List[ModelEvent]:
        return get_model_events(self, self.schema, event_name)

# Create and use model with helpers
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
    
    # Get events
    events = user.get_events("created")
    logger.info(f"Events: {events}")
    
    # Update timestamps
    update_model_timestamps(user)
    logger.info(f"Updated timestamps: {user.created_at}, {user.updated_at}")
    
    # Update status
    update_model_status(user, ModelStatus.ACTIVE)
    logger.info(f"Updated status: {user.status}")
    
    # Update version
    update_model_version(user, "1.1.0")
    logger.info(f"Updated version: {user.version}")
    
    # Update metadata
    update_model_metadata(user, {"source": "api", "tags": ["user", "active"]})
    logger.info(f"Updated metadata: {user.metadata}")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
""" 