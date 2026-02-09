from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union, ClassVar
from datetime import datetime
import logging
import os
import pytest
from pydantic import BaseModel, ValidationError
from ..model_types import (
from ..model_config import ModelConfig
from ..model_helpers import (
from ..model_mixins import (
from ..model_decorators import (
from ..model_exceptions import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Tests - Onyx Integration
Test suite for model operations and validations.
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
    validate_email, validate_url, validate_phone, validate_date, validate_datetime,
    validate_field_type, validate_field_value, validate_model_fields,
    create_model_index, create_model_cache, create_model_event,
    serialize_model, deserialize_model,
    get_model_indexes, get_model_cache, get_model_events,
    update_model_timestamps, update_model_status, update_model_version, update_model_metadata
)
    TimestampMixin, SoftDeleteMixin, VersionMixin, AuditMixin,
    ValidationMixin, CacheMixin, SerializationMixin, IndexingMixin, LoggingMixin
)
    register_model, cache_model, validate_model, track_changes,
    require_active, log_operations, enforce_version, validate_schema
)
    OnyxModelError, ValidationError, IndexingError, CacheError,
    SerializationError, VersionError, AuditError, SoftDeleteError,
    TimestampError, RegistryError, FactoryError, DeserializationError,
    PermissionError, StatusError
)

T = TypeVar('T', bound=OnyxBaseModel)

# Test fixtures
@pytest.fixture
def test_model_data():
    """Test model data fixture."""
    return {
        "name": "Test Model",
        "email": "test@example.com",
        "age": 30,
        "tags": ["test", "model"]
    }

@pytest.fixture
def test_schema():
    """Test schema."""
    return ModelSchema(
        fields={
            "name": ModelField(
                name="name",
                type="string",
                required=True,
                description="Model name"
            ),
            "email": ModelField(
                name="email",
                type="string",
                required=True,
                description="Email address",
                validation={"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            ),
            "age": ModelField(
                name="age",
                type="integer",
                required=True,
                description="Age",
                validation={"min": 0, "max": 150}
            ),
            "tags": ModelField(
                name="tags",
                type="array",
                required=False,
                description="Tags",
                validation={"min_items": 0}
            )
        }
    )

@pytest.fixture
def test_model(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model instance."""
    return OnyxBaseModel(schema=test_schema, data=test_model_data)

# Test model creation
def test_create_model(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model creation."""
    model = OnyxBaseModel(schema=test_schema, data=test_model_data)
    assert model.name == test_model_data["name"]
    assert model.email == test_model_data["email"]
    assert model.age == test_model_data["age"]
    assert model.tags == test_model_data["tags"]
    assert model.created_at is not None
    assert model.updated_at is not None
    assert model.version == "1.0.0"
    assert model.is_deleted is False
    assert model.deleted_at is None
    assert len(model.audit_log) == 1  # Initial creation

def test_create_model_with_id(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model creation with ID."""
    model_id = "test-123"
    model = OnyxBaseModel(schema=test_schema, data=test_model_data, id=model_id)
    assert model.id == model_id

def test_create_model_with_version(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model creation with version."""
    version = "2.0.0"
    model = OnyxBaseModel(schema=test_schema, data=test_model_data, version=version)
    assert model.version == version

# Test model validation
def test_validate_model(test_model: OnyxBaseModel):
    """Test model validation."""
    assert test_model.is_valid() is True
    assert len(test_model.validate()) == 0

def test_validate_invalid_email(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test invalid email validation."""
    test_model_data["email"] = "invalid-email"
    model = OnyxBaseModel(schema=test_schema, data=test_model_data)
    assert model.is_valid() is False
    assert len(model.validate()) > 0

def test_validate_invalid_age(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test invalid age validation."""
    test_model_data["age"] = 200
    model = OnyxBaseModel(schema=test_schema, data=test_model_data)
    assert model.is_valid() is False
    assert len(model.validate()) > 0

# Test model data operations
def test_set_data(test_model: OnyxBaseModel):
    """Test setting model data."""
    new_data = {
        "name": "Updated Model",
        "email": "updated@example.com",
        "age": 35,
        "tags": ["updated", "model"]
    }
    test_model.set_data(new_data)
    assert test_model.name == new_data["name"]
    assert test_model.email == new_data["email"]
    assert test_model.age == new_data["age"]
    assert test_model.tags == new_data["tags"]
    assert len(test_model.audit_log) == 2  # Creation + update

def test_get_data(test_model: OnyxBaseModel, test_model_data: Dict[str, Any]):
    """Test getting model data."""
    data = test_model.get_data()
    assert data["name"] == test_model_data["name"]
    assert data["email"] == test_model_data["email"]
    assert data["age"] == test_model_data["age"]
    assert data["tags"] == test_model_data["tags"]

# Test model serialization
def test_to_dict(test_model: OnyxBaseModel, test_model_data: Dict[str, Any]):
    """Test model serialization."""
    data = test_model.to_dict()
    assert data["id"] == test_model.id
    assert data["version"] == test_model.version
    assert data["name"] == test_model_data["name"]
    assert data["email"] == test_model_data["email"]
    assert data["age"] == test_model_data["age"]
    assert data["tags"] == test_model_data["tags"]
    assert "created_at" in data
    assert "updated_at" in data
    assert "is_deleted" in data
    assert "deleted_at" in data
    assert "audit_log" in data

def test_from_dict(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model deserialization."""
    model = OnyxBaseModel(schema=test_schema, data=test_model_data)
    data = model.to_dict()
    new_model = OnyxBaseModel.from_dict(data)
    assert new_model.id == model.id
    assert new_model.version == model.version
    assert new_model.name == model.name
    assert new_model.email == model.email
    assert new_model.age == model.age
    assert new_model.tags == model.tags
    assert new_model.created_at == model.created_at
    assert new_model.updated_at == model.updated_at
    assert new_model.is_deleted == model.is_deleted
    assert new_model.deleted_at == model.deleted_at
    assert len(new_model.audit_log) == len(model.audit_log)

# Test model deletion
def test_delete_model(test_model: OnyxBaseModel):
    """Test model deletion."""
    assert test_model.is_deleted is False
    test_model.delete()
    assert test_model.is_deleted is True
    assert test_model.deleted_at is not None
    assert len(test_model.audit_log) == 2  # Creation + delete

def test_restore_model(test_model: OnyxBaseModel):
    """Test model restoration."""
    test_model.delete()
    assert test_model.is_deleted is True
    test_model.restore()
    assert test_model.is_deleted is False
    assert test_model.deleted_at is None
    assert len(test_model.audit_log) == 3  # Creation + delete + restore

# Test model versioning
def test_update_version(test_model: OnyxBaseModel):
    """Test version update."""
    assert test_model.version == "1.0.0"
    test_model.update_version("2.0.0")
    assert test_model.version == "2.0.0"
    assert len(test_model.audit_log) == 2  # Creation + version update

def test_update_same_version(test_model: OnyxBaseModel):
    """Test updating to same version."""
    original_version = test_model.version
    test_model.update_version(original_version)
    assert test_model.version == original_version
    assert len(test_model.audit_log) == 1  # Only creation

# Test audit logging
def test_audit_log(test_model: OnyxBaseModel):
    """Test audit logging."""
    assert len(test_model.audit_log) == 1  # Initial creation
    assert test_model.audit_log[0]["action"] == "update"
    assert "data" in test_model.audit_log[0]

def test_audit_log_after_operations(test_model: OnyxBaseModel):
    """Test audit log after multiple operations."""
    test_model.set_data({"name": "Updated"})
    test_model.delete()
    test_model.restore()
    test_model.update_version("2.0.0")
    
    assert len(test_model.audit_log) == 5  # Creation + update + delete + restore + version update
    assert test_model.audit_log[1]["action"] == "update"
    assert test_model.audit_log[2]["action"] == "delete"
    assert test_model.audit_log[3]["action"] == "restore"
    assert test_model.audit_log[4]["action"] == "version_update"

# Test error handling
def test_validation_error(test_schema: ModelSchema):
    """Test validation error."""
    with pytest.raises(ValidationError):
        OnyxBaseModel(schema=test_schema, data={"email": "invalid-email"})

def test_version_error(test_model: OnyxBaseModel):
    """Test version error."""
    with pytest.raises(VersionError):
        test_model.update_version("invalid-version")

def test_serialization_error(test_model: OnyxBaseModel):
    """Test serialization error."""
    test_model._created_at = "invalid-date"  # Make created_at invalid
    with pytest.raises(SerializationError):
        test_model.to_dict()

def test_deserialization_error():
    """Test deserialization error."""
    with pytest.raises(DeserializationError):
        OnyxBaseModel.from_dict({"created_at": "invalid-date"})

# Test model operations
def test_model_operations(test_model) -> Any:
    """Test model operations."""
    # Test update
    new_name = "Updated Name"
    test_model.name = new_name
    assert test_model.name == new_name
    assert test_model.updated_at is not None
    
    # Test soft delete
    test_model.soft_delete()
    assert test_model.is_deleted is True
    assert test_model.deleted_at is not None
    
    # Test restore
    test_model.restore()
    assert test_model.is_deleted is False
    assert test_model.deleted_at is None

# Test model serialization
def test_serialize_model(test_model) -> Any:
    """Test model serialization."""
    data = test_model.to_dict()
    assert data["name"] == test_model.name
    assert data["email"] == test_model.email
    assert data["age"] == test_model.age
    assert data["created_at"] is not None
    assert data["updated_at"] is not None
    assert data["version"] == test_model.version
    assert data["is_deleted"] == test_model.is_deleted
    assert data["deleted_at"] == test_model.deleted_at

# Test model deserialization
def test_deserialize_model(test_model_data) -> Any:
    """Test model deserialization."""
    model = TestModel.from_dict(test_model_data)
    assert model.name == test_model_data["name"]
    assert model.email == test_model_data["email"]
    assert model.age == test_model_data["age"]

# Test model caching
def test_model_cache(test_model) -> Any:
    """Test model caching."""
    # Test cache set
    test_model.cache("email")
    # Note: Actual cache implementation would be tested here
    # This is just a placeholder for the cache functionality

# Test model indexing
def test_model_index(test_model) -> Any:
    """Test model indexing."""
    # Test index set
    test_model.index(None)  # Pass None as indexer since we're not implementing actual indexing
    # Note: Actual index implementation would be tested here
    # This is just a placeholder for the indexing functionality

# Test model mixins
def test_model_mixins(test_model) -> Any:
    """Test model mixins."""
    # Test TimestampMixin
    assert test_model.created_at is not None
    assert test_model.updated_at is not None
    test_model.update_timestamp()
    assert test_model.updated_at is not None
    
    # Test SoftDeleteMixin
    assert test_model.is_deleted is False
    test_model.soft_delete()
    assert test_model.is_deleted is True
    test_model.restore()
    assert test_model.is_deleted is False
    
    # Test VersionMixin
    assert test_model.version is not None
    test_model.update_version("2.0.0")
    assert test_model.version == "2.0.0"
    
    # Test AuditMixin
    test_model.set_audit_fields("test_user")
    assert test_model.created_by == "test_user"
    assert test_model.updated_by == "test_user"
    
    # Test ValidationMixin
    assert test_model.is_valid() is True
    test_model.email = "invalid-email"
    assert test_model.is_valid() is False
    
    # Test CacheMixin
    test_model.cache("email")
    # Note: Actual cache implementation would be tested here
    
    # Test SerializationMixin
    data = test_model.to_dict()
    assert isinstance(data, dict)
    json_str = test_model.to_json()
    assert isinstance(json_str, str)
    
    # Test IndexingMixin
    test_model.index(None)  # Pass None as indexer
    # Note: Actual index implementation would be tested here
    
    # Test LoggingMixin
    test_model.log_info("Test info message")
    test_model.log_error("Test error message")
    test_model.log_warning("Test warning message")
    test_model.log_debug("Test debug message")

# Test validation
def test_validate_email():
    """Test email validation."""
    assert TestModel(email="test@example.com").is_valid() is True
    assert TestModel(email="invalid-email").is_valid() is False

def test_validate_age():
    """Test age validation."""
    assert TestModel(name="Test", email="test@example.com", age=30).is_valid() is True
    assert TestModel(name="Test", email="test@example.com", age=200).is_valid() is False
    assert TestModel(name="Test", email="test@example.com", age=-1).is_valid() is False

def test_validate_url():
    """Test URL validation."""
    assert validate_url("https://example.com") is True
    assert validate_url("invalid-url") is False

def test_validate_phone():
    """Test phone validation."""
    assert validate_phone("+1234567890") is True
    assert validate_phone("invalid-phone") is False

def test_validate_date():
    """Test date validation."""
    assert validate_date("2024-01-01") is True
    assert validate_date("invalid-date") is False

def test_validate_datetime():
    """Test datetime validation."""
    assert validate_datetime("2024-01-01T12:00:00Z") is True
    assert validate_datetime("invalid-datetime") is False

def test_validate_field_type():
    """Test field type validation."""
    assert validate_field_type("test", "string") is True
    assert validate_field_type(123, "integer") is True
    assert validate_field_type(123.45, "float") is True
    assert validate_field_type(True, "boolean") is True
    assert validate_field_type([1, 2, 3], "array") is True
    assert validate_field_type({"key": "value"}, "object") is True
    assert validate_field_type("test@example.com", "email") is True
    assert validate_field_type("https://example.com", "url") is True
    assert validate_field_type("+1234567890", "phone") is True
    assert validate_field_type("2024-01-01", "date") is True
    assert validate_field_type("2024-01-01T12:00:00Z", "datetime") is True
    assert validate_field_type("test", "invalid") is False

def test_validate_field_value():
    """Test field value validation."""
    # Test required field
    errors = validate_field_value(None, {"required": True})
    assert len(errors) > 0
    
    # Test type validation
    errors = validate_field_value("test", {"type": "integer"})
    assert len(errors) > 0
    
    # Test minimum value
    errors = validate_field_value(5, {"type": "integer", "minimum": 10})
    assert len(errors) > 0
    
    # Test maximum value
    errors = validate_field_value(20, {"type": "integer", "maximum": 10})
    assert len(errors) > 0
    
    # Test min length
    errors = validate_field_value("test", {"type": "string", "min_length": 10})
    assert len(errors) > 0
    
    # Test max length
    errors = validate_field_value("test" * 10, {"type": "string", "max_length": 10})
    assert len(errors) > 0
    
    # Test pattern
    errors = validate_field_value("test", {"type": "string", "pattern": r"^\d+$"})
    assert len(errors) > 0
    
    # Test enum
    errors = validate_field_value("test", {"type": "string", "enum": ["value1", "value2"]})
    assert len(errors) > 0

def test_validate_model_fields(test_model) -> bool:
    """Test model fields validation."""
    validation = validate_model_fields(test_model, test_model.schema)
    assert validation.is_valid is True
    
    # Test invalid email
    test_model.email = "invalid-email"
    validation = validate_model_fields(test_model, test_model.schema)
    assert validation.is_valid is False
    
    # Test invalid age
    test_model.age = 200
    validation = validate_model_fields(test_model, test_model.schema)
    assert validation.is_valid is False

# Test model operations
def test_create_model_index(test_model) -> Any:
    """Test model index creation."""
    index = create_model_index(test_model, "email", test_model.email)
    assert index.field == "email"
    assert index.value == test_model.email
    assert index.model_id == test_model.id

def test_create_model_cache(test_model) -> Any:
    """Test model cache creation."""
    cache = create_model_cache(test_model, "email", test_model.email)
    assert cache.key == "email"
    assert cache.value == test_model.email

def test_create_model_event(test_model) -> Any:
    """Test model event creation."""
    event = create_model_event(test_model, "created", {"action": "create"})
    assert event.name == "created"
    assert event.data["action"] == "create"
    assert event.model_id == test_model.id

def test_serialize_model(test_model) -> Any:
    """Test model serialization."""
    data = serialize_model(test_model)
    assert data["name"] == test_model.name
    assert data["email"] == test_model.email
    assert data["age"] == test_model.age

def test_deserialize_model(test_model_data) -> Any:
    """Test model deserialization."""
    model = deserialize_model(TestModel, test_model_data)
    assert model.name == test_model_data["name"]
    assert model.email == test_model_data["email"]
    assert model.age == test_model_data["age"]

def test_get_model_indexes(test_model) -> Optional[Dict[str, Any]]:
    """Test model indexes retrieval."""
    indexes = get_model_indexes(test_model, test_model.schema)
    assert len(indexes) == 1
    assert indexes[0].field == "email"
    assert indexes[0].value == test_model.email

def test_get_model_cache(test_model) -> Optional[Dict[str, Any]]:
    """Test model cache retrieval."""
    cache = get_model_cache(test_model, test_model.schema)
    assert len(cache) == 2
    assert any(c.key == str(test_model.id) for c in cache)
    assert any(c.key == test_model.email for c in cache)

def test_get_model_events(test_model) -> Optional[Dict[str, Any]]:
    """Test model events retrieval."""
    events = get_model_events(test_model, test_model.schema, "created")
    assert len(events) == 0  # No event handlers defined

# Test model updates
def test_update_model_timestamps(test_model) -> Any:
    """Test model timestamps update."""
    update_model_timestamps(test_model)
    assert test_model.created_at is not None
    assert test_model.updated_at is not None

def test_update_model_status(test_model) -> Any:
    """Test model status update."""
    update_model_status(test_model, ModelStatus.INACTIVE)
    assert test_model.status == ModelStatus.INACTIVE

def test_update_model_version(test_model) -> Any:
    """Test model version update."""
    update_model_version(test_model, "1.1.0")
    assert test_model.version == "1.1.0"

def test_update_model_metadata(test_model) -> Any:
    """Test model metadata update."""
    metadata = {"source": "test", "tags": ["test"]}
    update_model_metadata(test_model, metadata)
    assert test_model.metadata == metadata

# Test model decorators
def test_register_model():
    """Test model registration."""
    assert TestModel in ModelRegistry.models.values()

def test_cache_model(test_model) -> Any:
    """Test model caching."""
    @cache_model("email")
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)

def test_validate_model(test_model) -> bool:
    """Test model validation."""
    @validate_model(validate_types=True, validate_custom=True)
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)
    
    test_model.email = "invalid-email"
    with pytest.raises(ValidationError):
        test_method(test_model)

def test_track_changes(test_model) -> Any:
    """Test change tracking."""
    @track_changes
    def test_method(self) -> Any:
        self.name = "Updated Name"
    
    test_method(test_model)
    assert test_model.name == "Updated Name"

def test_require_active(test_model) -> Any:
    """Test active requirement."""
    @require_active
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)
    
    test_model.status = ModelStatus.INACTIVE
    with pytest.raises(ValueError):
        test_method(test_model)

def test_log_operations(test_model) -> Any:
    """Test operation logging."""
    @log_operations(logging.getLogger(__name__))
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)

def test_enforce_version(test_model) -> Any:
    """Test version enforcement."""
    @enforce_version("1.0.0")
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)
    
    test_model.version = "1.1.0"
    with pytest.raises(ValueError):
        test_method(test_model)

def test_validate_schema(test_model) -> bool:
    """Test schema validation."""
    @validate_schema(test_model.schema.validation)
    def test_method(self) -> Any:
        return self.email
    
    test_method(test_model)
    
    test_model.email = "invalid-email"
    with pytest.raises(ValidationError):
        test_method(test_model)

# Test model exceptions
def test_validation_error():
    """Test validation error."""
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError("Validation failed", ["Invalid email"])
    assert str(exc_info.value) == "Validation failed"
    assert exc_info.value.errors == ["Invalid email"]

def test_indexing_error():
    """Test indexing error."""
    with pytest.raises(IndexingError) as exc_info:
        raise IndexingError("Indexing failed", "email", "test@example.com")
    assert str(exc_info.value) == "Indexing failed"
    assert exc_info.value.field == "email"
    assert exc_info.value.value == "test@example.com"

def test_cache_error():
    """Test cache error."""
    with pytest.raises(CacheError) as exc_info:
        raise CacheError("Caching failed", "email")
    assert str(exc_info.value) == "Caching failed"
    assert exc_info.value.key == "email"

def test_serialization_error():
    """Test serialization error."""
    with pytest.raises(SerializationError) as exc_info:
        raise SerializationError("Serialization failed", {"key": "value"})
    assert str(exc_info.value) == "Serialization failed"
    assert exc_info.value.data == {"key": "value"}

def test_version_error():
    """Test version error."""
    with pytest.raises(VersionError) as exc_info:
        raise VersionError("Version mismatch", "1.0.0", "1.1.0")
    assert str(exc_info.value) == "Version mismatch"
    assert exc_info.value.current_version == "1.0.0"
    assert exc_info.value.required_version == "1.1.0"

def test_audit_error():
    """Test audit error."""
    with pytest.raises(AuditError) as exc_info:
        raise AuditError("Audit failed", "user123")
    assert str(exc_info.value) == "Audit failed"
    assert exc_info.value.user_id == "user123"

def test_soft_delete_error():
    """Test soft delete error."""
    with pytest.raises(SoftDeleteError) as exc_info:
        raise SoftDeleteError("Soft delete failed", True)
    assert str(exc_info.value) == "Soft delete failed"
    assert exc_info.value.is_deleted is True

def test_timestamp_error():
    """Test timestamp error."""
    with pytest.raises(TimestampError) as exc_info:
        raise TimestampError("Timestamp error", "2024-01-01")
    assert str(exc_info.value) == "Timestamp error"
    assert exc_info.value.timestamp == "2024-01-01"

def test_registry_error():
    """Test registry error."""
    with pytest.raises(RegistryError) as exc_info:
        raise RegistryError("Registry error", "TestModel")
    assert str(exc_info.value) == "Registry error"
    assert exc_info.value.model_name == "TestModel"

def test_factory_error():
    """Test factory error."""
    with pytest.raises(FactoryError) as exc_info:
        raise FactoryError("Factory error", TestModel)
    assert str(exc_info.value) == "Factory error"
    assert exc_info.value.model_type == TestModel 