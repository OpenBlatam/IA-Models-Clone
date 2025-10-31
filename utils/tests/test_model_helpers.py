from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..base_model import OnyxBaseModel
from ..model_helpers import (
from ..model_types import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
    validate_email,
    validate_url,
    validate_phone,
    validate_date,
    validate_datetime,
    validate_field_type,
    validate_field_value,
    validate_model_fields,
    create_model_index,
    create_model_cache,
    create_model_event,
    serialize_model,
    deserialize_model,
    get_model_indexes,
    get_model_cache,
    get_model_events,
    update_model_timestamps,
    update_model_status,
    update_model_version,
    update_model_metadata
)
    ModelSchema,
    ModelValidation,
    ModelIndex,
    ModelCache,
    ModelEvent,
    ModelStatus
)

# Test model classes
class TestModel(OnyxBaseModel):
    """Test model for helper functionality."""
    name: str
    email: str
    age: Optional[int] = None
    url: Optional[str] = None
    phone: Optional[str] = None
    date: Optional[str] = None
    datetime: Optional[str] = None
    tags: List[str] = []
    
    index_fields = ["id", "name", "email"]
    search_fields = ["name", "email", "tags"]

@pytest.fixture
def test_model_data():
    """Create test model data."""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "age": 30,
        "url": "https://example.com",
        "phone": "+1234567890",
        "date": "2024-01-01",
        "datetime": "2024-01-01T12:00:00Z",
        "tags": ["test", "example"]
    }

@pytest.fixture
def test_model(test_model_data) -> Any:
    """Create a test model instance."""
    return TestModel(**test_model_data)

@pytest.fixture
def test_schema():
    """Create test schema."""
    return ModelSchema(
        fields={
            "name": {"type": "string", "required": True},
            "email": {"type": "email", "required": True},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "url": {"type": "url"},
            "phone": {"type": "phone"},
            "date": {"type": "date"},
            "datetime": {"type": "datetime"},
            "tags": {"type": "array"}
        },
        validation={
            "name": {"type": "string", "required": True, "min_length": 2},
            "email": {"type": "email", "required": True},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "url": {"type": "url"},
            "phone": {"type": "phone"},
            "date": {"type": "date"},
            "datetime": {"type": "datetime"},
            "tags": {"type": "array"}
        },
        indexes=["id", "name", "email"],
        cache=["id", "email"]
    )

# Test validation functions
def test_validate_email():
    """Test email validation."""
    assert validate_email("test@example.com") is True
    assert validate_email("invalid-email") is False
    assert validate_email("test@.com") is False
    assert validate_email("@example.com") is False

def test_validate_url():
    """Test URL validation."""
    assert validate_url("https://example.com") is True
    assert validate_url("http://example.com") is True
    assert validate_url("invalid-url") is False
    assert validate_url("ftp://example.com") is False

def test_validate_phone():
    """Test phone validation."""
    assert validate_phone("+1234567890") is True
    assert validate_phone("1234567890") is True
    assert validate_phone("invalid-phone") is False
    assert validate_phone("123") is False

def test_validate_date():
    """Test date validation."""
    assert validate_date("2024-01-01") is True
    assert validate_date("2024-13-01") is False
    assert validate_date("invalid-date") is False
    assert validate_date("01/01/2024") is False

def test_validate_datetime():
    """Test datetime validation."""
    assert validate_datetime("2024-01-01T12:00:00Z") is True
    assert validate_datetime("2024-01-01T12:00:00.123Z") is True
    assert validate_datetime("invalid-datetime") is False
    assert validate_datetime("2024-01-01 12:00:00") is False

def test_validate_field_type():
    """Test field type validation."""
    assert validate_field_type("test", "string") is True
    assert validate_field_type(42, "integer") is True
    assert validate_field_type(3.14, "float") is True
    assert validate_field_type(True, "boolean") is True
    assert validate_field_type([], "array") is True
    assert validate_field_type({}, "object") is True
    assert validate_field_type("test@example.com", "email") is True
    assert validate_field_type("https://example.com", "url") is True
    assert validate_field_type("+1234567890", "phone") is True
    assert validate_field_type("2024-01-01", "date") is True
    assert validate_field_type("2024-01-01T12:00:00Z", "datetime") is True

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

def test_validate_model_fields(test_model, test_schema) -> bool:
    """Test model fields validation."""
    validation = validate_model_fields(test_model, test_schema)
    assert validation.is_valid is True
    
    # Test invalid model
    test_model.email = "invalid-email"
    validation = validate_model_fields(test_model, test_schema)
    assert validation.is_valid is False
    assert any("email" in error for error in validation.errors)

# Test model operations
def test_create_model_index(test_model) -> Any:
    """Test model index creation."""
    index = create_model_index(test_model, "name", "Test User")
    assert isinstance(index, ModelIndex)
    assert index.field == "name"
    assert index.value == "Test User"
    assert index.model_id == test_model.id

def test_create_model_cache(test_model) -> Any:
    """Test model cache creation."""
    cache = create_model_cache(test_model, "id", test_model.id)
    assert isinstance(cache, ModelCache)
    assert cache.key == "id"
    assert cache.value == test_model.id

def test_create_model_event(test_model) -> Any:
    """Test model event creation."""
    event = create_model_event(test_model, "created", {"action": "create"})
    assert isinstance(event, ModelEvent)
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

def test_get_model_indexes(test_model, test_schema) -> Optional[Dict[str, Any]]:
    """Test getting model indexes."""
    indexes = get_model_indexes(test_model, test_schema)
    assert len(indexes) > 0
    assert any(index.field == "id" for index in indexes)
    assert any(index.field == "name" for index in indexes)
    assert any(index.field == "email" for index in indexes)

def test_get_model_cache(test_model, test_schema) -> Optional[Dict[str, Any]]:
    """Test getting model cache."""
    cache = get_model_cache(test_model, test_schema)
    assert len(cache) > 0
    assert any(c.key == "id" for c in cache)
    assert any(c.key == "email" for c in cache)

def test_get_model_events(test_model, test_schema) -> Optional[Dict[str, Any]]:
    """Test getting model events."""
    events = get_model_events(test_model, test_schema, "created")
    assert isinstance(events, list)

# Test model updates
def test_update_model_timestamps(test_model) -> Any:
    """Test model timestamp updates."""
    old_updated_at = test_model.updated_at
    update_model_timestamps(test_model)
    assert test_model.updated_at > old_updated_at

def test_update_model_status(test_model) -> Any:
    """Test model status update."""
    update_model_status(test_model, ModelStatus.INACTIVE)
    assert test_model.status == ModelStatus.INACTIVE

def test_update_model_version(test_model) -> Any:
    """Test model version update."""
    update_model_version(test_model, "2.0")
    assert test_model.version == "2.0"

def test_update_model_metadata(test_model) -> Any:
    """Test model metadata update."""
    metadata = {"key": "value"}
    update_model_metadata(test_model, metadata)
    assert test_model.metadata == metadata 