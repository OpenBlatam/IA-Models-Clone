from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from typing import Dict, Any, List
from ..base_model import OnyxBaseModel
from ..model_schema import ModelSchema
from ..model_field import ModelField
from ..model_utils import (
from ..model_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Model Utils - Onyx Integration
Tests for model utilities.
"""
    get_model_class,
    create_model_instance,
    validate_model_instance,
    get_model_events_for_instance,
    get_model_cache_for_instance,
    get_model_index_for_instance,
    get_model_permission_for_instance,
    get_model_status_for_instance,
    get_model_version_for_instance,
    get_model_audit_for_instance,
    get_model_validation_for_instance,
    get_model_serialization_for_instance,
    get_model_deserialization_for_instance
)
    ValidationError,
    RegistryError,
    FactoryError,
    CacheError,
    SerializationError,
    DeserializationError
)

# Test data
@pytest.fixture
def test_model_data() -> Dict[str, Any]:
    """Test model data."""
    return {
        "name": "Test Model",
        "email": "test@example.com",
        "age": 30,
        "tags": ["test", "model"]
    }

@pytest.fixture
def test_schema() -> ModelSchema:
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

# Test model class
class TestModel(OnyxBaseModel):
    """Test model class."""
    def __init__(self, **data: Any):
        
    """__init__ function."""
super().__init__(schema=test_schema(), data=data)

# Test model registry functions
def test_get_model_class():
    """Test getting model class by name."""
    model_class = get_model_class("TestModel")
    assert model_class == TestModel
    
    with pytest.raises(RegistryError):
        get_model_class("NonExistentModel")

# Test model instance functions
def test_create_model_instance(test_model_data: Dict[str, Any]):
    """Test creating model instance."""
    model = create_model_instance("TestModel", test_model_data)
    assert isinstance(model, TestModel)
    assert model.name == test_model_data["name"]
    assert model.email == test_model_data["email"]
    assert model.age == test_model_data["age"]
    assert model.tags == test_model_data["tags"]

def test_validate_model_instance(test_model: TestModel):
    """Test validating model instance."""
    validation = validate_model_instance(test_model)
    assert validation.is_valid is True
    
    # Test invalid model
    test_model.email = "invalid-email"
    validation = validate_model_instance(test_model)
    assert validation.is_valid is False
    assert len(validation.errors) > 0

# Test model events
def test_get_model_events_for_instance(test_model: TestModel):
    """Test getting model events for instance."""
    events = get_model_events_for_instance(test_model, "created")
    assert isinstance(events, list)
    assert len(events) > 0
    assert events[0]["action"] == "update"

# Test model cache
def test_get_model_cache_for_instance(test_model: TestModel):
    """Test getting model cache for instance."""
    cache = get_model_cache_for_instance(test_model)
    assert isinstance(cache, dict)
    assert "id" in cache
    assert "version" in cache
    assert "created_at" in cache
    assert "updated_at" in cache

# Test model index
def test_get_model_index_for_instance(test_model: TestModel):
    """Test getting model index for instance."""
    index = get_model_index_for_instance(test_model)
    assert isinstance(index, dict)
    assert "id" in index
    assert "name" in index
    assert "email" in index

# Test model permission
def test_get_model_permission_for_instance(test_model: TestModel):
    """Test getting model permission for instance."""
    permission = get_model_permission_for_instance(test_model)
    assert isinstance(permission, str)
    assert permission in ["read", "write", "delete", "admin", "owner", "viewer", "editor", "manager"]

# Test model status
def test_get_model_status_for_instance(test_model: TestModel):
    """Test getting model status for instance."""
    status = get_model_status_for_instance(test_model)
    assert isinstance(status, str)
    assert status in ["active", "inactive", "deleted", "archived", "draft", "published", "pending", "rejected", "approved"]

# Test model version
def test_get_model_version_for_instance(test_model: TestModel):
    """Test getting model version for instance."""
    version = get_model_version_for_instance(test_model)
    assert isinstance(version, str)
    assert version == "1.0.0"

# Test model audit
def test_get_model_audit_for_instance(test_model: TestModel):
    """Test getting model audit for instance."""
    audit = get_model_audit_for_instance(test_model)
    assert isinstance(audit, list)
    assert len(audit) > 0
    assert "timestamp" in audit[0]
    assert "action" in audit[0]
    assert "data" in audit[0]

# Test model validation
def test_get_model_validation_for_instance(test_model: TestModel):
    """Test getting model validation for instance."""
    validation = get_model_validation_for_instance(test_model)
    assert isinstance(validation, dict)
    assert "is_valid" in validation
    assert "errors" in validation
    assert validation["is_valid"] is True
    assert len(validation["errors"]) == 0

# Test model serialization
def test_get_model_serialization_for_instance(test_model: TestModel):
    """Test getting model serialization for instance."""
    serialization = get_model_serialization_for_instance(test_model)
    assert isinstance(serialization, dict)
    assert "id" in serialization
    assert "version" in serialization
    assert "name" in serialization
    assert "email" in serialization
    assert "age" in serialization
    assert "tags" in serialization
    assert "created_at" in serialization
    assert "updated_at" in serialization
    assert "is_deleted" in serialization
    assert "deleted_at" in serialization
    assert "audit_log" in serialization

# Test model deserialization
def test_get_model_deserialization_for_instance(test_model: TestModel):
    """Test getting model deserialization for instance."""
    serialization = get_model_serialization_for_instance(test_model)
    deserialization = get_model_deserialization_for_instance(TestModel, serialization)
    assert isinstance(deserialization, TestModel)
    assert deserialization.id == test_model.id
    assert deserialization.version == test_model.version
    assert deserialization.name == test_model.name
    assert deserialization.email == test_model.email
    assert deserialization.age == test_model.age
    assert deserialization.tags == test_model.tags
    assert deserialization.created_at == test_model.created_at
    assert deserialization.updated_at == test_model.updated_at
    assert deserialization.is_deleted == test_model.is_deleted
    assert deserialization.deleted_at == test_model.deleted_at
    assert len(deserialization.audit_log) == len(test_model.audit_log)

# Test error handling
def test_validation_error(test_model: TestModel):
    """Test validation error."""
    test_model.email = "invalid-email"
    with pytest.raises(ValidationError):
        validate_model_instance(test_model)

def test_cache_error(test_model: TestModel):
    """Test cache error."""
    test_model._created_at = "invalid-date"
    with pytest.raises(CacheError):
        get_model_cache_for_instance(test_model)

def test_serialization_error(test_model: TestModel):
    """Test serialization error."""
    test_model._created_at = "invalid-date"
    with pytest.raises(SerializationError):
        get_model_serialization_for_instance(test_model)

def test_deserialization_error():
    """Test deserialization error."""
    with pytest.raises(DeserializationError):
        get_model_deserialization_for_instance(TestModel, {"created_at": "invalid-date"}) 