from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from ..base_model import (
from ..redis_indexer import RedisIndexer
from typing import Any, List, Dict, Optional
import logging
import asyncio
    OnyxBaseModel,
    OnyxGenericModel,
    TimestampMixin,
    IdentifierMixin,
    StatusMixin,
    IndexingMixin,
    ValidationMixin,
    SerializationMixin
)

# Test model classes
class TestModel(OnyxBaseModel):
    """Test model for base model functionality."""
    name: str
    value: int
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    
    index_fields = ["id", "name"]
    search_fields = ["name", "description", "tags"]

class TestGenericModel(OnyxGenericModel[str]):
    """Test generic model."""
    pass

@pytest.fixture
def test_model_data():
    """Create test model data."""
    return {
        "name": "Test Model",
        "value": 42,
        "description": "A test model",
        "tags": ["test", "example"]
    }

@pytest.fixture
def test_model(test_model_data) -> Any:
    """Create a test model instance."""
    return TestModel(**test_model_data)

@pytest.fixture
def redis_indexer():
    """Create a Redis indexer instance."""
    return RedisIndexer(
        host="localhost",
        port=6379,
        db=15  # Use a different DB for testing
    )

# Test TimestampMixin
def test_timestamp_mixin(test_model) -> Any:
    """Test timestamp mixin functionality."""
    assert isinstance(test_model.created_at, datetime)
    assert isinstance(test_model.updated_at, datetime)
    
    # Test timestamp update
    old_updated_at = test_model.updated_at
    test_model.name = "Updated Name"
    assert test_model.updated_at > old_updated_at

# Test IdentifierMixin
def test_identifier_mixin(test_model) -> Any:
    """Test identifier mixin functionality."""
    assert isinstance(test_model.id, str)
    assert test_model.version == 1
    
    # Test version increment
    test_model.increment_version()
    assert test_model.version == 2

# Test StatusMixin
def test_status_mixin(test_model) -> Any:
    """Test status mixin functionality."""
    assert test_model.is_active is True
    assert test_model.status == "active"
    
    # Test deactivation
    test_model.deactivate()
    assert test_model.is_active is False
    assert test_model.status == "inactive"
    
    # Test activation
    test_model.activate()
    assert test_model.is_active is True
    assert test_model.status == "active"

# Test IndexingMixin
def test_indexing_mixin(test_model, redis_indexer) -> Any:
    """Test indexing mixin functionality."""
    # Test index data
    index_data = test_model.get_index_data()
    assert "id" in index_data
    assert "name" in index_data
    assert index_data["name"] == test_model.name
    
    # Test search data
    search_data = test_model.get_search_data()
    assert "name" in search_data
    assert "description" in search_data
    assert "tags" in search_data
    assert search_data["name"] == test_model.name
    assert search_data["description"] == test_model.description
    assert search_data["tags"] == test_model.tags
    
    # Test indexing operations
    test_model.index(redis_indexer)
    test_model.update_index(redis_indexer)
    test_model.remove_index(redis_indexer)

# Test ValidationMixin
def test_validation_mixin(test_model) -> Any:
    """Test validation mixin functionality."""
    # Test valid model
    assert test_model.is_valid() is True
    assert len(test_model.validate_fields()) == 0
    
    # Test invalid model
    test_model.name = None
    assert test_model.is_valid() is False
    assert len(test_model.validate_fields()) > 0
    assert any("name" in error for error in test_model.validate_fields())

# Test SerializationMixin
def test_serialization_mixin(test_model) -> Any:
    """Test serialization mixin functionality."""
    # Test to_dict
    model_dict = test_model.to_dict()
    assert "id" in model_dict
    assert "data" in model_dict
    assert "metadata" in model_dict
    assert model_dict["data"]["name"] == test_model.name
    assert model_dict["data"]["value"] == test_model.value
    
    # Test from_dict
    new_model = TestModel.from_dict(model_dict)
    assert new_model.name == test_model.name
    assert new_model.value == test_model.value
    assert new_model.id == test_model.id
    assert new_model.version == test_model.version
    
    # Test to_json and from_json
    json_str = test_model.to_json()
    new_model = TestModel.from_json(json_str)
    assert new_model.name == test_model.name
    assert new_model.value == test_model.value
    
    # Test hash generation
    hash_value = test_model.generate_hash()
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA-256 hash length

# Test OnyxBaseModel
def test_onyx_base_model(test_model_data) -> Any:
    """Test OnyxBaseModel functionality."""
    # Test model creation
    model = TestModel(**test_model_data)
    assert model.name == test_model_data["name"]
    assert model.value == test_model_data["value"]
    assert model.description == test_model_data["description"]
    assert model.tags == test_model_data["tags"]
    
    # Test model update
    new_name = "Updated Name"
    model.name = new_name
    assert model.name == new_name
    assert model.updated_at > model.created_at
    
    # Test model validation
    assert model.is_valid() is True
    
    # Test model serialization
    model_dict = model.to_dict()
    assert model_dict["data"]["name"] == new_name

# Test OnyxGenericModel
def test_onyx_generic_model():
    """Test OnyxGenericModel functionality."""
    # Test model creation
    data = "Test Data"
    metadata = {"type": "test"}
    model = TestGenericModel.create(data, **metadata)
    assert model.data == data
    assert model.metadata == metadata
    
    # Test model update
    new_data = "Updated Data"
    model.update(new_data)
    assert model.data == new_data 