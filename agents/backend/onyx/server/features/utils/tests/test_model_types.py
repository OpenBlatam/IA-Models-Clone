from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from typing import Dict, Any, List
from ..base_model import OnyxBaseModel
from ..model_schema import ModelSchema
from ..model_field import ModelField
from ..model_types import (
from ..model_exceptions import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Model Types - Onyx Integration
Tests for model types and registry.
"""
    ModelRegistry,
    ModelFactory,
    ModelValidation,
    ModelStatus,
    ModelPermission
)
    ValidationError,
    RegistryError,
    FactoryError
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

# Test model status
def test_model_status():
    """Test model status enum."""
    assert ModelStatus.ACTIVE == "active"
    assert ModelStatus.INACTIVE == "inactive"
    assert ModelStatus.DELETED == "deleted"
    assert ModelStatus.ARCHIVED == "archived"
    assert ModelStatus.DRAFT == "draft"
    assert ModelStatus.PUBLISHED == "published"
    assert ModelStatus.PENDING == "pending"
    assert ModelStatus.REJECTED == "rejected"
    assert ModelStatus.APPROVED == "approved"

# Test model permission
def test_model_permission():
    """Test model permission enum."""
    assert ModelPermission.READ == "read"
    assert ModelPermission.WRITE == "write"
    assert ModelPermission.DELETE == "delete"
    assert ModelPermission.ADMIN == "admin"
    assert ModelPermission.OWNER == "owner"
    assert ModelPermission.VIEWER == "viewer"
    assert ModelPermission.EDITOR == "editor"
    assert ModelPermission.MANAGER == "manager"

# Test model registry
def test_model_registry():
    """Test model registry."""
    registry = ModelRegistry()
    
    # Register model
    registry.register_model("testmodel", TestModel)
    
    # Get model
    model_class = registry.get_model_class("testmodel")
    assert model_class == TestModel
    
    # List models
    models = registry.list_models()
    assert "testmodel" in models
    
    # Test duplicate registration
    with pytest.raises(RegistryError):
        registry.register_model("testmodel", TestModel)
    
    # Test non-existent model
    with pytest.raises(RegistryError):
        registry.get_model_class("nonexistent")

def test_model_registry_operations(test_schema: ModelSchema, test_model_data: Dict[str, Any]):
    """Test model registry operations."""
    registry = ModelRegistry()
    registry.register_model("testmodel", TestModel)
    
    # Create model
    model = registry.create_model("testmodel", test_model_data)
    assert isinstance(model, TestModel)
    assert model.name == test_model_data["name"]
    assert model.email == test_model_data["email"]
    assert model.age == test_model_data["age"]
    assert model.tags == test_model_data["tags"]
    
    # Get model
    retrieved_model = registry.get_model("testmodel", model.id)
    assert retrieved_model == model
    
    # Update model
    updated_data = {"name": "Updated Model"}
    updated_model = registry.update_model("testmodel", model.id, updated_data)
    assert updated_model.name == "Updated Model"
    
    # Delete model
    registry.delete_model("testmodel", model.id)
    assert registry.get_model("testmodel", model.id) is None
    
    # Clear models
    registry.clear_models("testmodel")
    assert len(registry.get_models("testmodel")) == 0

# Test model factory
def test_model_factory():
    """Test model factory."""
    factory = ModelFactory()
    
    # Register model
    factory.register_model("testmodel", TestModel)
    
    # Create model
    model = factory.create_model("testmodel", test_model_data())
    assert isinstance(model, TestModel)
    assert model.name == test_model_data()["name"]
    assert model.email == test_model_data()["email"]
    assert model.age == test_model_data()["age"]
    assert model.tags == test_model_data()["tags"]
    
    # Validate model
    validation = factory.validate("testmodel", test_model_data())
    assert validation.is_valid is True
    assert len(validation.errors) == 0
    
    # Test invalid data
    invalid_data = {"email": "invalid-email"}
    validation = factory.validate("testmodel", invalid_data)
    assert validation.is_valid is False
    assert len(validation.errors) > 0
    
    # Test non-existent model
    with pytest.raises(FactoryError):
        factory.create_model("nonexistent", test_model_data())
    
    with pytest.raises(FactoryError):
        factory.validate("nonexistent", test_model_data())

def test_model_factory_with_schema(test_schema: ModelSchema):
    """Test model factory with schema."""
    factory = ModelFactory()
    
    # Register schema
    factory.register_schema("testschema", test_schema)
    
    # Register model with schema
    factory.register_model("testmodel", TestModel)
    
    # Create model with schema
    model = factory.create_model("testmodel", test_model_data())
    assert model._schema == test_schema
    
    # Validate with schema
    validation = factory.validate("testmodel", test_model_data())
    assert validation.is_valid is True
    assert len(validation.errors) == 0

# Test model validation
def test_model_validation():
    """Test model validation."""
    validation = ModelValidation(is_valid=True, errors=[])
    assert validation.is_valid is True
    assert len(validation.errors) == 0
    
    validation = ModelValidation(
        is_valid=False,
        errors=["Invalid email", "Invalid age"]
    )
    assert validation.is_valid is False
    assert len(validation.errors) == 2
    assert "Invalid email" in validation.errors
    assert "Invalid age" in validation.errors

def test_model_validation_with_schema(test_schema: ModelSchema):
    """Test model validation with schema."""
    factory = ModelFactory()
    factory.register_schema("testschema", test_schema)
    factory.register_model("testmodel", TestModel)
    
    # Test valid data
    validation = factory.validate("testmodel", test_model_data())
    assert validation.is_valid is True
    assert len(validation.errors) == 0
    
    # Test invalid email
    invalid_email_data = test_model_data().copy()
    invalid_email_data["email"] = "invalid-email"
    validation = factory.validate("testmodel", invalid_email_data)
    assert validation.is_valid is False
    assert len(validation.errors) > 0
    assert any("email" in error.lower() for error in validation.errors)
    
    # Test invalid age
    invalid_age_data = test_model_data().copy()
    invalid_age_data["age"] = 200
    validation = factory.validate("testmodel", invalid_age_data)
    assert validation.is_valid is False
    assert len(validation.errors) > 0
    assert any("age" in error.lower() for error in validation.errors)
    
    # Test missing required field
    missing_field_data = test_model_data().copy()
    del missing_field_data["name"]
    validation = factory.validate("testmodel", missing_field_data)
    assert validation.is_valid is False
    assert len(validation.errors) > 0
    assert any("name" in error.lower() for error in validation.errors) 