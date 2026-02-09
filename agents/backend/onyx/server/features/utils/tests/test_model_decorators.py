from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import logging
import re
from datetime import datetime
from typing import Optional

from ..base_model import OnyxBaseModel
from ..model_decorators import (
    from ..model_utils import ModelRegistry
    from ..model_utils import ModelCache
from typing import Any, List, Dict, Optional
import asyncio
    register_model,
    cache_model,
    validate_model,
    track_changes,
    require_active,
    log_operations,
    enforce_version,
    validate_schema
)

# Test model classes
@register_model
class TestModel(OnyxBaseModel):
    """Test model for decorator functionality."""
    name: str
    email: str
    age: Optional[int] = None
    version: str = "1.0"

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test schema
test_schema = {
    "name": {"type": str},
    "email": {"type": str, "pattern": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")},
    "age": {"type": int, "min": 0, "max": 150}
}

@pytest.fixture
def test_model_data():
    """Create test model data."""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "age": 30
    }

@pytest.fixture
def test_model(test_model_data) -> Any:
    """Create a test model instance."""
    return TestModel(**test_model_data)

# Test register_model decorator
def test_register_model():
    """Test model registration."""
    assert TestModel in ModelRegistry.models.values()

# Test cache_model decorator
def test_cache_model(test_model) -> Any:
    """Test model caching."""
    @cache_model("id")
    def get_model():
        
    """get_model function."""
return test_model
    
    # Get and cache model
    cached_model = get_model()
    assert cached_model == test_model
    
    # Verify model is cached
    assert ModelCache.get(str(test_model.id)) == test_model

# Test validate_model decorator
def test_validate_model(test_model) -> bool:
    """Test model validation."""
    @validate_model(validate_types=True, validate_custom=True)
    def validate_test_model():
        
    """validate_test_model function."""
return test_model
    
    # Test valid model
    validated_model = validate_test_model()
    assert validated_model == test_model
    
    # Test invalid model
    test_model.email = "invalid-email"
    with pytest.raises(ValueError):
        validate_test_model()

# Test track_changes decorator
def test_track_changes(test_model) -> Any:
    """Test change tracking."""
    @track_changes
    def update_model(model) -> Any:
        model.name = "Updated Name"
        return model
    
    # Update model and track changes
    updated_model = update_model(test_model)
    assert updated_model.name == "Updated Name"
    assert updated_model.updated_at > test_model.created_at

# Test require_active decorator
def test_require_active(test_model) -> Any:
    """Test active requirement."""
    @require_active
    def process_model(model) -> Any:
        return model
    
    # Test active model
    result = process_model(test_model)
    assert result == test_model
    
    # Test inactive model
    test_model.deactivate()
    with pytest.raises(ValueError):
        process_model(test_model)

# Test log_operations decorator
def test_log_operations(test_model) -> Any:
    """Test operation logging."""
    @log_operations(logger)
    def log_test_operation(model) -> Any:
        return model
    
    # Test operation logging
    result = log_test_operation(test_model)
    assert result == test_model

# Test enforce_version decorator
def test_enforce_version(test_model) -> Any:
    """Test version enforcement."""
    @enforce_version("1.0")
    def version_check(model) -> Any:
        return model
    
    # Test correct version
    result = version_check(test_model)
    assert result == test_model
    
    # Test incorrect version
    test_model.version = "2.0"
    with pytest.raises(ValueError):
        version_check(test_model)

# Test validate_schema decorator
def test_validate_schema(test_model) -> bool:
    """Test schema validation."""
    @validate_schema(test_schema)
    def schema_check(model) -> Any:
        return model
    
    # Test valid schema
    result = schema_check(test_model)
    assert result == test_model
    
    # Test invalid email
    test_model.email = "invalid-email"
    with pytest.raises(ValueError):
        schema_check(test_model)
    
    # Test invalid age
    test_model.email = "test@example.com"  # Reset to valid email
    test_model.age = 200
    with pytest.raises(ValueError):
        schema_check(test_model)

# Test multiple decorators
def test_multiple_decorators(test_model) -> Any:
    """Test multiple decorators together."""
    @validate_model()
    @track_changes
    @require_active
    @log_operations(logger)
    def process_with_decorators(model) -> Any:
        model.name = "Processed Name"
        return model
    
    # Test combined decorators
    result = process_with_decorators(test_model)
    assert result.name == "Processed Name"
    assert result.updated_at > test_model.created_at
    
    # Test with inactive model
    test_model.deactivate()
    with pytest.raises(ValueError):
        process_with_decorators(test_model) 