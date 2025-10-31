from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from functools import wraps
import time
import logging
from datetime import datetime
    from .model_utils import ModelRegistry
                from .model_utils import ModelCache
                from .model_utils import ModelValidator
from datetime import datetime
from typing import List, Optional
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Model Decorators - Onyx Integration
Decorators for model operations and validations.
"""
T = TypeVar('T', bound="OnyxBaseModel")

def register_model(model_class: Type[T]) -> Type[T]:
    """Decorator to register a model class."""
    ModelRegistry.register(model_class)
    return model_class

def cache_model(key_field: str):
    """Decorator to cache model instances."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the model instance
            result = func(*args, **kwargs)
            
            # Cache the model if it's an instance of OnyxBaseModel
            if isinstance(result, OnyxBaseModel):
                key = getattr(result, key_field)
                ModelCache.set(result, str(key))
            
            return result
        return wrapper
    return decorator

def validate_model(validate_types: bool = True, validate_custom: bool = True):
    """Decorator to validate model instances."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the model instance
            result = func(*args, **kwargs)
            
            # Validate the model if it's an instance of OnyxBaseModel
            if isinstance(result, OnyxBaseModel):
                validator = ModelValidator()
                
                # Validate required fields
                errors = validator.validate_required_fields(result)
                if errors:
                    raise ValueError(f"Required field errors: {', '.join(errors)}")
                
                # Validate field types if requested
                if validate_types:
                    type_errors = validator.validate_field_types(result)
                    if type_errors:
                        raise ValueError(f"Type errors: {', '.join(type_errors)}")
                
                # Validate custom rules if requested
                if validate_custom:
                    custom_errors = validator.validate_custom_rules(result)
                    if custom_errors:
                        raise ValueError(f"Custom validation errors: {', '.join(custom_errors)}")
            
            return result
        return wrapper
    return decorator

def track_changes(func: Callable) -> Callable:
    """Decorator to track model changes."""
    @wraps(func)
    def wrapper(self: OnyxBaseModel, *args, **kwargs) -> Any:
        # Store original state
        original_state = self.model_dump()
        
        # Execute the function
        result = func(self, *args, **kwargs)
        
        # Compare states
        new_state = self.model_dump()
        changes = {
            field: (original_state[field], new_state[field])
            for field in new_state
            if field in original_state and original_state[field] != new_state[field]
        }
        
        # Log changes
        if changes:
            logging.info(f"Model {self.__class__.__name__} changes: {changes}")
        
        return result
    return wrapper

def require_active(func: Callable) -> Callable:
    """Decorator to require model to be active."""
    @wraps(func)
    def wrapper(self: OnyxBaseModel, *args, **kwargs) -> Any:
        if not self.is_active:
            raise ValueError(f"Model {self.__class__.__name__} is not active")
        return func(self, *args, **kwargs)
    return wrapper

def log_operations(logger: Optional[logging.Logger] = None):
    """Decorator to log model operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: OnyxBaseModel, *args, **kwargs) -> Any:
            # Get logger
            log = logger or logging.getLogger(self.__class__.__name__)
            
            # Log operation start
            start_time = time.time()
            log.info(f"Starting {func.__name__} on {self.__class__.__name__}")
            
            try:
                # Execute the function
                result = func(self, *args, **kwargs)
                
                # Log operation success
                duration = time.time() - start_time
                log.info(f"Completed {func.__name__} in {duration:.2f}s")
                
                return result
            except Exception as e:
                # Log operation failure
                log.error(f"Failed {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def enforce_version(version: str):
    """Decorator to enforce model version."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: OnyxBaseModel, *args, **kwargs) -> Any:
            if self.version != version:
                raise ValueError(
                    f"Model version mismatch: expected {version}, got {self.version}"
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def validate_schema(schema: Dict[str, Any]):
    """Decorator to validate model against a schema."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: OnyxBaseModel, *args, **kwargs) -> Any:
            # Validate against schema
            data = self.model_dump()
            for field, rules in schema.items():
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
                
                value = data[field]
                if "type" in rules and not isinstance(value, rules["type"]):
                    raise ValueError(
                        f"Invalid type for {field}: expected {rules['type']}, got {type(value)}"
                    )
                
                if "min" in rules and value < rules["min"]:
                    raise ValueError(f"{field} must be greater than {rules['min']}")
                
                if "max" in rules and value > rules["max"]:
                    raise ValueError(f"{field} must be less than {rules['max']}")
                
                if "pattern" in rules and not rules["pattern"].match(str(value)):
                    raise ValueError(f"{field} does not match pattern: {rules['pattern']}")
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# Example usage:
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define schema
user_schema = {
    "name": {"type": str},
    "email": {"type": str, "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
    "age": {"type": int, "min": 0, "max": 150}
}

# Create model with decorators
@register_model
class UserModel(OnyxBaseModel):
    name: str
    email: str
    age: Optional[int] = None
    
    @validate_model(validate_types=True, validate_custom=True)
    @cache_model(key_field="email")
    @track_changes
    @require_active
    @log_operations(logger)
    @enforce_version("1.0.0")
    @validate_schema(user_schema)
    def update_profile(self, name: str, email: str, age: Optional[int] = None) -> None:
        self.name = name
        self.email = email
        self.age = age
        self.updated_at = datetime.utcnow()

# Create and use model
user = UserModel(
    name="John",
    email="john@example.com",
    age=30
)

# Update profile with decorators
user.update_profile(
    name="John Updated",
    email="john.updated@example.com",
    age=31
)
""" 