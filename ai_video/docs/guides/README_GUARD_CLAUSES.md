# 🛡️ Guard Clauses & Early Validation

## Overview

The AI Video System implements a comprehensive guard clause and early validation system that follows the "fail fast" principle. This system validates all inputs and parameters at the beginning of functions before any processing begins, ensuring robust error handling and preventing invalid operations.

## Key Principles

### 1. Fail Fast
- Detect errors as early as possible
- Validate inputs before processing
- Prevent invalid operations from starting

### 2. Guard Clauses
- Check conditions at the beginning of functions
- Return early if conditions are not met
- Make code more readable and maintainable

### 3. Early Validation
- Validate all parameters before processing
- Use structured validation rules
- Provide clear error messages

## Quick Start

```python
from ai_video.guard_clauses import (
    guard_validation, guard_resources, fail_fast, require_not_none
)
from ai_video.early_validation import (
    early_validate, ValidationRule, ValidationType
)

# Basic guard clause
@guard_validation([
    lambda video_path, **kwargs: Path(video_path).exists(),
    lambda batch_size, **kwargs: 0 < batch_size <= 32
])
def process_video(video_path: str, batch_size: int):
    # Function body - only executes if validations pass
    pass

# Early validation
@early_validate({
    "model_path": ValidationRule(
        name="model_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Model '{field}' does not exist",
        required=True
    )
})
def load_model(model_path: str):
    # Function body - only executes if validations pass
    pass

# Fail fast helpers
def process_data(data: np.ndarray):
    require_not_none(data, "data")
    fail_fast(data.size > 0, "Data cannot be empty")
    # Function body
    pass
```

## Guard Clause Types

### 1. Validation Guards
```python
from ai_video.guard_clauses import ValidationGuards

@guard_validation([
    lambda video_path, **kwargs: ValidationGuards.validate_file_exists(video_path),
    lambda batch_size, **kwargs: ValidationGuards.validate_batch_size(batch_size, 32),
    lambda video_data, **kwargs: ValidationGuards.validate_video_data(video_data)
])
def process_video_pipeline(video_path: str, batch_size: int, video_data: np.ndarray):
    # Process video
    pass
```

### 2. Resource Guards
```python
from ai_video.guard_clauses import guard_resources

@guard_resources(
    required_memory_mb=2048.0,
    required_disk_gb=1.0,
    max_cpu_percent=90.0
)
def memory_intensive_operation():
    # Memory-intensive operation
    pass
```

### 3. State Guards
```python
from ai_video.guard_clauses import guard_state

def check_system_ready():
    return system_initialized and not processing

@guard_state(
    state_checker=check_system_ready,
    error_message="System not ready for processing"
)
def process_request():
    # Process request
    pass
```

## Early Validation System

### Validation Rules
```python
from ai_video.early_validation import (
    ValidationRule, ValidationType, ValidationLevel
)

# Create validation rule
rule = ValidationRule(
    name="file_exists",
    validation_type=ValidationType.EXISTENCE,
    validator=lambda x: Path(x).exists(),
    error_message="File '{field}' does not exist",
    level=ValidationLevel.STRICT,
    required=True
)
```

### Validation Types
- `TYPE`: Type checking (string, int, float, etc.)
- `RANGE`: Range validation (min/max values)
- `FORMAT`: Format validation (email, URL, file format)
- `EXISTENCE`: Existence validation (file exists, not None)
- `SIZE`: Size validation (length, dimensions)
- `CONTENT`: Content validation (no NaN, required keys)
- `RELATIONSHIP`: Relationship validation (array compatibility)
- `CUSTOM`: Custom validation functions

### Validation Levels
- `STRICT`: Fail immediately on validation error
- `NORMAL`: Log warning on validation error
- `LENIENT`: Continue with warning

## Built-in Validators

### Type Validators
```python
from ai_video.early_validation import TypeValidators

validators = [
    TypeValidators.is_string,
    TypeValidators.is_integer,
    TypeValidators.is_float,
    TypeValidators.is_boolean,
    TypeValidators.is_list,
    TypeValidators.is_dict,
    TypeValidators.is_numpy_array,
    TypeValidators.is_path,
    TypeValidators.is_callable,
    TypeValidators.is_async_callable
]
```

### Range Validators
```python
from ai_video.early_validation import RangeValidators

validators = [
    lambda x: RangeValidators.in_range(x, 0, 100),
    RangeValidators.positive,
    RangeValidators.non_negative,
    RangeValidators.between_zero_one,
    RangeValidators.power_of_two,
    RangeValidators.even,
    RangeValidators.odd
]
```

### Format Validators
```python
from ai_video.early_validation import FormatValidators

validators = [
    FormatValidators.is_email,
    FormatValidators.is_url,
    FormatValidators.is_filename,
    FormatValidators.is_video_format,
    FormatValidators.is_image_format,
    FormatValidators.is_json_string,
    FormatValidators.is_hex_color
]
```

### Existence Validators
```python
from ai_video.early_validation import ExistenceValidators

validators = [
    ExistenceValidators.file_exists,
    ExistenceValidators.directory_exists,
    ExistenceValidators.file_readable,
    ExistenceValidators.file_writable,
    ExistenceValidators.directory_writable,
    ExistenceValidators.not_empty,
    ExistenceValidators.not_none
]
```

### Size Validators
```python
from ai_video.early_validation import SizeValidators

validators = [
    lambda x: SizeValidators.string_length(x, 1, 1000),
    lambda x: SizeValidators.list_length(x, 1, 100),
    lambda x: SizeValidators.array_shape(x, (100, 256, 256, 3)),
    lambda x: SizeValidators.array_size(x, 1, 1000000),
    lambda x: SizeValidators.file_size(x, 100.0),  # 100MB
    lambda x: SizeValidators.memory_usage(x, 1024.0)  # 1GB
]
```

### Content Validators
```python
from ai_video.early_validation import ContentValidators

validators = [
    ContentValidators.no_nan_values,
    ContentValidators.no_inf_values,
    lambda x: ContentValidators.in_range_values(x, 0.0, 1.0),
    ContentValidators.positive_values,
    ContentValidators.normalized_values,
    ContentValidators.unique_values,
    lambda x: ContentValidators.valid_keys(x, {"key1", "key2"}),
    lambda x: ContentValidators.required_keys(x, {"required_key"})
]
```

## Validation Schemas

### Pre-built Schemas
```python
from ai_video.early_validation import (
    create_video_validation_schema,
    create_model_validation_schema,
    create_data_validation_schema
)

# Video validation schema
video_schema = create_video_validation_schema()
@video_schema.to_decorator()
def process_video(video_path: str, batch_size: int):
    pass

# Model validation schema
model_schema = create_model_validation_schema()
@model_schema.to_decorator()
def load_model(model_path: str, model_config: dict):
    pass

# Data validation schema
data_schema = create_data_validation_schema()
@data_schema.to_decorator()
def process_data(data: np.ndarray):
    pass
```

### Custom Schemas
```python
from ai_video.early_validation import ValidationSchema, ValidationRule

def create_custom_schema():
    schema = ValidationSchema("custom_validation")
    
    schema.add_rule("user_input", ValidationRule(
        name="input_not_empty",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: len(x) > 0,
        error_message="User input '{field}' cannot be empty",
        required=True
    ))
    
    schema.add_rule("quality", ValidationRule(
        name="quality_range",
        validation_type=ValidationType.RANGE,
        validator=lambda x: 0.0 <= x <= 1.0,
        error_message="Quality '{field}' must be between 0 and 1",
        required=True
    ))
    
    return schema

custom_schema = create_custom_schema()
@custom_schema.to_decorator()
def process_user_request(user_input: str, quality: float):
    pass
```

## Fail Fast Helpers

### Basic Helpers
```python
from ai_video.guard_clauses import (
    fail_fast, require_not_none, require_not_empty,
    require_file_exists, require_valid_range
)

def process_data(data: np.ndarray, batch_size: int):
    # Fail fast validation
    require_not_none(data, "data")
    require_not_empty(data, "data")
    require_valid_range(batch_size, 1, 32, "batch_size")
    
    # Function body
    pass
```

### Custom Fail Fast
```python
from ai_video.guard_clauses import fail_fast
from ai_video.error_handling import ValidationError

def process_video(video_path: str, quality: float):
    # Custom fail fast conditions
    fail_fast(Path(video_path).exists(), f"Video file not found: {video_path}")
    fail_fast(0.0 <= quality <= 1.0, f"Quality must be between 0 and 1, got: {quality}")
    fail_fast(quality > 0.5, "Quality too low for processing", ValidationError)
    
    # Function body
    pass
```

## Context Managers

### Guard Context
```python
from ai_video.guard_clauses import guard_context, get_guard_manager

def process_with_guard_context():
    guard_manager = get_guard_manager()
    
    with guard_context(guard_manager, "video_processing"):
        # System state is verified before entering
        # Process video
        pass
```

### Resource Guard Context
```python
from ai_video.guard_clauses import resource_guard_context

def memory_intensive_operation():
    with resource_guard_context(required_memory_mb=1024.0):
        # Memory is verified before entering
        # Garbage collection is forced after exiting
        pass
```

## Async Support

### Async Guard Clauses
```python
@guard_validation([
    lambda video_path, **kwargs: Path(video_path).exists()
])
async def async_process_video(video_path: str):
    # Async function with guard clauses
    await asyncio.sleep(1)
    return {"processed": True}
```

### Async Early Validation
```python
@early_validate({
    "model_path": ValidationRule(
        name="model_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Model '{field}' does not exist",
        required=True
    )
})
async def async_load_model(model_path: str):
    # Async function with early validation
    await asyncio.sleep(1)
    return {"loaded": True}
```

## Best Practices

### 1. Validate Early
```python
# Good: Validate at the beginning
def process_video(video_path: str, quality: float):
    # Validate inputs first
    require_file_exists(video_path, "video_path")
    require_valid_range(quality, 0.0, 1.0, "quality")
    
    # Then process
    return process_video_implementation(video_path, quality)

# Bad: Validate in the middle
def process_video(video_path: str, quality: float):
    # Process first
    result = process_video_implementation(video_path, quality)
    
    # Validate later (too late!)
    if not Path(video_path).exists():
        raise ValueError("File not found")
```

### 2. Use Appropriate Validation Levels
```python
# Strict validation for critical operations
@early_validate({
    "model_path": ValidationRule(
        name="model_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Model '{field}' does not exist",
        level=ValidationLevel.STRICT,  # Fail immediately
        required=True
    )
})
def load_critical_model(model_path: str):
    pass

# Lenient validation for optional features
@early_validate({
    "cache_path": ValidationRule(
        name="cache_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Cache '{field}' not found",
        level=ValidationLevel.LENIENT,  # Continue with warning
        required=False
    )
})
def process_with_cache(cache_path: str = None):
    pass
```

### 3. Combine Multiple Validation Types
```python
@early_validate({
    "video_path": ValidationRule(
        name="video_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Video '{field}' does not exist",
        required=True
    ),
    "video_path": ValidationRule(
        name="video_format",
        validation_type=ValidationType.FORMAT,
        validator=lambda x: Path(x).suffix.lower() in {'.mp4', '.avi'},
        error_message="Video '{field}' format not supported",
        required=True
    ),
    "batch_size": ValidationRule(
        name="batch_range",
        validation_type=ValidationType.RANGE,
        validator=lambda x: 1 <= x <= 32,
        error_message="Batch size '{field}' must be between 1 and 32",
        required=True
    )
})
def process_video(video_path: str, batch_size: int):
    pass
```

### 4. Use Guard Clauses for Complex Conditions
```python
@guard_validation([
    # Check file exists
    lambda video_path, **kwargs: Path(video_path).exists(),
    # Check file size
    lambda video_path, **kwargs: Path(video_path).stat().st_size < 1024 * 1024 * 100,  # 100MB
    # Check memory available
    lambda **kwargs: psutil.virtual_memory().available > 1024 * 1024 * 1024,  # 1GB
    # Check system not overloaded
    lambda **kwargs: psutil.cpu_percent() < 90.0
])
def process_video_safely(video_path: str):
    pass
```

### 5. Create Reusable Validation Schemas
```python
def create_video_processing_schema():
    """Create reusable schema for video processing."""
    schema = ValidationSchema("video_processing")
    
    # Common video validations
    schema.add_rule("video_path", ValidationRule(
        name="video_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Video '{field}' does not exist",
        required=True
    ))
    
    schema.add_rule("quality", ValidationRule(
        name="quality_range",
        validation_type=ValidationType.RANGE,
        validator=lambda x: 0.0 <= x <= 1.0,
        error_message="Quality '{field}' must be between 0 and 1",
        required=True
    ))
    
    return schema

# Use the schema in multiple functions
video_schema = create_video_processing_schema()

@video_schema.to_decorator()
def process_video(video_path: str, quality: float):
    pass

@video_schema.to_decorator()
def enhance_video(video_path: str, quality: float):
    pass
```

## Error Handling

### Custom Error Messages
```python
@early_validate({
    "model_path": ValidationRule(
        name="model_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Model file '{field}' not found. Please check the path: {value}",
        required=True
    )
})
def load_model(model_path: str):
    pass
```

### Error Categories
```python
from ai_video.error_handling import ErrorCategory

@early_validate({
    "video_path": ValidationRule(
        name="video_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: Path(x).exists(),
        error_message="Video '{field}' does not exist",
        required=True
    )
}, error_category=ErrorCategory.DATA_VALIDATION)
def process_video(video_path: str):
    pass
```

## Performance Considerations

### Validation Overhead
- Guard clauses add minimal overhead when conditions are met
- Early validation prevents expensive operations on invalid data
- Use appropriate validation levels to balance safety and performance

### Caching Validations
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def validate_model_path(model_path: str) -> bool:
    """Cache model path validation results."""
    return Path(model_path).exists()

@early_validate({
    "model_path": ValidationRule(
        name="model_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=validate_model_path,  # Cached validation
        error_message="Model '{field}' does not exist",
        required=True
    )
})
def load_model(model_path: str):
    pass
```

## Integration with Error Handling

### Combined Error Handling and Validation
```python
from ai_video.error_handling import handle_errors, retry_on_error
from ai_video.guard_clauses import guard_validation

@handle_errors(error_types=[ValidationError, FileNotFoundError])
@retry_on_error(max_retries=3, delay=1.0)
@guard_validation([
    lambda video_path, **kwargs: Path(video_path).exists()
])
def robust_video_processing(video_path: str):
    # Multiple layers of protection
    pass
```

### Error Recovery with Validation
```python
from ai_video.error_handling import safe_execute
from ai_video.guard_clauses import require_file_exists

def process_video_with_fallback(video_path: str, fallback_path: str):
    try:
        require_file_exists(video_path, "video_path")
        return process_video(video_path)
    except ValidationError:
        # Try fallback path
        require_file_exists(fallback_path, "fallback_path")
        return process_video(fallback_path)
```

## Testing

### Unit Testing Guard Clauses
```python
import pytest
from ai_video.guard_clauses import fail_fast, require_not_none

def test_fail_fast():
    # Test valid case
    require_not_none("valid", "test")
    
    # Test invalid case
    with pytest.raises(ValidationError):
        require_not_none(None, "test")

def test_guard_validation():
    @guard_validation([
        lambda x, **kwargs: x > 0
    ])
    def test_function(x: int):
        return x * 2
    
    # Test valid case
    assert test_function(5) == 10
    
    # Test invalid case
    with pytest.raises(ValidationError):
        test_function(-1)
```

### Integration Testing
```python
def test_video_processing_pipeline():
    # Test with valid inputs
    result = process_video("valid_video.mp4", quality=0.8)
    assert result["processed"] == True
    
    # Test with invalid inputs
    with pytest.raises(ValidationError):
        process_video("nonexistent_video.mp4", quality=0.8)
    
    with pytest.raises(ValidationError):
        process_video("valid_video.mp4", quality=1.5)  # Invalid quality
```

## Monitoring and Logging

### Validation Logging
```python
import logging

# Configure validation logging
logging.getLogger('ai_video.guard_clauses').setLevel(logging.INFO)
logging.getLogger('ai_video.early_validation').setLevel(logging.INFO)

# Validation failures will be logged
@early_validate({
    "input": ValidationRule(
        name="input_valid",
        validation_type=ValidationType.EXISTENCE,
        validator=lambda x: x is not None,
        error_message="Input '{field}' is required",
        required=True
    )
})
def process_with_logging(input_data):
    pass
```

### Validation Metrics
```python
from ai_video.guard_clauses import get_guard_manager

def get_validation_metrics():
    guard_manager = get_guard_manager()
    # Access validation metrics and statistics
    return guard_manager.get_validation_summary()
```

This comprehensive guard clause and early validation system ensures that your AI Video System is robust, maintainable, and follows best practices for error handling and input validation. 