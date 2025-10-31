# Error Factories and Custom Error Types Guide

## Overview

This guide covers the implementation of custom error types and error factories for consistent error handling throughout the video-OpusClip project. The system provides structured error management with context tracking, categorization, and integration with existing error handling.

## Table of Contents

1. [Architecture](#architecture)
2. [Error Categories](#error-categories)
3. [Custom Error Types](#custom-error-types)
4. [Error Context Management](#error-context-management)
5. [Error Factory System](#error-factory-system)
6. [Integration with Existing System](#integration-with-existing-system)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Testing Strategy](#testing-strategy)
10. [Performance Considerations](#performance-considerations)
11. [Migration Guide](#migration-guide)
12. [Troubleshooting](#troubleshooting)

## Architecture

### Core Components

The error factory system consists of several key components:

1. **Error Categories**: Enumeration of error types for classification
2. **Error Context**: Structured context information for error tracking
3. **Custom Error Types**: Domain-specific error classes
4. **Error Factory**: Factory pattern for creating consistent errors
5. **Context Manager**: Manages error context throughout request lifecycle
6. **Utility Functions**: Helper functions for common operations

### System Flow

```
Request → Context Manager → Error Factory → Custom Error → Error Handler → Response
```

## Error Categories

### Category Enumeration

```python
class ErrorCategory(Enum):
    # Input and Validation
    VALIDATION = "validation"
    INPUT = "input"
    FORMAT = "format"
    
    # Processing
    PROCESSING = "processing"
    ENCODING = "encoding"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    
    # Resources
    RESOURCE = "resource"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    
    # External Services
    EXTERNAL = "external"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    
    # System
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    CRITICAL = "critical"
    
    # AI/ML Specific
    MODEL = "model"
    INFERENCE = "inference"
    TRAINING = "training"
    PIPELINE = "pipeline"
```

### Category Usage

Categories help organize errors and provide consistent handling:

```python
# Error with category
error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)
assert error.category == ErrorCategory.VALIDATION
```

## Custom Error Types

### Base Error Types

#### VideoValidationError
For input validation failures:

```python
error = VideoValidationError(
    message="Invalid YouTube URL format",
    field="youtube_url",
    value="invalid_url",
    context=error_context
)
```

#### VideoProcessingError
For general processing failures:

```python
error = VideoProcessingError(
    message="Video processing failed",
    operation="video_encoding",
    context=error_context,
    details={"video_id": "abc123"}
)
```

#### VideoEncodingError
For video encoding specific errors:

```python
error = VideoEncodingError(
    message="Encoding failed",
    video_id="video-123",
    context=error_context,
    details={"codec": "h264"}
)
```

#### ModelInferenceError
For AI model inference errors:

```python
error = ModelInferenceError(
    message="Inference failed",
    model_name="stable_diffusion",
    context=error_context,
    details={"batch_size": 1}
)
```

#### ResourceExhaustionError
For resource-related errors:

```python
error = ResourceExhaustionError(
    message="Memory exhausted",
    resource="gpu_memory",
    available="2GB",
    required="4GB",
    context=error_context
)
```

#### APIError
For external API errors:

```python
error = APIError(
    message="API call failed",
    service="youtube",
    endpoint="/videos",
    status_code=500,
    context=error_context
)
```

#### SecurityViolationError
For security-related errors:

```python
error = SecurityViolationError(
    message="Malicious input detected",
    threat_type="injection",
    severity="high",
    context=error_context
)
```

## Error Context Management

### ErrorContext Class

The `ErrorContext` class provides structured context information:

```python
@dataclass
class ErrorContext:
    # Request information
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Operation information
    operation: Optional[str] = None
    component: Optional[str] = None
    step: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    
    # Timing information
    start_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Context Manager

The `ErrorContextManager` provides lifecycle management:

```python
# Initialize manager
manager = ErrorContextManager()

# Set request context
manager.set_request_context("request-123", "user-456", "session-789")

# Set operation context
manager.set_operation_context("video_processing", "api", "validation")

# Start timing
manager.start_timing()

# Add metadata
manager.add_metadata("test_key", "test_value")

# End timing
manager.end_timing()

# Get context
context = manager.get_context()
```

## Error Factory System

### ErrorFactory Class

The factory provides consistent error creation:

```python
class ErrorFactory:
    def __init__(self):
        self.error_registry: Dict[str, Type[VideoProcessingError]] = {}
        self._register_default_errors()
    
    def register_error(self, error_type: str, error_class: Type[VideoProcessingError]):
        """Register a custom error type."""
        
    def create_error(self, error_type: str, message: str, **kwargs) -> VideoProcessingError:
        """Create an error instance of the specified type."""
        
    def create_validation_error(self, field: str, value: Any, message: str, context: Optional[ErrorContext] = None):
        """Create a validation error."""
        
    def create_processing_error(self, operation: str, message: str, context: Optional[ErrorContext] = None, details: Optional[Dict[str, Any]] = None):
        """Create a processing error."""
```

### Factory Usage

```python
# Create factory
factory = ErrorFactory()

# Create validation error
error = factory.create_validation_error(
    field="youtube_url",
    value="invalid_url",
    message="Invalid URL format",
    context=error_context
)

# Create processing error
error = factory.create_processing_error(
    operation="video_encoding",
    message="Encoding failed",
    context=error_context,
    details={"video_id": "abc123"}
)

# Create custom error type
error = factory.create_error(
    "validation",
    "Invalid input",
    field="test",
    value="invalid",
    context=error_context
)
```

## Integration with Existing System

### Enhanced Error Handler

The `ErrorHandler` class has been enhanced to work with custom errors:

```python
class ErrorHandler:
    def __init__(self):
        # Initialize error factory and context manager
        self.error_factory = error_factory
        self.context_manager = context_manager
    
    def create_error_with_context(self, error_type: str, message: str, context: Optional[ErrorContext] = None, **kwargs):
        """Create an error using the error factory with context."""
        
    def set_request_context(self, request_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """Set request context for error tracking."""
        
    def handle_custom_error(self, error: VideoProcessingError, request_id: Optional[str] = None) -> ErrorResponse:
        """Handle custom error types with enhanced context."""
```

### API Integration

API endpoints can use error factories for consistent error handling:

```python
@app.post("/api/v1/video/process")
async def process_video(request: VideoClipRequest, req: Request = None):
    # Set operation context for error tracking
    if context_manager:
        context_manager.set_operation_context("video_processing", "api", "process_video")
        error_handler.set_operation_context("video_processing", "api", "process_video")
    
    # GUARD CLAUSE: Check for None/empty request
    if request is None:
        context = create_error_context(request_id=request_id, operation="video_processing", component="api")
        raise create_validation_error("request", None, "Request cannot be None", context)
    
    # Process video...
```

### Gradio Demo Integration

The Gradio demo uses error factories for image generation:

```python
def generate_image(prompt: str, guidance_scale: float = 7.5, num_inference_steps: int = 30):
    # Set operation context for error tracking
    if context_manager:
        context_manager.set_operation_context("image_generation", "gradio_demo", "generate_image")
        context_manager.start_timing()
    
    # GUARD CLAUSE: Check for None/empty prompt
    if prompt is None:
        context = create_error_context(operation="image_generation", component="gradio_demo", step="prompt_validation")
        error = create_validation_error("prompt", None, "Prompt cannot be None", context)
        return None, error.message
    
    # Generate image...
```

## Usage Examples

### Basic Error Creation

```python
# Create error with context
context = create_error_context(
    request_id="request-123",
    operation="video_processing",
    component="api"
)

error = create_validation_error(
    field="youtube_url",
    value="invalid_url",
    message="Invalid URL format",
    context=context
)
```

### Error with Timing

```python
# Set up context manager
context_manager.set_request_context("request-123")
context_manager.set_operation_context("video_processing")
context_manager.start_timing()

try:
    # Process video
    result = process_video(video_data)
    
    # End timing
    context_manager.end_timing()
    
    return result
    
except Exception as e:
    # Enrich error with context
    enrich_error_with_context(e, context_manager.get_context())
    raise
```

### Error Factory with Custom Types

```python
# Register custom error type
error_factory.register_error("custom_error", VideoValidationError)

# Create custom error
error = error_factory.create_error(
    "custom_error",
    "Custom validation failed",
    field="custom_field",
    value="invalid_value",
    context=error_context
)
```

### Error Summary Generation

```python
# Create error
error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)

# Get error summary
summary = get_error_summary(error)

# Summary contains:
# - error_type: "VideoValidationError"
# - error_message: "Invalid URL"
# - error_category: ErrorCategory.VALIDATION
# - timestamp: ISO format timestamp
# - traceback: Full stack trace
# - context: Error context as dictionary
```

## Best Practices

### 1. Always Use Context

```python
# Good: Error with context
context = create_error_context(request_id=request_id, operation="video_processing")
error = create_validation_error("youtube_url", "invalid_url", "Invalid URL", context)

# Bad: Error without context
error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url")
```

### 2. Use Appropriate Error Types

```python
# For validation errors
error = create_validation_error("field", "value", "message", context)

# For processing errors
error = create_processing_error("operation", "message", context, details)

# For resource errors
error = create_resource_error("message", "resource", "available", "required", context)

# For API errors
error = create_api_error("message", "service", "endpoint", status_code, context)
```

### 3. Set Operation Context Early

```python
# Set context at the beginning of operations
if context_manager:
    context_manager.set_operation_context("video_processing", "api", "process_video")
    context_manager.start_timing()
```

### 4. Enrich Existing Errors

```python
try:
    # Some operation
    result = some_operation()
except Exception as e:
    # Enrich with context
    if context_manager:
        enrich_error_with_context(e, context_manager.get_context())
    raise
```

### 5. Use Timing for Performance Monitoring

```python
# Start timing
context_manager.start_timing()

try:
    # Operation
    result = perform_operation()
finally:
    # End timing
    context_manager.end_timing()
```

## Testing Strategy

### Unit Tests

Test individual components:

```python
def test_error_factory_creation():
    factory = ErrorFactory()
    context = ErrorContext(request_id="test-123")
    
    error = factory.create_validation_error(
        "youtube_url", "invalid_url", "Invalid URL", context
    )
    
    assert isinstance(error, VideoValidationError)
    assert error.message == "Invalid URL"
    assert error.context == context
```

### Integration Tests

Test integration with existing systems:

```python
def test_error_handler_with_custom_errors():
    handler = ErrorHandler()
    context = ErrorContext(request_id="test-123", operation="video_processing")
    
    error = VideoValidationError("Invalid URL", "youtube_url", "invalid_url", context)
    response = handler.handle_custom_error(error, "test-123")
    
    assert response.message is not None
    assert response.request_id == "test-123"
    assert "error_category" in response.details
```

### Performance Tests

Test performance characteristics:

```python
def test_error_creation_performance():
    factory = ErrorFactory()
    context = ErrorContext(request_id="test-123")
    
    start_time = time.perf_counter()
    
    for _ in range(1000):
        factory.create_validation_error("test_field", "test_value", "Test error", context)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Should be very fast
    assert duration < 1.0
```

## Performance Considerations

### Error Creation Overhead

- Error creation is very fast (< 1ms per error)
- Context management adds minimal overhead
- Factory pattern provides efficient error instantiation

### Memory Usage

- Error objects are lightweight
- Context objects use minimal memory
- No memory leaks from context management

### Optimization Tips

1. **Reuse Context**: Create context once and reuse for multiple errors
2. **Lazy Context Creation**: Only create context when needed
3. **Context Pooling**: For high-frequency operations, consider context pooling

## Migration Guide

### From Basic Exceptions

**Before:**
```python
if not youtube_url:
    raise ValueError("YouTube URL is required")
```

**After:**
```python
if not youtube_url:
    context = create_error_context(request_id=request_id, operation="video_processing")
    raise create_validation_error("youtube_url", youtube_url, "YouTube URL is required", context)
```

### From Custom Exceptions

**Before:**
```python
class VideoProcessingError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

raise VideoProcessingError("Processing failed")
```

**After:**
```python
context = create_error_context(request_id=request_id, operation="video_processing")
raise create_processing_error("video_processing", "Processing failed", context)
```

### From Error Handler

**Before:**
```python
def handle_error(error, request_id):
    return {
        "error": str(error),
        "request_id": request_id
    }
```

**After:**
```python
def handle_error(error, request_id):
    if hasattr(error, 'context'):
        enrich_error_with_context(error, context_manager.get_context())
    
    summary = get_error_summary(error)
    return {
        "error": summary,
        "request_id": request_id
    }
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'error_factories'`

**Solution:** Ensure the module is in the Python path and properly imported:

```python
try:
    from .error_factories import error_factory, context_manager
except ImportError:
    error_factory = None
    context_manager = None
```

#### 2. Context Not Available

**Problem:** Error context is None or empty

**Solution:** Ensure context is properly set:

```python
# Set context before operations
if context_manager:
    context_manager.set_request_context(request_id)
    context_manager.set_operation_context(operation)
```

#### 3. Error Factory Not Registered

**Problem:** `ValueError: Unknown error type`

**Solution:** Register the error type or use existing types:

```python
# Register custom error type
error_factory.register_error("custom_type", CustomErrorClass)

# Or use existing types
error = error_factory.create_error("validation", "message", **kwargs)
```

#### 4. Performance Issues

**Problem:** Slow error creation

**Solution:** Check for unnecessary context creation:

```python
# Good: Reuse context
context = create_error_context(request_id=request_id)
for field in fields:
    error = create_validation_error(field, value, message, context)

# Bad: Create context for each error
for field in fields:
    context = create_error_context(request_id=request_id)
    error = create_validation_error(field, value, message, context)
```

### Debugging Tips

1. **Enable Debug Logging**: Set log level to DEBUG for detailed error information
2. **Check Context**: Verify context is properly set before error creation
3. **Validate Error Types**: Ensure error types are registered in the factory
4. **Monitor Performance**: Use timing functions to identify bottlenecks

### Error Patterns

#### Pattern 1: Validation Errors
```python
# Use for input validation failures
context = create_error_context(request_id=request_id, operation="validation")
error = create_validation_error("field", "value", "message", context)
```

#### Pattern 2: Processing Errors
```python
# Use for processing failures
context = create_error_context(request_id=request_id, operation="processing")
error = create_processing_error("operation", "message", context, details)
```

#### Pattern 3: Resource Errors
```python
# Use for resource exhaustion
context = create_error_context(request_id=request_id, operation="resource_check")
error = create_resource_error("message", "resource", "available", "required", context)
```

#### Pattern 4: API Errors
```python
# Use for external service failures
context = create_error_context(request_id=request_id, operation="api_call")
error = create_api_error("message", "service", "endpoint", status_code, context)
```

## Conclusion

The error factory system provides a robust, consistent, and maintainable approach to error handling in the video-OpusClip project. By using custom error types, structured context management, and factory patterns, the system ensures:

- **Consistency**: All errors follow the same structure and patterns
- **Context**: Rich error context for debugging and monitoring
- **Maintainability**: Easy to extend and modify error handling
- **Performance**: Efficient error creation and management
- **Integration**: Seamless integration with existing error handling systems

The system is designed to scale with the application and provides the foundation for enterprise-grade error management. 