# Error Handling and Validation Guide

## Overview

The video-OpusClip project implements a comprehensive error handling and validation system designed to provide clear, actionable error messages while maintaining system stability and observability.

## Architecture

### Error Handling Components

1. **Error Codes** - Standardized error codes for different error types
2. **Custom Exceptions** - Domain-specific exceptions with rich context
3. **Error Handler** - Centralized error processing and logging
4. **Validation Functions** - Input validation with detailed error messages
5. **Error Response Model** - Standardized error response format

## Error Codes

### Validation Errors (1000-1999)
- `1001` - Invalid YouTube URL
- `1002` - Invalid language code
- `1003` - Invalid clip length
- `1004` - Invalid viral score
- `1005` - Invalid caption
- `1006` - Invalid variant ID
- `1007` - Invalid audience profile
- `1008` - Invalid batch size

### Processing Errors (2000-2999)
- `2001` - Video processing failed
- `2002` - LangChain processing failed
- `2003` - Viral analysis failed
- `2004` - Batch processing failed
- `2005` - Cache operation failed
- `2006` - Parallel processing failed

### External Service Errors (3000-3999)
- `3001` - YouTube API error
- `3002` - LangChain API error
- `3003` - Redis connection error
- `3004` - Database error
- `3005` - File storage error

### Resource Errors (4000-4999)
- `4001` - Insufficient memory
- `4002` - GPU not available
- `4003` - Disk space full
- `4004` - Rate limit exceeded
- `4005` - Timeout error

### Configuration Errors (5000-5999)
- `5001` - Missing configuration
- `5002` - Invalid configuration
- `5003` - Environment error

### Security Errors (6000-6999)
- `6001` - Unauthorized access
- `6002` - Invalid token
- `6003` - Rate limit violation

### Unknown Errors (9000-9999)
- `9001` - Unknown error
- `9002` - Internal server error

## Custom Exceptions

### ValidationError
Raised when input validation fails.

```python
from .error_handling import ValidationError, ErrorCode

# Create validation error
error = ValidationError("Invalid YouTube URL", "youtube_url", "invalid-url")
error.error_code = ErrorCode.INVALID_YOUTUBE_URL
```

### ProcessingError
Raised when video processing operations fail.

```python
from .error_handling import ProcessingError, ErrorCode

# Create processing error
error = ProcessingError("Video processing failed", "process_video", {"duration": 10})
error.error_code = ErrorCode.VIDEO_PROCESSING_FAILED
```

### ExternalServiceError
Raised when external service calls fail.

```python
from .error_handling import ExternalServiceError, ErrorCode

# Create external service error
error = ExternalServiceError("API timeout", "youtube_api", 408)
error.error_code = ErrorCode.YOUTUBE_API_ERROR
```

### ResourceError
Raised when system resources are insufficient.

```python
from .error_handling import ResourceError, ErrorCode

# Create resource error
error = ResourceError("Memory insufficient", "gpu_memory", "4GB", "8GB")
error.error_code = ErrorCode.INSUFFICIENT_MEMORY
```

## Error Handler

The `ErrorHandler` class provides centralized error processing with logging and response formatting.

```python
from .error_handling import ErrorHandler

handler = ErrorHandler()

# Handle different error types
validation_response = handler.handle_validation_error(error, request_id)
processing_response = handler.handle_processing_error(error, request_id)
external_response = handler.handle_external_service_error(error, request_id)
resource_response = handler.handle_resource_error(error, request_id)
unknown_response = handler.handle_unknown_error(error, request_id)
```

## Validation Functions

### Basic Validation

```python
from .validation import (
    validate_youtube_url,
    validate_language_code,
    validate_clip_length,
    validate_viral_score,
    validate_caption,
    validate_variant_id,
    validate_audience_profile,
    validate_batch_size
)

# Validate individual fields
validate_youtube_url("https://www.youtube.com/watch?v=123")
validate_language_code("en")
validate_clip_length(60)
validate_viral_score(0.8)
validate_caption("Test caption")
validate_variant_id("var-1")
validate_audience_profile({"age": "18-24", "interests": ["tech"]})
validate_batch_size(10)
```

### Composite Validation

```python
from .validation import (
    validate_video_request_data,
    validate_viral_variant_data,
    validate_batch_request_data
)

# Validate complete request data
validate_video_request_data(
    youtube_url="https://www.youtube.com/watch?v=123",
    language="en",
    max_clip_length=60,
    min_clip_length=10,
    audience_profile={"age": "18-24", "interests": ["tech"]}
)

# Validate viral variant data
validate_viral_variant_data(
    start=0.0,
    end=10.0,
    caption="Test caption",
    viral_score=0.8,
    variant_id="var-1"
)

# Validate batch request data
validate_batch_request_data(
    requests=[{"youtube_url": "https://youtube.com/watch?v=123", "language": "en"}],
    batch_size=1
)
```

### URL Utilities

```python
from .validation import (
    sanitize_youtube_url,
    extract_youtube_video_id,
    validate_and_sanitize_url
)

# Sanitize URL (remove tracking parameters, ensure HTTPS)
clean_url = sanitize_youtube_url("http://youtube.com/watch?v=123&utm_source=test")

# Extract video ID
video_id = extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Validate and sanitize in one operation
clean_url = validate_and_sanitize_url("https://youtube.com/watch?v=123&utm_source=test")
```

## Error Response Format

All errors return a standardized response format:

```json
{
  "error": {
    "code": 1001,
    "message": "Invalid YouTube URL",
    "details": {
      "field": "youtube_url",
      "value": "invalid-url"
    },
    "timestamp": 1640995200.0,
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## API Integration

### FastAPI Exception Handlers

The API includes comprehensive exception handlers for all error types:

```python
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Validation exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_validation_error(exc, request_id)
    
    return JSONResponse(
        status_code=400,
        content=error_response.to_dict()
    )
```

### Request ID Tracking

All requests include a unique request ID for tracking:

```python
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

### Decorators

Use decorators for automatic error handling:

```python
from .error_handling import validate_request, handle_processing_errors

@app.post("/api/v1/video/process")
@handle_processing_errors
async def process_video(request: VideoClipRequest):
    # Validation and processing with automatic error handling
    validate_video_request_data(**request.dict())
    return process_video_internal(request)
```

## Logging

All errors are logged with structured logging:

```python
logger.error(
    "Video processing failed", 
    error=str(e),
    processing_time=processing_time,
    request_id=request_id
)
```

## Testing

### Running Tests

```bash
# Run all error handling tests
pytest tests/test_error_handling.py -v

# Run specific test categories
pytest tests/test_error_handling.py::test_validation -v
pytest tests/test_error_handling.py::test_error_handler -v
```

### Test Coverage

The test suite covers:
- Error code enumeration
- Custom exception creation and properties
- Error response serialization
- Error handler functionality
- All validation functions
- Composite validation scenarios
- URL utility functions
- Error creation utilities
- Integration scenarios

## Best Practices

### 1. Use Specific Error Codes
Always use the most specific error code for the situation:

```python
# Good
error.error_code = ErrorCode.INVALID_YOUTUBE_URL

# Avoid
error.error_code = ErrorCode.UNKNOWN_ERROR
```

### 2. Provide Detailed Error Messages
Include context in error messages:

```python
# Good
raise ValidationError(f"Invalid YouTube URL format: {url}", "youtube_url", url)

# Avoid
raise ValidationError("Invalid URL", "youtube_url")
```

### 3. Validate Early
Validate inputs as early as possible in the request lifecycle:

```python
@app.post("/api/v1/video/process")
async def process_video(request: VideoClipRequest):
    # Validate immediately
    validate_video_request_data(**request.dict())
    
    # Process with confidence
    return process_video_internal(request)
```

### 4. Use Request ID Tracking
Always include request IDs in error responses for debugging:

```python
logger.error("Processing failed", request_id=request.state.request_id)
```

### 5. Handle Resource Errors Gracefully
Provide actionable error messages for resource issues:

```python
if not torch.cuda.is_available():
    raise ResourceError(
        "GPU not available for processing", 
        "gpu", 
        "CPU only", 
        "CUDA GPU required"
    )
```

### 6. Sanitize User Inputs
Always sanitize URLs and user inputs:

```python
# Sanitize before processing
request.youtube_url = validate_and_sanitize_url(request.youtube_url)
```

## Error Recovery

### Retry Logic
For transient errors, implement retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_with_retry(request):
    try:
        return process_video(request)
    except ExternalServiceError as e:
        if e.details.get("status_code") in [408, 429, 500, 502, 503, 504]:
            raise  # Retry
        raise  # Don't retry
```

### Circuit Breaker
For external service failures, implement circuit breaker pattern:

```python
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@breaker
def call_external_service():
    # External service call
    pass
```

## Monitoring and Alerting

### Error Metrics
Track error rates and types:

```python
from prometheus_client import Counter, Histogram

validation_errors = Counter('validation_errors_total', 'Total validation errors', ['error_code'])
processing_errors = Counter('processing_errors_total', 'Total processing errors', ['error_code'])
error_duration = Histogram('error_handling_duration_seconds', 'Error handling duration')
```

### Health Checks
Implement health checks for error handling components:

```python
@app.get("/health/error-handling")
async def error_handling_health():
    return {
        "status": "healthy",
        "error_handler": "operational",
        "validation": "operational",
        "timestamp": time.time()
    }
```

## Troubleshooting

### Common Issues

1. **ValidationError not caught**: Ensure you're using the correct exception handler
2. **Missing request ID**: Check that the middleware is properly configured
3. **Inconsistent error format**: Verify all endpoints use the ErrorResponse model
4. **Silent failures**: Ensure all exceptions are properly logged

### Debug Mode

Enable debug mode for detailed error information:

```python
if settings.DEBUG:
    error_response.details["traceback"] = traceback.format_exc()
```

## Migration Guide

### From Basic Error Handling

If migrating from basic error handling:

1. Replace generic exceptions with specific error types
2. Add error codes to all error responses
3. Implement request ID tracking
4. Update logging to use structured format
5. Add comprehensive validation

### Example Migration

```python
# Before
try:
    process_video(request)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# After
try:
    validate_video_request_data(**request.dict())
    return process_video(request)
except ValidationError as e:
    error_response = error_handler.handle_validation_error(e, request.state.request_id)
    raise HTTPException(status_code=400, detail=error_response.to_dict())
except ProcessingError as e:
    error_response = error_handler.handle_processing_error(e, request.state.request_id)
    raise HTTPException(status_code=422, detail=error_response.to_dict())
```

This comprehensive error handling system ensures robust, maintainable, and user-friendly error management throughout the video-OpusClip project. 