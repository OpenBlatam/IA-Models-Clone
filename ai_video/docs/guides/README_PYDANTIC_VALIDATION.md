# Pydantic Validation System for AI Video

## Overview

This document describes the comprehensive Pydantic BaseModel validation system implemented for the AI Video system. The system provides consistent input/output validation, response schemas, and error handling using Pydantic v2 best practices.

## Features

- **Comprehensive Input Validation**: Validate all incoming requests with detailed error messages
- **Response Schema Validation**: Ensure consistent API responses
- **Advanced Custom Validators**: Business logic validation with custom rules
- **Performance Monitoring**: Track validation performance and cache results
- **Error Handling**: Standardized error responses with detailed context
- **Middleware Integration**: Seamless integration with FastAPI middleware
- **Type Safety**: Full type hints and IDE support
- **Documentation**: Auto-generated API documentation

## Architecture

```
pydantic_schemas.py          # Core Pydantic models and schemas
pydantic_validation.py       # Validation middleware and utilities
pydantic_examples.py         # Usage examples and demonstrations
```

## Core Components

### 1. Pydantic Schemas (`pydantic_schemas.py`)

#### Enumerations
```python
class VideoStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
```

#### Input Models
```python
class VideoGenerationInput(BaseModel, BaseConfig):
    prompt: str = Field(..., min_length=1, max_length=2000)
    num_frames: int = Field(default=16, ge=8, le=128)
    height: int = Field(default=512, ge=256, le=1024)
    width: int = Field(default=512, ge=256, le=1024)
    quality: QualityLevel = Field(default=QualityLevel.MEDIUM)
    # ... additional fields with validation
```

#### Response Models
```python
class VideoGenerationResponse(BaseModel, BaseConfig):
    video_id: str = Field(..., description="Unique video identifier")
    status: VideoStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    video_url: Optional[str] = Field(None, description="Video download URL")
    # ... additional fields
```

### 2. Validation Middleware (`pydantic_validation.py`)

#### Configuration
```python
class ValidationConfig:
    enable_request_validation: bool = True
    enable_response_validation: bool = True
    enable_performance_monitoring: bool = True
    enable_validation_caching: bool = True
    max_validation_time: float = 1.0
    detailed_error_messages: bool = True
    log_validation_errors: bool = True
```

#### Middleware Usage
```python
# Create middleware with configuration
config = ValidationConfig(
    enable_request_validation=True,
    enable_response_validation=True,
    enable_performance_monitoring=True
)

middleware = create_validation_middleware(config)

# Add to FastAPI app
app.add_middleware(PydanticValidationMiddleware, config=config)
```

### 3. Validation Decorators

#### Request Validation
```python
@app.post("/api/v1/videos/generate")
@validate_request(VideoGenerationInput)
async def generate_video(request: Request, validated_data: VideoGenerationInput):
    # validated_data is guaranteed to be valid
    return await process_video_generation(validated_data)
```

#### Response Validation
```python
@app.get("/api/v1/videos/{video_id}")
@validate_response(VideoGenerationResponse)
async def get_video(video_id: str):
    # Response will be validated against VideoGenerationResponse schema
    return await get_video_data(video_id)
```

#### Input/Output Validation
```python
@app.post("/api/v1/videos/batch")
@validate_input_output(BatchGenerationInput, BatchGenerationResponse)
async def generate_batch(request: Request, validated_input: BatchGenerationInput):
    # Both input and output are validated
    return await process_batch_generation(validated_input)
```

## Usage Examples

### Basic Video Generation

```python
from .pydantic_schemas import VideoGenerationInput, VideoGenerationResponse

# Create and validate input
input_data = VideoGenerationInput(
    prompt="A beautiful sunset over mountains",
    quality=QualityLevel.HIGH,
    model_type=ModelType.STABLE_DIFFUSION
)

# Process generation
video_id = create_video_id()
response = VideoGenerationResponse(
    video_id=video_id,
    status=VideoStatus.PROCESSING,
    message="Video generation started"
)
```

### Batch Processing

```python
from .pydantic_schemas import BatchGenerationInput, BatchGenerationResponse

# Create batch request
batch_request = BatchGenerationInput(
    requests=[
        VideoGenerationInput(prompt="A cat playing", quality=QualityLevel.MEDIUM),
        VideoGenerationInput(prompt="A dog running", quality=QualityLevel.HIGH)
    ],
    batch_name="Pet Videos",
    priority=ProcessingPriority.HIGH
)

# Create batch response
batch_response = BatchGenerationResponse(
    batch_id=create_batch_id(),
    video_ids=[create_video_id() for _ in batch_request.requests],
    total_videos=len(batch_request.requests),
    status=VideoStatus.PROCESSING
)
```

### Error Handling

```python
from .pydantic_schemas import create_error_response

# Create standardized error response
error_response = create_error_response(
    error_code="QUOTA_EXCEEDED",
    error_type="quota_limit",
    message="Daily video generation limit exceeded",
    details={
        "daily_limit": 10,
        "daily_used": 10,
        "reset_time": "2024-01-01T00:00:00Z"
    }
)
```

## Advanced Features

### Custom Validators

```python
class VideoGenerationInput(BaseModel, BaseConfig):
    prompt: str = Field(..., min_length=1, max_length=2000)
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt_content(cls, v: str) -> str:
        """Validate and sanitize prompt content."""
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty')
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for inappropriate content
        inappropriate_words = ['inappropriate', 'explicit', 'nsfw']
        if any(word in v.lower() for word in inappropriate_words):
            raise ValueError('Prompt contains inappropriate content')
        
        return v
    
    @model_validator(mode='after')
    def validate_aspect_ratio(self) -> 'VideoGenerationInput':
        """Validate aspect ratio constraints."""
        aspect_ratio = self.width / self.height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            raise ValueError('Aspect ratio must be between 0.5 and 2.0')
        return self
```

### Computed Fields

```python
class VideoMetadata(BaseModel, BaseConfig):
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    file_size: int = Field(..., description="File size in bytes")
    
    @computed_field
    @property
    def resolution(self) -> str:
        """Get video resolution string."""
        return f"{self.width}x{self.height}"
    
    @computed_field
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024)
```

### Performance Monitoring

```python
from .pydantic_validation import ValidationPerformanceMonitor

# Create performance monitor
monitor = create_performance_monitor()

# Monitor validation performance
async with monitor.monitor_validation("VideoGenerationInput"):
    # Perform validation
    validated_data = VideoGenerationInput(**input_data)

# Get performance stats
stats = monitor.get_stats()
print(f"Average validation time: {stats['average_validation_time']:.4f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2f}%")
```

## Error Handling

### Validation Error Transformation

```python
from .pydantic_validation import ValidationUtils

try:
    validated_data = VideoGenerationInput(**input_data)
except ValidationError as e:
    # Transform validation error to API format
    api_error = ValidationUtils.transform_validation_error(e)
    return JSONResponse(
        status_code=422,
        content=api_error
    )
```

### Standardized Error Responses

```python
# Rate limit error
rate_limit_error = create_error_response(
    error_code="RATE_LIMIT_EXCEEDED",
    error_type="rate_limit",
    message="Too many requests",
    details={"retry_after": 60},
    retry_after=60
)

# Quota error
quota_error = create_error_response(
    error_code="QUOTA_EXCEEDED",
    error_type="quota_limit",
    message="Daily limit exceeded",
    details={"daily_limit": 10, "daily_used": 10}
)
```

## Integration with FastAPI

### Complete API Setup

```python
from fastapi import FastAPI
from .pydantic_validation import ValidationConfig, create_validation_middleware

# Create FastAPI app
app = FastAPI(title="AI Video API")

# Setup validation middleware
config = ValidationConfig(
    enable_request_validation=True,
    enable_response_validation=True,
    enable_performance_monitoring=True
)

middleware = create_validation_middleware(config)
app.add_middleware(PydanticValidationMiddleware, config=config)

# Define endpoints with validation
@app.post("/api/v1/videos/generate")
@validate_input_output(VideoGenerationInput, VideoGenerationResponse)
async def generate_video(request: Request, validated_input: VideoGenerationInput):
    """Generate video with full validation."""
    try:
        video_id = create_video_id()
        response = create_success_response(
            video_id=video_id,
            status=VideoStatus.PROCESSING,
            message="Video generation started"
        )
        return response.model_dump()
    except Exception as e:
        logger.error(f"Video generation error: {e}")
        raise HTTPException(status_code=500, detail="Video generation failed")
```

### Error Handlers

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    error_details = []
    for err in exc.errors():
        detail = {
            'field': '.'.join(str(loc) for loc in err['loc']),
            'message': err['msg'],
            'type': err['type']
        }
        error_details.append(detail)
    
    api_error = create_error_response(
        error_code="VALIDATION_ERROR",
        error_type="validation_failed",
        message="Request validation failed",
        details={'errors': error_details}
    )
    
    return JSONResponse(
        status_code=422,
        content=api_error.model_dump()
    )
```

## Best Practices

### 1. Model Design

- Use descriptive field names and add comprehensive descriptions
- Set appropriate validation constraints (min/max values, patterns)
- Use enums for categorical data
- Implement custom validators for business logic
- Add computed fields for derived values

### 2. Error Handling

- Always provide meaningful error messages
- Include error codes for programmatic handling
- Add context information when possible
- Use standardized error response format
- Log validation errors for debugging

### 3. Performance

- Enable validation caching for repeated requests
- Monitor validation performance
- Set appropriate timeouts
- Use async validation when possible
- Optimize validation order (fast checks first)

### 4. Security

- Sanitize input data
- Validate file uploads
- Check for malicious content
- Implement rate limiting
- Use HTTPS for all communications

### 5. Documentation

- Add comprehensive field descriptions
- Include usage examples
- Document error codes and messages
- Provide migration guides
- Keep documentation up to date

## Configuration Options

### ValidationConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_request_validation` | bool | True | Enable request validation |
| `enable_response_validation` | bool | True | Enable response validation |
| `enable_performance_monitoring` | bool | True | Enable performance monitoring |
| `enable_validation_caching` | bool | True | Enable validation result caching |
| `max_validation_time` | float | 1.0 | Maximum validation time in seconds |
| `detailed_error_messages` | bool | True | Include detailed error information |
| `log_validation_errors` | bool | True | Log validation errors |

### Environment Variables

```bash
# Validation configuration
VALIDATION_ENABLE_REQUEST=true
VALIDATION_ENABLE_RESPONSE=true
VALIDATION_ENABLE_MONITORING=true
VALIDATION_ENABLE_CACHING=true
VALIDATION_MAX_TIME=1.0
VALIDATION_DETAILED_ERRORS=true
VALIDATION_LOG_ERRORS=true
```

## Testing

### Unit Tests

```python
import pytest
from .pydantic_schemas import VideoGenerationInput, ValidationError

def test_valid_video_generation_input():
    """Test valid video generation input."""
    input_data = {
        "prompt": "A beautiful sunset",
        "quality": "high",
        "width": 512,
        "height": 512
    }
    
    validated = VideoGenerationInput(**input_data)
    assert validated.prompt == "A beautiful sunset"
    assert validated.quality == "high"

def test_invalid_video_generation_input():
    """Test invalid video generation input."""
    input_data = {
        "prompt": "",  # Empty prompt
        "height": 100,  # Invalid height
        "width": 100    # Invalid width
    }
    
    with pytest.raises(ValidationError):
        VideoGenerationInput(**input_data)
```

### Integration Tests

```python
import pytest
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_video_generation_endpoint():
    """Test video generation endpoint with validation."""
    response = client.post(
        "/api/v1/videos/generate",
        json={
            "prompt": "A beautiful sunset",
            "quality": "high"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "video_id" in data
    assert data["status"] == "processing"

def test_invalid_video_generation_endpoint():
    """Test video generation endpoint with invalid input."""
    response = client.post(
        "/api/v1/videos/generate",
        json={
            "prompt": "",  # Invalid empty prompt
            "height": 100  # Invalid height
        }
    )
    
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "VALIDATION_ERROR"
```

## Performance Considerations

### Validation Caching

- Cache validation results for repeated requests
- Use request content hash as cache key
- Set appropriate cache TTL
- Monitor cache hit rates

### Async Validation

- Use async validators for I/O operations
- Implement timeout handling
- Avoid blocking operations in validators
- Use connection pooling for database validations

### Memory Management

- Limit validation cache size
- Use weak references for large objects
- Implement cache eviction policies
- Monitor memory usage

## Monitoring and Observability

### Metrics

- Validation success/failure rates
- Average validation time
- Cache hit/miss ratios
- Error rates by type
- Request volume

### Logging

- Log validation errors with context
- Include request IDs for tracing
- Log performance metrics
- Monitor validation timeouts

### Alerts

- High validation error rates
- Slow validation times
- Cache miss rates
- Memory usage thresholds

## Migration Guide

### From Pydantic v1 to v2

1. Update import statements
2. Replace `Config` class with `model_config`
3. Update validator decorators
4. Use new field validation methods
5. Update error handling

### From Manual Validation

1. Define Pydantic models for all inputs/outputs
2. Add validation decorators to endpoints
3. Update error handling to use standardized format
4. Implement performance monitoring
5. Add comprehensive tests

## Troubleshooting

### Common Issues

1. **Validation Timeouts**: Increase `max_validation_time` or optimize validators
2. **Memory Leaks**: Implement cache eviction policies
3. **Performance Issues**: Enable caching and monitor metrics
4. **Error Handling**: Ensure proper exception handlers are registered

### Debug Mode

```python
# Enable debug mode for detailed validation information
config = ValidationConfig(
    detailed_error_messages=True,
    log_validation_errors=True
)

# Add debug logging
import logging
logging.getLogger("pydantic").setLevel(logging.DEBUG)
```

## Conclusion

The Pydantic validation system provides a robust, performant, and maintainable solution for input/output validation in the AI Video system. By following the patterns and best practices outlined in this document, you can ensure consistent, reliable, and secure API behavior.

For additional information, refer to:
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [FastAPI Validation](https://fastapi.tiangolo.com/tutorial/body/)
- [API Design Best Practices](https://restfulapi.net/) 