# 🚀 HTTP EXCEPTION SYSTEM GUIDE

## Overview

This guide covers the comprehensive HTTP exception system for AI Video applications, providing:
- **Specific HTTP status codes** for different error types
- **Detailed error messages** with context and categorization
- **FastAPI integration** with proper error responses
- **Error monitoring** and analytics
- **Best practices** for error handling

## Table of Contents

1. [Error Categories](#error-categories)
2. [Exception Types](#exception-types)
3. [Error Context](#error-context)
4. [FastAPI Integration](#fastapi-integration)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Error Monitoring](#error-monitoring)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Error Categories

### ErrorCategory Enum

```python
class ErrorCategory(Enum):
    VALIDATION = "validation"                    # 400, 422
    AUTHENTICATION = "authentication"            # 401
    AUTHORIZATION = "authorization"              # 403
    RESOURCE_NOT_FOUND = "resource_not_found"    # 404
    RESOURCE_CONFLICT = "resource_conflict"      # 409
    PROCESSING_ERROR = "processing_error"        # 422
    MODEL_ERROR = "model_error"                  # 500
    DATABASE_ERROR = "database_error"            # 500
    CACHE_ERROR = "cache_error"                  # 500
    EXTERNAL_SERVICE_ERROR = "external_service_error"  # 502
    RATE_LIMIT_ERROR = "rate_limit_error"        # 429
    SYSTEM_ERROR = "system_error"                # 500
    TIMEOUT_ERROR = "timeout_error"              # 408
    MEMORY_ERROR = "memory_error"                # 500
```

### ErrorSeverity Enum

```python
class ErrorSeverity(Enum):
    LOW = "low"           # Info level logging
    MEDIUM = "medium"     # Warning level logging
    HIGH = "high"         # Error level logging
    CRITICAL = "critical" # Critical level logging
```

## Exception Types

### 1. Validation Errors (400, 422)

```python
# Basic validation error
raise ValidationError(
    detail="Field is required",
    field="video_id",
    value=None
)

# Video-specific validation
raise InvalidVideoRequestError(
    detail="Video ID is required",
    video_id=""
)

# Model-specific validation
raise InvalidModelRequestError(
    detail="Model name is required",
    model_name=""
)
```

### 2. Authentication Errors (401)

```python
# General authentication error
raise AuthenticationError("Authentication required")

# Invalid token
raise InvalidTokenError()
```

### 3. Authorization Errors (403)

```python
# General authorization error
raise AuthorizationError("Access denied")

# Insufficient permissions
raise InsufficientPermissionsError(
    operation="load_model",
    required_permissions=["model:load"]
)
```

### 4. Resource Not Found Errors (404)

```python
# General resource not found
raise ResourceNotFoundError("Video", "video_123")

# Video not found
raise VideoNotFoundError("video_123")

# Model not found
raise ModelNotFoundError("stable-diffusion")
```

### 5. Resource Conflict Errors (409)

```python
# General resource conflict
raise ResourceConflictError(
    detail="Resource already exists",
    resource_type="Video",
    resource_id="video_123"
)

# Video already exists
raise VideoAlreadyExistsError("video_123")
```

### 6. Processing Errors (422)

```python
# General processing error
raise ProcessingError(
    detail="Video processing failed",
    video_id="video_123"
)

# Video generation error
raise VideoGenerationError(
    detail="Generation failed due to inappropriate content",
    video_id="video_123",
    model_name="stable-diffusion"
)

# Processing timeout
raise VideoProcessingTimeoutError(
    video_id="video_123",
    timeout_seconds=30
)
```

### 7. Model Errors (500)

```python
# General model error
raise ModelError(
    detail="Model inference failed",
    model_name="stable-diffusion"
)

# Model loading error
raise ModelLoadError(
    model_name="stable-diffusion",
    detail="Model file is corrupted"
)

# Model inference error
raise ModelInferenceError(
    model_name="stable-diffusion",
    detail="Inference failed due to memory constraints"
)
```

### 8. Database Errors (500)

```python
# General database error
raise DatabaseError(
    detail="Database operation failed",
    operation="query"
)

# Connection error
raise DatabaseConnectionError("Database connection lost")

# Query error
raise DatabaseQueryError(
    detail="Failed to execute query",
    query="SELECT * FROM videos WHERE id = 'video_123'"
)
```

### 9. Cache Errors (500)

```python
# General cache error
raise CacheError(
    detail="Cache operation failed",
    operation="get"
)

# Connection error
raise CacheConnectionError("Cache connection lost")
```

### 10. External Service Errors (502)

```python
# External service error
raise ExternalServiceError(
    service_name="video_storage",
    detail="Service unavailable"
)
```

### 11. Rate Limit Errors (429)

```python
# Rate limit exceeded
raise RateLimitError(
    limit=100,
    window_seconds=3600,
    retry_after=60
)
```

### 12. System Errors (500)

```python
# General system error
raise SystemError("An unexpected error occurred")

# Memory error
raise MemoryError(
    detail="Insufficient memory",
    available_memory=1024,
    required_memory=2048
)
```

### 13. Timeout Errors (408)

```python
# Request timeout
raise TimeoutError(
    detail="Request timed out",
    timeout_seconds=30
)
```

## Error Context

### ErrorContext Class

```python
@dataclass
class ErrorContext:
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    video_id: Optional[str] = None
    model_name: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)
```

### Usage Examples

```python
# Create context with video information
context = ErrorContext(
    user_id="user_123",
    video_id="video_456",
    model_name="stable-diffusion",
    operation="video_generation"
)

# Use context in exception
raise VideoGenerationError(
    detail="Generation failed",
    video_id="video_456",
    context=context
)
```

## FastAPI Integration

### Setup Error Handlers

```python
from fastapi import FastAPI
from http_exceptions import setup_error_handlers

app = FastAPI()

# Setup error handlers
setup_error_handlers(app)
```

### Error Response Format

All errors return a consistent JSON format:

```json
{
  "error": {
    "type": "VideoGenerationError",
    "message": "Video generation failed due to inappropriate content",
    "category": "processing_error",
    "severity": "high",
    "status_code": 422,
    "timestamp": 1640995200.0,
    "context": {
      "user_id": "user_123",
      "request_id": "req_456",
      "video_id": "video_789",
      "model_name": "stable-diffusion",
      "operation": "video_generation",
      "additional_data": {}
    }
  }
}
```

### Custom Error Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from http_exceptions import HTTPExceptionHandler

error_handler = HTTPExceptionHandler()

@app.exception_handler(AIVideoHTTPException)
async def ai_video_exception_handler(request: Request, exc: AIVideoHTTPException):
    return error_handler.handle_exception(exc, request)
```

## Error Handling Patterns

### 1. Context Manager Pattern

```python
from http_exceptions import error_context

async def process_video(video_id: str, user_id: str):
    with error_context("video_processing", user_id=user_id, video_id=video_id):
        # Your code here
        result = await generate_video(video_id)
        return result
```

### 2. Decorator Pattern

```python
from http_exceptions import handle_errors

@handle_errors
async def risky_operation(video_id: str):
    # This function will automatically handle errors
    result = await some_risky_function(video_id)
    return result
```

### 3. Try-Catch Pattern

```python
async def process_with_error_handling(video_id: str):
    try:
        result = await process_video(video_id)
        return result
    except VideoNotFoundError:
        # Handle specific error
        logger.warning(f"Video {video_id} not found")
        raise
    except AIVideoHTTPException:
        # Re-raise AI Video exceptions
        raise
    except Exception as exc:
        # Convert to system error
        raise SystemError(f"Unexpected error: {str(exc)}")
```

### 4. Validation Pattern

```python
async def validate_video_request(request: VideoRequest):
    # Validate required fields
    if not request.video_id:
        raise InvalidVideoRequestError(
            "Video ID is required",
            video_id=request.video_id
        )
    
    # Validate dimensions
    if not (64 <= request.width <= 4096):
        raise ValidationError(
            f"Width must be between 64 and 4096, got {request.width}",
            field="width",
            value=request.width
        )
    
    # Validate prompt length
    if len(request.prompt) > 1000:
        raise ValidationError(
            f"Prompt too long: {len(request.prompt)} characters (max 1000)",
            field="prompt",
            value=request.prompt[:100] + "..."
        )
```

## Error Monitoring

### ErrorMonitor Class

```python
from http_exceptions import ErrorMonitor

# Initialize monitor
error_monitor = ErrorMonitor()

# Record error
error_monitor.record_error(exc)

# Get statistics
stats = error_monitor.get_error_stats()
print(stats)
# {
#   "error_counts": {"processing_error:VideoGenerationError": 5},
#   "total_errors": 10,
#   "recent_errors": [...]
# }

# Get error rate
rate = error_monitor.get_error_rate(window_minutes=5)
print(f"Error rate: {rate} errors/minute")
```

### Error Statistics Endpoint

```python
@app.get("/errors/stats")
async def get_error_stats():
    """Get error statistics endpoint."""
    return error_monitor.get_error_stats()
```

## Best Practices

### 1. Use Specific Exception Types

```python
# ✅ GOOD: Use specific exception types
if not video_id:
    raise InvalidVideoRequestError("Video ID is required", video_id=video_id)

if not await video_exists(video_id):
    raise VideoNotFoundError(video_id)

# ❌ BAD: Use generic exceptions
if not video_id:
    raise HTTPException(status_code=400, detail="Bad request")
```

### 2. Provide Context Information

```python
# ✅ GOOD: Include context
context = ErrorContext(
    user_id=user_id,
    video_id=video_id,
    operation="video_generation"
)

raise VideoGenerationError(
    detail="Generation failed",
    video_id=video_id,
    context=context
)

# ❌ BAD: No context
raise VideoGenerationError("Generation failed")
```

### 3. Use Appropriate Status Codes

```python
# Validation errors: 400, 422
raise ValidationError("Invalid input")

# Authentication: 401
raise AuthenticationError("Authentication required")

# Authorization: 403
raise AuthorizationError("Access denied")

# Not found: 404
raise VideoNotFoundError(video_id)

# Conflict: 409
raise VideoAlreadyExistsError(video_id)

# Processing: 422
raise VideoGenerationError("Generation failed")

# Rate limit: 429
raise RateLimitError(limit=100, window_seconds=3600)

# Server errors: 500, 502
raise SystemError("Internal server error")
```

### 4. Handle Errors Gracefully

```python
async def robust_operation(video_id: str):
    try:
        # Try primary operation
        result = await primary_operation(video_id)
        return result
    except VideoNotFoundError:
        # Try fallback
        result = await fallback_operation(video_id)
        return result
    except AIVideoHTTPException:
        # Re-raise AI Video exceptions
        raise
    except Exception as exc:
        # Convert unexpected errors
        raise SystemError(f"Unexpected error: {str(exc)}")
```

### 5. Log Errors Appropriately

```python
# Log based on severity
if exc.severity == ErrorSeverity.CRITICAL:
    logger.critical(f"Critical error: {exc.detail}")
elif exc.severity == ErrorSeverity.HIGH:
    logger.error(f"High severity error: {exc.detail}")
elif exc.severity == ErrorSeverity.MEDIUM:
    logger.warning(f"Medium severity error: {exc.detail}")
else:
    logger.info(f"Low severity error: {exc.detail}")
```

### 6. Monitor Error Rates

```python
# Monitor error rates
error_rate = error_monitor.get_error_rate(window_minutes=5)

if error_rate > 10:  # More than 10 errors per minute
    logger.critical(f"High error rate detected: {error_rate} errors/minute")
    # Trigger alert or scaling
```

## Examples

### Complete API Example

```python
from fastapi import FastAPI, Request
from http_exceptions import *

app = FastAPI()

# Setup error handlers
setup_error_handlers(app)

@app.post("/videos/generate")
async def generate_video(request: VideoRequest):
    """Generate video with comprehensive error handling."""
    
    # Validate request
    if not request.video_id:
        raise InvalidVideoRequestError("Video ID is required", request.video_id)
    
    if not request.prompt:
        raise InvalidVideoRequestError("Prompt is required", request.video_id)
    
    # Check rate limit
    if await is_rate_limited(request.user_id):
        raise RateLimitError(limit=100, window_seconds=3600, retry_after=60)
    
    # Check if video exists
    if await video_exists(request.video_id):
        raise VideoAlreadyExistsError(request.video_id)
    
    # Load model
    try:
        model = await load_model(request.model_name)
    except Exception as e:
        raise ModelLoadError(request.model_name, str(e))
    
    # Generate video
    try:
        result = await generate_video_internal(model, request)
        return result
    except Exception as e:
        raise VideoGenerationError(str(e), request.video_id, request.model_name)

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Get video with error handling."""
    
    video = await get_video_from_db(video_id)
    if not video:
        raise VideoNotFoundError(video_id)
    
    return video

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get model info with error handling."""
    
    if not await model_exists(model_name):
        raise ModelNotFoundError(model_name)
    
    return await get_model_info_internal(model_name)
```

### Error Handling in Background Tasks

```python
@app.post("/videos/{video_id}/process")
async def process_video_background(video_id: str, background_tasks: BackgroundTasks):
    """Process video in background with error handling."""
    
    # Validate video exists
    if not await video_exists(video_id):
        raise VideoNotFoundError(video_id)
    
    # Add background task with error handling
    background_tasks.add_task(process_video_with_errors, video_id)
    
    return {"message": "Video processing started", "video_id": video_id}

async def process_video_with_errors(video_id: str):
    """Background task with error handling."""
    try:
        await process_video(video_id)
    except AIVideoHTTPException as exc:
        # Log error but don't raise (background task)
        logger.error(f"Background processing failed: {exc.detail}")
    except Exception as exc:
        # Convert to system error
        logger.error(f"Unexpected error in background task: {exc}")
```

### Rate Limiting Example

```python
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    async def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 3600):
        """Check rate limit for user."""
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            k: v for k, v in self.requests.items()
            if current_time - v["timestamp"] < window
        }
        
        # Check user's requests
        user_requests = self.requests.get(user_id, {"count": 0, "timestamp": current_time})
        
        if current_time - user_requests["timestamp"] >= window:
            user_requests = {"count": 0, "timestamp": current_time}
        
        if user_requests["count"] >= limit:
            retry_after = int(window - (current_time - user_requests["timestamp"]))
            raise RateLimitError(
                limit=limit,
                window_seconds=window,
                retry_after=retry_after
            )
        
        # Update count
        user_requests["count"] += 1
        self.requests[user_id] = user_requests

# Usage
rate_limiter = RateLimiter()

@app.post("/api/endpoint")
async def protected_endpoint(user_id: str):
    await rate_limiter.check_rate_limit(user_id)
    # Process request...
```

## Integration with Monitoring

### Error Metrics

```python
# Track error metrics
error_metrics = {
    "total_errors": 0,
    "errors_by_category": {},
    "errors_by_severity": {},
    "error_rate": 0.0
}

def update_error_metrics(exc: AIVideoHTTPException):
    """Update error metrics."""
    error_metrics["total_errors"] += 1
    
    # Update category metrics
    category = exc.category.value
    if category not in error_metrics["errors_by_category"]:
        error_metrics["errors_by_category"][category] = 0
    error_metrics["errors_by_category"][category] += 1
    
    # Update severity metrics
    severity = exc.severity.value
    if severity not in error_metrics["errors_by_severity"]:
        error_metrics["errors_by_severity"][severity] = 0
    error_metrics["errors_by_severity"][severity] += 1
```

### Alerting

```python
def check_error_alerts():
    """Check for error conditions that require alerting."""
    
    # High error rate
    if error_monitor.get_error_rate(window_minutes=5) > 10:
        send_alert("High error rate detected")
    
    # Critical errors
    critical_errors = error_monitor.get_error_stats().get("errors_by_severity", {}).get("critical", 0)
    if critical_errors > 5:
        send_alert("Multiple critical errors detected")
    
    # Specific error patterns
    if "database_error" in error_monitor.get_error_stats().get("error_counts", {}):
        send_alert("Database errors detected")
```

This comprehensive HTTP exception system provides a robust foundation for handling errors in AI Video applications with proper HTTP status codes, detailed error messages, and comprehensive monitoring capabilities. 