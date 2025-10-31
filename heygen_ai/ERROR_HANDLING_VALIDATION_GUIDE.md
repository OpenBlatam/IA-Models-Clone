# 🚨 HeyGen AI API - Error Handling & Validation Guide

## Overview

This guide documents the comprehensive error handling and validation system implemented for the HeyGen AI FastAPI backend. The system provides structured error responses, robust validation, and consistent error handling across all endpoints.

## 🏗️ System Architecture

### Core Components

1. **Error Handling System** (`api/core/error_handling.py`)
   - Custom exception hierarchy with specific error types
   - Error factory for creating consistent errors
   - Exception handlers for FastAPI integration
   - Decorator-based error handling

2. **Validation System** (`api/utils/validators.py`)
   - Comprehensive input validation functions
   - Business logic validation
   - Video processing validation
   - User data validation

3. **Middleware Integration** (`api/__init__.py`)
   - Request tracking and timing
   - Automatic error logging
   - Request ID generation
   - Performance monitoring

## 📋 Error Types & Categories

### Error Hierarchy

```python
HeyGenBaseError (Base)
├── ValidationError
├── AuthenticationError
├── AuthorizationError
├── DatabaseError
├── ResourceNotFoundError
├── VideoProcessingError
├── RateLimitError
├── ExternalServiceError
└── SystemError
```

### Error Categories

| Category | Description | HTTP Status |
|----------|-------------|-------------|
| `validation` | Input validation failures | 400 |
| `authentication` | Authentication failures | 401 |
| `authorization` | Permission failures | 403 |
| `database` | Database operation failures | 500 |
| `resource_not_found` | Resource not found | 404 |
| `video_processing` | Video processing errors | 500 |
| `rate_limit` | Rate limiting violations | 429 |
| `external_service` | Third-party service failures | 503 |
| `system` | System-level failures | 500 |

### Error Severity Levels

- **LOW**: Non-critical validation errors
- **MEDIUM**: Business logic errors, rate limits
- **HIGH**: Authentication, authorization, database errors
- **CRITICAL**: System failures, security issues

## 🔧 Usage Examples

### 1. Basic Error Handling

```python
from api.core.error_handling import error_factory, handle_errors, ErrorCategory

@router.post("/generate")
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="generate_video")
async def generate_video(request_data: Dict[str, Any]):
    # Validate input
    if not request_data.get("script"):
        raise error_factory.validation_error(
            message="Script is required",
            field="script",
            context={"operation": "generate_video"}
        )
    
    # Process video generation
    try:
        result = await process_video(request_data)
        return result
    except DatabaseError as e:
        raise error_factory.database_error(
            message="Failed to save video record",
            operation="create_video",
            context={"video_id": video_id}
        )
```

### 2. Validation Functions

```python
from api.utils.validators import (
    validate_video_generation_request,
    validate_script_content,
    validate_voice_id
)

# Validate complete video generation request
validate_video_generation_request(request_data)

# Validate specific components
is_valid, errors = validate_script_content(script)
if not is_valid:
    raise error_factory.validation_error(
        message="Script validation failed",
        field="script",
        validation_errors=errors
    )
```

### 3. Custom Validation

```python
from api.core.error_handling import validate_required, validate_length, validate_enum

# Validate required fields
validate_required("user_id", user_id)

# Validate string length
validate_length("username", username, min_length=3, max_length=30)

# Validate enum values
validate_enum("quality", quality, QualityLevel)
```

## 📊 Error Response Format

### Standard Error Response

```json
{
    "success": false,
    "error": {
        "error_code": "VALIDATION_ERROR",
        "message": "Script validation failed",
        "user_friendly_message": "Please check your script content",
        "category": "validation",
        "severity": "low",
        "timestamp": "2024-01-15T10:30:00Z",
        "details": {
            "field": "script",
            "value": "short script",
            "validation_errors": [
                "Script must be at least 10 characters long"
            ]
        },
        "context": {
            "operation": "generate_video",
            "user_id": "user_123"
        }
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
}
```

### Success Response

```json
{
    "success": true,
    "data": {
        "video_id": "video_123",
        "status": "processing"
    },
    "message": "Video generation started successfully",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
}
```

## 🛡️ Validation Functions

### Video Processing Validation

```python
# Validate video generation request
validate_video_generation_request(data)

# Validate script content
is_valid, errors = validate_script_content(script)

# Validate voice ID
is_valid, errors = validate_voice_id(voice_id)

# Validate language code
is_valid, errors = validate_language_code(language)

# Validate quality settings
is_valid, errors = validate_quality_settings(quality, duration)

# Validate video duration
is_valid, errors = validate_video_duration(duration)
```

### User Data Validation

```python
# Validate user registration
validate_user_registration_data(data)

# Validate email format
is_valid, errors = validate_email_format(email)

# Validate username format
is_valid, errors = validate_username_format(username)

# Validate password strength
is_valid, errors = validate_password_strength(password)

# Validate user ID
is_valid, errors = validate_user_id(user_id)
```

### Business Logic Validation

```python
# Validate pagination parameters
is_valid, errors = validate_pagination_parameters(page, page_size)

# Validate date range
is_valid, errors = validate_date_range(date_from, date_to)

# Validate API key format
is_valid, errors = validate_api_key_format(api_key)

# Validate rate limit parameters
is_valid, errors = validate_rate_limit_parameters(requests_per_minute, burst_limit)

# Validate business constraints
is_valid, errors = validate_business_logic_constraints(user_id, operation, constraints)
```

## 🔄 Error Handling Decorators

### Automatic Error Handling

```python
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="generate_video")
async def generate_video_endpoint():
    # Function implementation
    # Errors are automatically caught and logged
    pass
```

### Custom Error Handling

```python
@handle_errors(category=ErrorCategory.DATABASE, operation="create_user", log_errors=True)
async def create_user_endpoint():
    # Function implementation
    # Errors are logged and re-raised with context
    pass
```

## 📝 Logging & Monitoring

### Request Logging

```python
# Request start log
logger.info(
    f"Request started: {request.method} {request.url.path}",
    extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "client_ip": client_ip,
        "user_agent": user_agent
    }
)

# Request completion log
logger.info(
    f"Request completed: {request.method} {request.url.path} - {status_code}",
    extra={
        "request_id": request_id,
        "processing_time": processing_time,
        "status_code": status_code
    }
)
```

### Error Logging

```python
# Error log with context
logger.error(
    f"HeyGen error in {operation}: {message}",
    extra={
        "error_code": error_code,
        "category": category,
        "severity": severity,
        "details": details,
        "context": context
    }
)
```

## 🚀 Best Practices

### 1. Use Specific Error Types

```python
# Good: Specific error type
raise error_factory.validation_error(
    message="Invalid email format",
    field="email",
    value=email
)

# Avoid: Generic exceptions
raise Exception("Something went wrong")
```

### 2. Provide Rich Context

```python
# Good: Rich context information
raise error_factory.database_error(
    message="Failed to create video record",
    operation="create_video",
    context={
        "video_id": video_id,
        "user_id": user_id,
        "operation": "generate_video"
    }
)
```

### 3. Use Validation Functions

```python
# Good: Use validation functions
validate_video_generation_request(request_data)

# Avoid: Manual validation
if not request_data.get("script"):
    raise ValueError("Script required")
```

### 4. Handle Errors Gracefully

```python
# Good: Graceful error handling
try:
    result = await risky_operation()
    return result
except SpecificError as e:
    # Handle specific error
    raise error_factory.specific_error(message=str(e))
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise error_factory.system_error(message="Internal server error")
```

## 🔧 Configuration

### Error Handler Registration

```python
# Register exception handlers
app.add_exception_handler(HeyGenBaseError, heygen_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(PydanticValidationError, pydantic_validation_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
```

### Middleware Setup

```python
# Request middleware for tracking
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    # Add request ID and timing
    # Log request start/end
    # Handle errors
    pass
```

## 📈 Monitoring & Alerting

### Error Metrics

- **Error Rate**: Percentage of requests that result in errors
- **Error Distribution**: Breakdown by error category and severity
- **Response Time**: Processing time for successful vs failed requests
- **User Impact**: Number of users affected by errors

### Alerting Rules

- **High Error Rate**: >5% error rate for any endpoint
- **Critical Errors**: Any CRITICAL severity errors
- **Database Errors**: >1% database error rate
- **External Service Errors**: >2% external service error rate

## 🧪 Testing

### Error Handling Tests

```python
import pytest
from api.core.error_handling import ValidationError, error_factory

def test_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        raise error_factory.validation_error(
            message="Test validation error",
            field="test_field"
        )
    
    assert exc_info.value.error_code == "VALIDATION_ERROR"
    assert exc_info.value.category.value == "validation"
```

### Validation Tests

```python
def test_script_validation():
    # Test valid script
    is_valid, errors = validate_script_content("This is a valid script with sufficient length")
    assert is_valid
    assert len(errors) == 0
    
    # Test invalid script
    is_valid, errors = validate_script_content("short")
    assert not is_valid
    assert len(errors) > 0
```

## 📚 Integration Examples

### FastAPI Route Integration

```python
@router.post("/videos/generate")
@handle_errors(category=ErrorCategory.VIDEO_PROCESSING, operation="generate_video")
async def generate_video(
    request_data: Dict[str, Any],
    user_id: str = Depends(get_current_user)
):
    # Validate request
    validate_video_generation_request(request_data)
    
    # Process request
    result = await video_service.generate_video(request_data, user_id)
    
    return result
```

### Service Layer Integration

```python
class VideoService:
    @handle_errors(category=ErrorCategory.DATABASE, operation="create_video")
    async def create_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        # Database operation
        video = await self.db.create_video(video_data)
        return video.to_dict()
```

## 🔒 Security Considerations

### Input Sanitization

- All user inputs are validated and sanitized
- Script content is checked for inappropriate content
- File uploads are validated for type and size
- API keys and tokens are validated for format

### Error Information Disclosure

- Internal error details are not exposed to users
- User-friendly messages are provided instead
- Sensitive information is redacted from error responses
- Error logs contain full details for debugging

### Rate Limiting

- API endpoints are rate-limited to prevent abuse
- Rate limit errors include retry-after headers
- Different limits for different user tiers
- Burst protection for high-traffic scenarios

## 📖 Conclusion

The HeyGen AI API error handling and validation system provides:

- **Comprehensive Error Coverage**: All error scenarios are handled consistently
- **Rich Context Information**: Errors include detailed context for debugging
- **User-Friendly Messages**: Clear, actionable error messages for users
- **Robust Validation**: Multi-layer validation for all inputs
- **Performance Monitoring**: Request tracking and timing information
- **Security**: Input sanitization and error information protection

This system ensures reliable, secure, and user-friendly API operations while providing developers with the tools needed for effective debugging and monitoring. 