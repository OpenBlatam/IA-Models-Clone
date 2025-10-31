# HTTP Exception Guide

A comprehensive guide for using HTTP exceptions in the HeyGen AI FastAPI application with proper error modeling and structured responses.

## 🎯 Overview

This guide covers:
- **HTTP Exception Types**: Specific exception classes for different error scenarios
- **Error Response Format**: Standardized error response structure
- **Exception Handling**: Proper exception handling and logging
- **Best Practices**: Guidelines for effective error management
- **Integration Patterns**: How to integrate exceptions into your API

## 📋 Table of Contents

1. [Exception Types](#exception-types)
2. [Error Response Format](#error-response-format)
3. [Exception Handling](#exception-handling)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Integration Guide](#integration-guide)
7. [Error Monitoring](#error-monitoring)

## 🚨 Exception Types

### Validation Errors (400, 422)

```python
from api.exceptions.http_exceptions import (
    ValidationError, InvalidInputError, MissingRequiredFieldError,
    InvalidVideoFormatError, InvalidScriptError
)

# General validation error
raise ValidationError(
    message="Request validation failed",
    details=[
        {
            "field": "email",
            "message": "Invalid email format",
            "value": "invalid-email",
            "suggestion": "Please provide a valid email address"
        }
    ]
)

# Specific validation errors
raise InvalidVideoFormatError(
    message="Unsupported video format",
    details=[
        {
            "field": "video_file",
            "message": "Format 'wmv' is not supported",
            "value": "video.wmv",
            "suggestion": "Please use mp4, mov, or avi format"
        }
    ]
)

raise InvalidScriptError(
    message="Script validation failed",
    details=[
        {
            "field": "script",
            "message": "Script contains inappropriate content",
            "value": "script content...",
            "suggestion": "Please review and modify the script content"
        }
    ]
)
```

### Authentication Errors (401)

```python
from api.exceptions.http_exceptions import (
    AuthenticationError, InvalidCredentialsError,
    ExpiredTokenError, MissingTokenError
)

# Invalid credentials
raise InvalidCredentialsError(
    message="Invalid username or password"
)

# Expired token
raise ExpiredTokenError(
    message="Authentication token has expired",
    retry_after=300  # 5 minutes
)

# Missing token
raise MissingTokenError(
    message="Authentication token is required"
)
```

### Authorization Errors (403)

```python
from api.exceptions.http_exceptions import (
    AuthorizationError, InsufficientPermissionsError,
    ResourceAccessDeniedError, SubscriptionRequiredError
)

# Insufficient permissions
raise InsufficientPermissionsError(
    message="You don't have permission to create videos"
)

# Subscription required
raise SubscriptionRequiredError(
    message="Premium subscription required for this feature",
    details=[
        {
            "field": "subscription",
            "message": "Premium plan required",
            "suggestion": "Upgrade to premium plan to access this feature"
        }
    ]
)
```

### Not Found Errors (404)

```python
from api.exceptions.http_exceptions import (
    NotFoundError, UserNotFoundError, VideoNotFoundError,
    TemplateNotFoundError, ProjectNotFoundError
)

# User not found
raise UserNotFoundError(
    message=f"User with ID {user_id} not found"
)

# Video not found
raise VideoNotFoundError(
    message=f"Video with ID {video_id} not found"
)

# Template not found
raise TemplateNotFoundError(
    message=f"Template '{template_name}' not found"
)
```

### Rate Limit Errors (429)

```python
from api.exceptions.http_exceptions import (
    RateLimitError, APIRateLimitError, VideoCreationRateLimitError
)

# General rate limit
raise RateLimitError(
    message="Too many requests",
    retry_after=60  # 1 minute
)

# API rate limit
raise APIRateLimitError(
    message="API rate limit exceeded",
    retry_after=300  # 5 minutes
)

# Video creation rate limit
raise VideoCreationRateLimitError(
    message="Video creation limit reached",
    retry_after=3600  # 1 hour
)
```

### External Service Errors (502, 503, 504)

```python
from api.exceptions.http_exceptions import (
    ExternalServiceError, HeyGenAPIError, VideoProcessingError,
    ExternalAPITimeoutError
)

# HeyGen API error
raise HeyGenAPIError(
    message="HeyGen AI service is temporarily unavailable"
)

# Video processing error
raise VideoProcessingError(
    message="Video processing service is down"
)

# External API timeout
raise ExternalAPITimeoutError(
    message="External service request timed out"
)
```

### Resource Conflict Errors (409)

```python
from api.exceptions.http_exceptions import (
    ResourceConflictError, DuplicateResourceError,
    VideoAlreadyProcessingError
)

# Duplicate resource
raise DuplicateResourceError(
    message="Video with this name already exists"
)

# Video already processing
raise VideoAlreadyProcessingError(
    message="Video is already being processed"
)
```

### Payload Too Large Errors (413)

```python
from api.exceptions.http_exceptions import (
    PayloadTooLargeError, VideoFileTooLargeError, ScriptTooLongError
)

# Video file too large
raise VideoFileTooLargeError(
    message="Video file size exceeds 100MB limit"
)

# Script too long
raise ScriptTooLongError(
    message="Script length exceeds 5000 character limit"
)
```

### Business Logic Errors (422)

```python
from api.exceptions.http_exceptions import (
    BusinessLogicError, VideoDurationLimitError,
    InvalidVideoTemplateError, ScriptContentError
)

# Video duration limit
raise VideoDurationLimitError(
    message="Video duration exceeds 10-minute limit"
)

# Invalid template
raise InvalidVideoTemplateError(
    message="Selected template is not available for this account type"
)

# Script content error
raise ScriptContentError(
    message="Script contains prohibited content"
)
```

## 📊 Error Response Format

All HTTP exceptions return a standardized error response format:

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "category": "validation",
  "severity": "medium",
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "details": [
    {
      "field": "email",
      "message": "Invalid email format",
      "value": "invalid-email",
      "suggestion": "Please provide a valid email address"
    }
  ],
  "documentation_url": "https://docs.example.com/validation",
  "retry_after": null
}
```

### Response Fields

- **error_code**: Unique identifier for the error type
- **message**: Human-readable error message
- **category**: Error category (validation, authentication, etc.)
- **severity**: Error severity level (low, medium, high, critical)
- **timestamp**: When the error occurred
- **request_id**: Request identifier for tracking
- **details**: Detailed error information (optional)
- **documentation_url**: Link to relevant documentation (optional)
- **retry_after**: Seconds to wait before retry (optional)

## 🔧 Exception Handling

### Register Exception Handlers

```python
from fastapi import FastAPI
from api.exceptions.exception_handlers import (
    register_exception_handlers, RequestIDMiddleware
)

app = FastAPI()

# Register exception handlers
register_exception_handlers(app)

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)
```

### Custom Exception Handler

```python
from fastapi import Request
from api.exceptions.http_exceptions import BaseHTTPException

@app.exception_handler(BaseHTTPException)
async def custom_http_exception_handler(
    request: Request,
    exc: BaseHTTPException
):
    """Custom exception handler."""
    # Log the exception
    logger.error(
        "Custom HTTP exception",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code
    )
    
    # Return standardized error response
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_error_response().dict()
    )
```

## 💡 Usage Examples

### Route Handler Examples

```python
from fastapi import APIRouter, Depends
from api.exceptions.http_exceptions import (
    UserNotFoundError, ValidationError, RateLimitError,
    VideoNotFoundError, InsufficientPermissionsError
)

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(user_id: int, current_user: User = Depends(get_current_user)):
    """Get user by ID."""
    try:
        user = await user_service.get_user(user_id)
        if not user:
            raise UserNotFoundError(
                message=f"User with ID {user_id} not found"
            )
        return user
    except DatabaseError as e:
        logger.error("Database error", error=str(e))
        raise InternalServerError("Failed to retrieve user")

@router.post("/videos")
async def create_video(
    video_data: VideoCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new video."""
    # Check permissions
    if not current_user.can_create_videos():
        raise InsufficientPermissionsError(
            message="You don't have permission to create videos"
        )
    
    # Validate video data
    if not video_data.script or len(video_data.script.strip()) == 0:
        raise ValidationError(
            message="Script is required",
            details=[
                {
                    "field": "script",
                    "message": "Script cannot be empty",
                    "suggestion": "Please provide a script for the video"
                }
            ]
        )
    
    # Check rate limits
    if await rate_limiter.is_limited(current_user.id, "video_creation"):
        raise VideoCreationRateLimitError(
            message="Video creation rate limit exceeded",
            retry_after=3600
        )
    
    # Create video
    try:
        video = await video_service.create_video(video_data, current_user)
        return video
    except ExternalServiceError as e:
        raise VideoProcessingError(
            message="Video processing service is unavailable"
        )

@router.get("/videos/{video_id}")
async def get_video(video_id: str, current_user: User = Depends(get_current_user)):
    """Get video by ID."""
    video = await video_service.get_video(video_id)
    if not video:
        raise VideoNotFoundError(
            message=f"Video with ID {video_id} not found"
        )
    
    # Check access permissions
    if not video.can_access(current_user):
        raise ResourceAccessDeniedError(
            message="You don't have access to this video"
        )
    
    return video
```

### Service Layer Examples

```python
class UserService:
    """User service with proper exception handling."""
    
    async def get_user(self, user_id: int) -> User:
        """Get user by ID."""
        try:
            user = await self.db.get_user(user_id)
            if not user:
                raise UserNotFoundError(
                    message=f"User with ID {user_id} not found"
                )
            return user
        except DatabaseConnectionError:
            raise DatabaseError("Database connection failed")
    
    async def create_user(self, user_data: UserCreateRequest) -> User:
        """Create a new user."""
        # Validate email uniqueness
        existing_user = await self.db.get_user_by_email(user_data.email)
        if existing_user:
            raise DuplicateResourceError(
                message="User with this email already exists"
            )
        
        # Validate password strength
        if not self.is_password_strong(user_data.password):
            raise ValidationError(
                message="Password does not meet requirements",
                details=[
                    {
                        "field": "password",
                        "message": "Password must be at least 8 characters",
                        "suggestion": "Use a stronger password"
                    }
                ]
            )
        
        try:
            user = await self.db.create_user(user_data)
            return user
        except DatabaseError:
            raise DatabaseError("Failed to create user")

class VideoService:
    """Video service with proper exception handling."""
    
    async def create_video(self, video_data: VideoCreateRequest, user: User) -> Video:
        """Create a new video."""
        # Validate video format
        if not self.is_supported_format(video_data.format):
            raise InvalidVideoFormatError(
                message=f"Format '{video_data.format}' is not supported"
            )
        
        # Check video duration limit
        if video_data.duration > self.max_duration:
            raise VideoDurationLimitError(
                message=f"Video duration exceeds {self.max_duration} seconds"
            )
        
        # Create video via external API
        try:
            response = await self.heygen_api.create_video(video_data)
            return Video.from_api_response(response)
        except APITimeoutError:
            raise ExternalAPITimeoutError(
                message="Video creation request timed out"
            )
        except APIError as e:
            raise HeyGenAPIError(
                message=f"Video creation failed: {e.message}"
            )
    
    async def get_video_status(self, video_id: str) -> VideoStatus:
        """Get video processing status."""
        try:
            status = await self.heygen_api.get_video_status(video_id)
            return VideoStatus.from_api_response(status)
        except APINotFoundError:
            raise VideoNotFoundError(
                message=f"Video with ID {video_id} not found"
            )
        except APIError as e:
            raise HeyGenAPIError(
                message=f"Failed to get video status: {e.message}"
            )
```

### Exception Factory Usage

```python
from api.exceptions.http_exceptions import ExceptionFactory

# Create validation error with factory
validation_error = ExceptionFactory.create_validation_error(
    field="email",
    message="Invalid email format",
    value="invalid-email",
    suggestion="Please provide a valid email address"
)

# Create rate limit error with factory
rate_limit_error = ExceptionFactory.create_rate_limit_error(
    retry_after=60,
    message="Too many requests"
)

# Create external service error with factory
external_error = ExceptionFactory.create_external_service_error(
    service_name="HeyGen API",
    error_message="Service temporarily unavailable",
    status_code=503
)
```

## 🏆 Best Practices

### 1. Use Specific Exception Types

```python
# ❌ Bad: Generic exception
raise HTTPException(status_code=400, detail="Bad request")

# ✅ Good: Specific exception
raise ValidationError(
    message="Invalid email format",
    details=[{"field": "email", "message": "Invalid format"}]
)
```

### 2. Provide Helpful Error Messages

```python
# ❌ Bad: Generic message
raise UserNotFoundError("User not found")

# ✅ Good: Specific message
raise UserNotFoundError(
    message=f"User with ID {user_id} not found",
    details=[
        {
            "field": "user_id",
            "message": "User does not exist",
            "suggestion": "Please check the user ID and try again"
        }
    ]
)
```

### 3. Include Request ID for Tracking

```python
# ✅ Good: Include request ID
raise ValidationError(
    message="Validation failed",
    request_id=request.headers.get("X-Request-ID")
)
```

### 4. Use Appropriate HTTP Status Codes

```python
# Validation errors: 400, 422
raise ValidationError(...)

# Authentication: 401
raise AuthenticationError(...)

# Authorization: 403
raise AuthorizationError(...)

# Not found: 404
raise NotFoundError(...)

# Rate limits: 429
raise RateLimitError(...)

# Server errors: 500, 502, 503, 504
raise InternalServerError(...)
```

### 5. Log Exceptions Properly

```python
from api.exceptions.http_exceptions import log_exception

try:
    # Your code here
    pass
except BaseHTTPException as e:
    log_exception(e, request_id)
    raise
```

### 6. Handle External Service Errors

```python
try:
    response = await external_api.call()
except APITimeoutError:
    raise ExternalAPITimeoutError("External service timed out")
except APIError as e:
    raise ExternalServiceError(f"External service error: {e.message}")
```

### 7. Use Retry-After for Rate Limits

```python
raise RateLimitError(
    message="Rate limit exceeded",
    retry_after=60  # 1 minute
)
```

## 🔗 Integration Guide

### FastAPI Application Setup

```python
from fastapi import FastAPI
from api.exceptions.exception_handlers import (
    register_exception_handlers, RequestIDMiddleware
)

def create_app() -> FastAPI:
    """Create FastAPI application with exception handling."""
    app = FastAPI(
        title="HeyGen AI API",
        description="AI-powered video creation API",
        version="1.0.0"
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Add middleware
    app.add_middleware(RequestIDMiddleware)
    
    # Include routers
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    app.include_router(video_router, prefix="/videos", tags=["Videos"])
    app.include_router(user_router, prefix="/users", tags=["Users"])
    
    return app

app = create_app()
```

### Dependency Injection with Exception Handling

```python
from fastapi import Depends, HTTPException
from api.exceptions.http_exceptions import (
    AuthenticationError, InsufficientPermissionsError
)

async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get current authenticated user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token")
    except JWTError:
        raise AuthenticationError("Invalid token")
    
    user = await user_service.get_user(user_id)
    if not user:
        raise AuthenticationError("User not found")
    
    return user

async def require_premium_subscription(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require premium subscription."""
    if not current_user.has_premium_subscription():
        raise SubscriptionRequiredError(
            message="Premium subscription required"
        )
    return current_user
```

### Error Response Documentation

```python
from fastapi import APIRouter
from api.exceptions.exception_handlers import ERROR_RESPONSE_EXAMPLES

router = APIRouter()

@router.post(
    "/videos",
    responses={
        400: {"description": "Validation Error", "content": ERROR_RESPONSE_EXAMPLES["validation_error"]},
        401: {"description": "Authentication Error", "content": ERROR_RESPONSE_EXAMPLES["authentication_error"]},
        403: {"description": "Authorization Error", "content": ERROR_RESPONSE_EXAMPLES["authorization_error"]},
        429: {"description": "Rate Limit Error", "content": ERROR_RESPONSE_EXAMPLES["rate_limit_error"]},
        500: {"description": "Internal Server Error", "content": ERROR_RESPONSE_EXAMPLES["internal_server_error"]},
    }
)
async def create_video(video_data: VideoCreateRequest):
    """Create a new video."""
    # Implementation here
    pass
```

## 📊 Error Monitoring

### Structured Logging

```python
import structlog
from api.exceptions.http_exceptions import log_exception

logger = structlog.get_logger()

# Log exceptions with context
def log_error_with_context(exception: BaseHTTPException, context: dict):
    """Log exception with additional context."""
    logger.error(
        "HTTP exception occurred",
        error_code=exception.error_code,
        message=exception.message,
        category=exception.category.value,
        severity=exception.severity.value,
        status_code=exception.status_code,
        request_id=exception.request_id,
        context=context,
        exc_info=True
    )
```

### Error Metrics Collection

```python
from prometheus_client import Counter, Histogram

# Error metrics
error_counter = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['error_code', 'category', 'severity']
)

error_duration = Histogram(
    'http_error_duration_seconds',
    'HTTP error duration',
    ['error_code']
)

def track_error_metrics(exception: BaseHTTPException):
    """Track error metrics."""
    error_counter.labels(
        error_code=exception.error_code,
        category=exception.category.value,
        severity=exception.severity.value
    ).inc()
```

### Error Alerting

```python
from api.exceptions.http_exceptions import ErrorSeverity

def should_alert(exception: BaseHTTPException) -> bool:
    """Determine if error should trigger alert."""
    return exception.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]

def send_alert(exception: BaseHTTPException):
    """Send error alert."""
    if should_alert(exception):
        # Send alert to monitoring system
        alert_service.send_alert(
            title=f"High severity error: {exception.error_code}",
            message=exception.message,
            severity=exception.severity.value,
            context={
                "error_code": exception.error_code,
                "category": exception.category.value,
                "request_id": exception.request_id
            }
        )
```

## 📚 Additional Resources

- [FastAPI Exception Handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- [REST API Error Handling](https://restfulapi.net/error-handling/)
- [Structured Logging](https://structlog.readthedocs.io/)

## 🚀 Next Steps

1. **Implement exception handlers** in your FastAPI application
2. **Use specific exception types** for different error scenarios
3. **Add request ID middleware** for error tracking
4. **Set up error monitoring** and alerting
5. **Document error responses** in your API documentation
6. **Test error scenarios** to ensure proper handling
7. **Monitor error rates** and optimize based on metrics

This HTTP exception guide provides a comprehensive framework for handling errors in your HeyGen AI API with proper HTTP status codes, structured responses, and effective error management patterns. 