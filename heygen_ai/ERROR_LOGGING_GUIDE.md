# Error Logging and User-Friendly Error Messages Guide

## Overview

This guide documents the enhanced error logging and user-friendly error message system implemented in the HeyGen AI FastAPI backend. The system provides comprehensive error tracking, structured logging, and user-friendly error responses.

## Key Features

### 1. Structured Error Logging
- **ErrorLogger Class**: Centralized logging with structured data
- **Unique Error IDs**: Each error gets a unique identifier for tracking
- **Context Preservation**: Request context, user information, and operation details
- **Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL for proper error categorization

### 2. User-Friendly Error Messages
- **UserFriendlyMessageGenerator**: Generates appropriate user-facing messages
- **Context-Aware Messages**: Different messages based on error type and context
- **Localization Ready**: Structured for easy translation and customization
- **Actionable Messages**: Provide clear guidance on what users should do

### 3. Comprehensive Error Categories
- **Validation Errors**: Input validation and data format issues
- **Authentication/Authorization**: User authentication and permission issues
- **Resource Errors**: Not found, exhaustion, and concurrency issues
- **System Errors**: Database, external service, and infrastructure issues
- **Video Processing**: Specific to video generation and processing

## Error Logging System

### ErrorLogger Class

```python
class ErrorLogger:
    @staticmethod
    def log_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """Log error with structured data and context"""
```

#### Usage Examples

```python
# Basic error logging
ErrorLogger.log_error(
    error=validation_error,
    context={'field': 'email', 'value': 'invalid-email'},
    user_id='user123',
    request_id='req456',
    operation='user_registration'
)

# With additional context
ErrorLogger.log_error(
    error=database_error,
    context={
        'operation': 'user_create',
        'database': 'postgresql',
        'table': 'users',
        'query': 'INSERT INTO users...'
    },
    user_id='user123',
    request_id='req456'
)
```

### User Action Logging

```python
# Log successful operations
ErrorLogger.log_user_action(
    action='video_generation_started',
    user_id='user123',
    request_id='req456',
    details={'video_id': 'vid789', 'template': 'business_presentation'}
)

# Log failed operations
ErrorLogger.log_user_action(
    action='video_generation_failed',
    user_id='user123',
    request_id='req456',
    success=False,
    details={'error': 'Template not found', 'video_id': 'vid789'}
)
```

## User-Friendly Error Messages

### UserFriendlyMessageGenerator Class

```python
class UserFriendlyMessageGenerator:
    @staticmethod
    def get_validation_message(field: Optional[str] = None, error_type: str = "validation") -> str:
        """Generate user-friendly validation error messages"""
    
    @staticmethod
    def get_authentication_message() -> str:
        """Generate user-friendly authentication error messages"""
    
    @staticmethod
    def get_authorization_message() -> str:
        """Generate user-friendly authorization error messages"""
    
    @staticmethod
    def get_resource_not_found_message(resource_type: Optional[str] = None) -> str:
        """Generate user-friendly resource not found messages"""
    
    @staticmethod
    def get_rate_limit_message(retry_after: Optional[int] = None) -> str:
        """Generate user-friendly rate limit messages"""
    
    @staticmethod
    def get_timeout_message() -> str:
        """Generate user-friendly timeout messages"""
    
    @staticmethod
    def get_video_processing_message(stage: Optional[str] = None) -> str:
        """Generate user-friendly video processing messages"""
    
    @staticmethod
    def get_system_error_message() -> str:
        """Generate user-friendly system error messages"""
    
    @staticmethod
    def get_database_error_message() -> str:
        """Generate user-friendly database error messages"""
```

### Message Examples

| Error Type | Technical Message | User-Friendly Message |
|------------|------------------|----------------------|
| Validation | "Field 'email' is required" | "Please check the 'email' field and try again." |
| Authentication | "Invalid API key" | "Please log in again to continue." |
| Authorization | "Insufficient permissions" | "You don't have permission to perform this action." |
| Resource Not Found | "Video with ID '123' not found" | "The video you're looking for doesn't exist." |
| Rate Limit | "Rate limit exceeded" | "Too many requests. Please try again in 60 seconds." |
| Timeout | "Request timeout after 30s" | "The request took too long to process. Please try again." |
| Video Processing | "Template processing failed" | "Video processing failed. Please try again." |
| System Error | "Database connection failed" | "We're experiencing technical difficulties. Please try again later." |

## Error Factory with Logging

### ErrorFactory Class

The `ErrorFactory` class automatically logs errors when creating them:

```python
class ErrorFactory:
    @staticmethod
    def validation_error(
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ) -> ValidationError:
        """Create validation error with logging"""
        error = ValidationError(
            message=message,
            field=field,
            value=value,
            validation_errors=validation_errors,
            **kwargs
        )
        ErrorLogger.log_error(error, context=kwargs.get('context'))
        return error
```

### Usage Examples

```python
# Create and log validation error
validation_error = error_factory.validation_error(
    message="Invalid email format",
    field="email",
    value="invalid-email",
    context={'operation': 'user_registration'}
)

# Create and log authentication error
auth_error = error_factory.authentication_error(
    message="API key expired",
    context={'api_key_id': 'key123', 'expired_at': '2024-01-01T00:00:00Z'}
)

# Create and log video processing error
video_error = error_factory.video_processing_error(
    message="Template processing failed",
    video_id="vid123",
    processing_stage="template_rendering",
    context={'template_id': 'tpl456', 'error_code': 'RENDER_FAILED'}
)
```

## Exception Handlers

### HeyGen Exception Handler

```python
async def heygen_exception_handler(request: Request, exc: HeyGenBaseError) -> JSONResponse:
    """Handle HeyGen AI exceptions with proper logging and user-friendly responses"""
```

#### Features:
- **Automatic Logging**: Logs error with request context
- **User-Friendly Response**: Returns appropriate user message
- **HTTP Status Mapping**: Maps error categories to HTTP status codes
- **Response Headers**: Includes error ID and retry information
- **Structured Response**: Consistent error response format

#### Response Format:

```json
{
  "error": {
    "id": "abc12345",
    "code": "VALIDATION_ERROR",
    "message": "Please check the 'email' field and try again.",
    "category": "validation",
    "severity": "low",
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "field": "email",
      "value": "invalid-email"
    },
    "retry_after": null
  }
}
```

### Pydantic Validation Handler

```python
async def pydantic_validation_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle Pydantic validation errors with user-friendly messages"""
```

#### Features:
- **Error Conversion**: Converts Pydantic errors to user-friendly format
- **Field Mapping**: Maps nested field paths to readable names
- **Message Generation**: Creates actionable error messages
- **Structured Logging**: Logs validation failures with context

### HTTP Exception Handler

```python
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with user-friendly messages"""
```

#### Features:
- **Status Code Mapping**: Maps HTTP status codes to user messages
- **Context Logging**: Logs HTTP exceptions with request context
- **Consistent Format**: Returns errors in standard format

### General Exception Handler

```python
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with proper logging"""
```

#### Features:
- **Unexpected Error Handling**: Catches unhandled exceptions
- **Full Context Logging**: Logs stack traces and error details
- **Safe User Response**: Returns generic message for security
- **Error Tracking**: Generates unique error IDs for tracking

## Error Response Headers

### Standard Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-Error-ID` | Unique error identifier | `abc12345` |
| `X-Error-Category` | Error category | `validation` |
| `Retry-After` | Seconds to wait before retry | `60` |
| `Content-Type` | Response content type | `application/json` |

### Usage in FastAPI

```python
from fastapi import FastAPI
from api.core.error_handling import (
    heygen_exception_handler,
    pydantic_validation_handler,
    http_exception_handler,
    general_exception_handler,
    HeyGenBaseError
)

app = FastAPI()

# Register exception handlers
app.add_exception_handler(HeyGenBaseError, heygen_exception_handler)
app.add_exception_handler(PydanticValidationError, pydantic_validation_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
```

## Logging Configuration

### Structured Logging Setup

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create structured logger for errors
error_logger = logging.getLogger('heygen.errors')
error_logger.setLevel(logging.ERROR)

# Create structured logger for user actions
user_logger = logging.getLogger('heygen.user_actions')
user_logger.setLevel(logging.INFO)
```

### Log Output Examples

#### Error Log Entry:
```json
{
  "error_id": "abc12345",
  "error_type": "ValidationError",
  "error_message": "Field 'email' is required",
  "timestamp": "2024-01-01T12:00:00Z",
  "user_id": "user123",
  "request_id": "req456",
  "operation": "user_registration",
  "context": {
    "field": "email",
    "value": null
  },
  "severity": "low",
  "category": "validation",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "field": "email"
  }
}
```

#### User Action Log Entry:
```json
{
  "action": "video_generation_started",
  "user_id": "user123",
  "request_id": "req456",
  "timestamp": "2024-01-01T12:00:00Z",
  "success": true,
  "details": {
    "video_id": "vid789",
    "template": "business_presentation"
  }
}
```

## Best Practices

### 1. Error Logging
- **Always log errors**: Use `ErrorLogger.log_error()` for all exceptions
- **Include context**: Provide relevant context for debugging
- **Use appropriate severity**: Set correct severity levels
- **Preserve user privacy**: Don't log sensitive user data

### 2. User Messages
- **Be helpful**: Provide actionable guidance
- **Be consistent**: Use consistent message patterns
- **Be concise**: Keep messages short and clear
- **Be positive**: Focus on solutions, not problems

### 3. Error Handling
- **Fail fast**: Validate early and return errors quickly
- **Be specific**: Provide specific error details when safe
- **Include retry info**: Add retry guidance when appropriate
- **Track errors**: Use error IDs for monitoring and debugging

### 4. Security
- **Don't expose internals**: Keep technical details in logs, not user messages
- **Sanitize data**: Remove sensitive information from logs
- **Rate limit errors**: Prevent error message abuse
- **Monitor patterns**: Watch for unusual error patterns

## Monitoring and Analytics

### Error Metrics
- **Error rates**: Track error frequency by category
- **User impact**: Monitor errors affecting user experience
- **Performance**: Track error response times
- **Trends**: Identify increasing error patterns

### Log Analysis
- **Error correlation**: Link errors to specific operations
- **User journey**: Track errors through user workflows
- **System health**: Monitor system-wide error patterns
- **Debugging**: Use error IDs for quick issue resolution

## Integration with Monitoring Tools

### Structured Logging for ELK Stack
```python
# Configure for ELK Stack
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'error_data'):
            log_entry.update(record.error_data)
        
        return json.dumps(log_entry)

# Apply formatter
error_logger.handlers[0].setFormatter(JSONFormatter())
```

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram

# Error metrics
error_counter = Counter('heygen_errors_total', 'Total errors', ['category', 'severity'])
error_duration = Histogram('heygen_error_duration_seconds', 'Error handling duration')

# Usage in exception handler
async def heygen_exception_handler(request: Request, exc: HeyGenBaseError) -> JSONResponse:
    start_time = time.time()
    
    # ... handle error ...
    
    # Record metrics
    error_counter.labels(
        category=exc.category.value,
        severity=exc.severity.value
    ).inc()
    
    error_duration.observe(time.time() - start_time)
    
    return response
```

## Conclusion

The enhanced error logging and user-friendly error message system provides:

1. **Comprehensive Error Tracking**: Every error is logged with full context
2. **User-Friendly Experience**: Clear, actionable error messages
3. **Developer-Friendly Debugging**: Structured logs with unique error IDs
4. **Monitoring Ready**: Integration with logging and monitoring tools
5. **Security Conscious**: Proper handling of sensitive information
6. **Scalable Design**: Easy to extend and customize

This system ensures that errors are properly tracked, users receive helpful feedback, and developers can quickly identify and resolve issues. 