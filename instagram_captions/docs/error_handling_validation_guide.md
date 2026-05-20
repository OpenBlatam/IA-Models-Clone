# Error Handling and Validation Guide

## Overview

This guide covers the comprehensive error handling and validation system for the Instagram Captions API, ensuring robust, secure, and user-friendly error management.

## Table of Contents

1dling Architecture](#error-handling-architecture)
2. [Custom Exceptions](#custom-exceptions)
3. [Validation System](#validation-system)
4or Response Models](#error-response-models)
5. [Input Validation](#input-validation)
6. [Security Validation](#security-validation)
7. [FastAPI Integration](#fastapi-integration)
8[Best Practices](#best-practices)
9. [Examples](#examples)

## Error Handling Architecture

### Core Components

- **Custom Exceptions**: Domain-specific exceptions with proper HTTP status codes
- **Error Response Models**: Standardized error response structure
- **Validation Decorators**: Automatic input validation and error handling
- **Global Exception Handler**: Centralized error handling for FastAPI
- **Error Logging**: Comprehensive error tracking and monitoring

### Error Flow

```
Request → Validation → Processing → Response
    ↓         ↓           ↓          ↓
  Input    Pydantic    Business   Success/
Validation  Models      Logic      Error
    ↓         ↓           ↓          ↓
  Error    Field      Custom      Standardized
Response  Errors     Exceptions   Error Response
```

## Custom Exceptions

### Base Exception

```python
class InstagramCaptionsException(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500   ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now(timezone.utc)
        self.request_id = str(uuid.uuid4())
```

### Specific Exceptions

```python
# Validation Errors (400)
class ValidationException(InstagramCaptionsException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
            status_code=400
        )

# Authentication Errors (401)
class AuthenticationException(InstagramCaptionsException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.UNAUTHORIZED,
            message=message,
            details=details,
            status_code=401
        )

# Resource Errors (44eNotFoundException(InstagramCaptionsException):
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            error_code=ErrorCode.NOT_FOUND,
            message=f"{resource_type} with id '{resource_id}' not found",
            details={"resource_type": resource_type, "resource_id": resource_id},
            status_code=404
        )

# Rate Limiting (429RateLimitException(InstagramCaptionsException):
    def __init__(self, retry_after: int = 60        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message="Rate limit exceeded. Please try again later.",
            details={retry_after": retry_after},
            status_code=429)

# AI Processing Errors (422)
class AIProcessingException(InstagramCaptionsException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            error_code=ErrorCode.AI_PROCESSING_ERROR,
            message=message,
            details=details,
            status_code=422
        )
```

## Validation System

### Pydantic v2 Models

```python
class CaptionRequest(BaseValidationModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    content_type: ContentType = Field(default=ContentType.POST)
    tone: ToneType = Field(default=ToneType.PROFESSIONAL)
    hashtags: List[str] = Field(default_factory=list)
    max_length: Optionalint] = Field(None, ge=1, le=2200
    include_hashtags: bool = Field(default=True)

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        v = re.sub(r'\s+', ' ', v.strip())
        v = html.escape(v)
        return v

    @field_validator('hashtags')
    @classmethod
    def validate_hashtags(cls, v: List[str]) -> List[str]:
        if not v:
            return v
        
        sanitized = []
        for hashtag in v:
            clean_tag = re.sub(r'[^\w\-_]',, hashtag.strip().lstrip('#))          if clean_tag and len(clean_tag) <= 30             sanitized.append(f#{clean_tag.lower()}")
        
        unique_hashtags =        seen = set()
        for tag in sanitized:
            if tag not in seen:
                unique_hashtags.append(tag)
                seen.add(tag)
        
        return unique_hashtags[:30]

    @model_validator(mode='after')
    def validate_total_length(self) -> CaptionRequest:
        if not self.include_hashtags:
            return self
        
        hashtag_length = sum(len(tag) +1or tag in self.hashtags)
        effective_max = self.max_length or CONTENT_LIMITS[self.content_type]
        total_length = len(self.prompt) + hashtag_length
        
        if total_length > effective_max:
            raise ValueError(
                f"Total length ({total_length}) would exceed limit ({effective_max})"
                f" for {self.content_type}"
            )
        
        return self
```

### Content Type Limits

```python
CONTENT_LIMITS =[object Object]  ContentType.POST:2200
    ContentType.STORY: 500,
    ContentType.REEL:1000
    ContentType.CAROUSEL:2200,
    ContentType.IGTV: 220
## Error Response Models

### Standard Error Response

```python
class ErrorResponse(BaseModel):
    error_code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details)   request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    path: Optionalstr] = Field(None, description="Request path")
    method: Optionalstr] = Field(None, description="HTTP method")
```

### Validation Error Response

```python
class ValidationErrorResponse(ErrorResponse):
    field_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Field-specific validation errors")
```

## Input Validation

### String Validation

```python
def validate_string(*, config: StringValidationConfig) -> Dict[str, Any]:
    lidate string with comprehensive rules (RORO)."""
    value = config.value
    
    # Check if empty
    if not config.allow_empty and (not value or not value.strip()):
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be empty",
       error_code": ErrorCode.MISSING_REQUIRED_FIELD
        }
    
    # Check minimum length
    if config.min_length and len(value) < config.min_length:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at least [object Object]config.min_length} characters",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check maximum length
    if config.max_length and len(value) > config.max_length:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at most [object Object]config.max_length} characters",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check pattern
    if config.pattern:
        import re
        if not re.match(config.pattern, value):
            return[object Object]
               is_valid": False,
                error: f[object Object]config.field_name} format is invalid,
           error_code": ErrorCode.INVALID_FORMAT
            }
    
    return {"is_valid": True, "value: value.strip()}
```

### Numeric Validation

```python
def validate_numeric(*, config: NumericValidationConfig) -> Dict[str, Any]:
    "date numeric value with comprehensive rules (RORO)."""
    value = config.value
    
    # Check zero
    if not config.allow_zero and value ==0
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be zero",
       error_code": ErrorCode.INVALID_INPUT
        }
    
    # Check negative
    if not config.allow_negative and value <0
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} cannot be negative",
       error_code": ErrorCode.INVALID_INPUT
        }
    
    # Check minimum value
    if config.min_value is not None and value < config.min_value:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at least {config.min_value}",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    # Check maximum value
    if config.max_value is not None and value > config.max_value:
        return[object Object]        is_valid": False,
            error: f[object Object]config.field_name} must be at most {config.max_value}",
       error_code": ErrorCode.OUT_OF_RANGE
        }
    
    return {"is_valid": True, "value": value}
```

## Security Validation

### HTML Sanitization

```python
def sanitize_html(*, html_content: str, allowed_tags: List[str] = None) -> Dict[str, Any]:
  itize HTML content (RORO).   if allowed_tags is None:
        allowed_tags =bi', u, 'strong',em]  
    if not html_content:
        return[object Object]sanitized":,removed_tags": []}
    
    # Remove all HTML tags except allowed ones
    pattern = r'<(/?)([^>]+)>|<!--.*?-->    removed_tags = []
    
    def replace_tag(match):
        tag = match.group(2).split()[0lower()
        if tag not in allowed_tags:
            removed_tags.append(tag)
            return      return match.group(0)
    
    sanitized = re.sub(pattern, replace_tag, html_content)
    
    return {
        sanitized": sanitized,
     removed_tags": list(set(removed_tags))
    }
```

### Instagram Username Validation

```python
def validate_instagram_username(*, username: str) -> Dict[str, Any]:
    """Validate Instagram username (RORO)."""
    username_pattern = r^[a-zA-Z0]{1,30}$'
    
    if not username or not username.strip():
        return [object Object]is_valid:falseerror: "Username cannot be empty"}
    
    username = username.strip()
    
    if len(username) < 1or len(username) > 30
        return [object Object]is_valid:falseerror": Username must be 1characters}    
    if not re.match(username_pattern, username):
        return [object Object]is_valid:falseerrorUsername contains invalid characters"}
    
    # Check for reserved words
    reserved_words = ['admin', instagram',meta',facebook', 'help',support] if username.lower() in reserved_words:
        return [object Object]is_valid:falseerror": "Username is reserved}    
    return {"is_valid:truesername: username}
```

## FastAPI Integration

### Global Exception Handler

```python
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    " exception handler for FastAPI.  request_id = str(uuid.uuid4())
    
    if isinstance(exc, InstagramCaptionsException):
        error_response = create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = exc.status_code
    elif isinstance(exc, ValidationError):
        field_errors = []
        for error in exc.errors():
            field_errors.append({
               field":..join(str(loc) for loc in error["loc"]),
                message": error["msg"],
             type": error["type]      })
        
        error_response = create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Input validation failed",
            details={"field_errors: field_errors},
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = 400
    else:
        # Log unexpected errors
        log_error(error=exc, request_id=request_id)
        
        error_response = create_error_response(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method
        )
        status_code = 50  
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )
```

### API Route with Validation

```python
from fastapi import APIRouter, Depends
from .validation import CaptionRequest, CaptionResponse
from .error_handling import handle_api_errors, ValidationException

router = APIRouter()

@router.post("/generate, response_model=CaptionResponse)
@handle_api_errors
async def generate_caption(request: CaptionRequest) -> CaptionResponse:
    """Generate Instagram caption with full validation.""    try:
        # Business logic here
        caption = await generate_caption_logic(request)
        
        return CaptionResponse(
            caption=caption,
            content_type=request.content_type,
            length=len(caption),
            hashtags=request.hashtags,
            tone=request.tone,
            generation_time=0.5,
            quality_score=85      )
    except Exception as e:
        raise ValidationException(
            message="Failed to generate caption",
            details={"error:str(e)}
        )
```

## Best Practices

### 1. Use Named Exports

```python
__all__ = [
    ErrorCode",
    InstagramCaptionsException,
    alidationException",create_error_response,
    handle_api_errors",
  validate_string,validate_numeric"
]
```

### 2. Follow RORO Pattern

```python
# Good: RORO pattern
def validate_string(*, config: StringValidationConfig) -> Dict[str, Any]:
    # Implementation
    return {"is_valid": True, value: value}

# Usage
result = validate_string(config=StringValidationConfig(value=test", field_name="username"))

# Bad: Multiple parameters
def validate_string(value: str, field_name: str) -> str:
    # Implementation
    return value
```

### 3. Comprehensive Error Messages

```python
# Good: Detailed error message
raise ValidationException(
    message="Input validation failed",
    details={
        field_errors": [
           [object Object]
                field,
          message": "Invalid email format,
            type": "value_error"
            }
        ]
    }
)

# Bad: Generic error message
raise ValueError("Invalid input")
```

### 4. Proper HTTP Status Codes

```python
# Use appropriate status codes
400: Validation errors
41hentication errors
403thorization errors404Resource not found
429 Rate limiting
422: Processing errors
500rnal server errors
```

### 5Error Logging

```python
def log_error(*, error: Exception, request_id: Optional[str] = None, context: Optional[Dictstr, Any]] = None) -> Dict[str, Any]:
   error with context (RORO).   error_data = {
        error_type": type(error).__name__,
      error_message": str(error),
   request_id": request_id or str(uuid.uuid4()),
  timestamp": datetime.now(timezone.utc).isoformat(),
        context: context or [object Object]        traceback": traceback.format_exc()
    }
    
    logger.error(fError occurred: {error_data}")
    return error_data
```

## Examples

### 1. Caption Generation with Validation

```python
from .validation import CaptionRequest, validate_caption_content
from .error_handling import handle_api_errors, AIProcessingException

@handle_api_errors
async def generate_caption(request: CaptionRequest) -> Dict[str, Any]:
    # Validate caption content
    validation_result = validate_caption_content(
        caption=request.prompt,
        content_type=request.content_type
    )
    
    if not validation_result["is_valid"]:
        raise ValidationException(
            message=validation_result["error"],
            details=validation_result
        )
    
    try:
        # Generate caption
        caption = await ai_service.generate(request)
        return {caption": caption, status": "success"}
    except Exception as e:
        raise AIProcessingException(
            message="Failed to generate caption",
            details={"error:str(e)}
        )
```

### 2. User Input Validation

```python
from .validation import validate_email, validate_instagram_username

def validate_user_input(*, email: str, username: str) -> Dict[str, Any]:
    # Validate email
    email_result = validate_email(email=email)
    if not email_result["is_valid]:      return email_result
    
    # Validate username
    username_result = validate_instagram_username(username=username)
    if not username_result["is_valid]:   return username_result
    
    return {
        is_valid: True,
    email": email_result["email"],
 username: username_result[username]
    }
```

###3r Response Example

```json[object Object]
 error_code":VALIDATION_ERROR",
   messageInput validation failed",details":[object Object]     field_errors": [
           [object Object]
                field,
          message": "Invalid email format,
            type": "value_error"
            },
           [object Object]
                field": "username,
          message": Username must be 1,
            type": "value_error"
            }
        ]
    },
    request_id: 55040e29b-41-a716446655440000,
    timestamp: 2024115T10:30:00Z",
    path/api/v1rate,  method": "POST"
}
```

## Conclusion

This comprehensive error handling and validation system ensures:

- **Robust Error Management**: Custom exceptions with proper HTTP status codes
- **Input Validation**: Pydantic v2 models with field validators
- **Security**: Input sanitization and security validation
- **User Experience**: Clear, actionable error messages
- **Monitoring**: Comprehensive error logging and tracking
- **Maintainability**: RORO pattern and named exports
- **Type Safety**: Full type hints throughout the system

The system follows Python and FastAPI best practices while providing a solid foundation for production-ready error handling and validation. 