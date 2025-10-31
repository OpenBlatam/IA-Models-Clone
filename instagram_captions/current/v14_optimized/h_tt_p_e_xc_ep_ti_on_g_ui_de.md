# 🚨 HTTP Exception Handling Guide - Instagram Captions API v14.0

## 📋 Overview

This guide documents the comprehensive HTTP exception handling system implemented in v14.0, featuring specific error types, structured responses, and proper HTTP status codes for all error scenarios.

## 🎯 **Exception Architecture**

### **1. Base Exception Classes**

#### **ErrorDetail Model**
```python
class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")
```

#### **APIErrorResponse Model**
```python
class APIErrorResponse(BaseModel):
    """Standardized API error response"""
    error: bool = Field(default=True, description="Error flag")
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")
    status_code: int = Field(description="HTTP status code")
```

## 🔴 **Client Errors (4xx)**

### **1. Validation Errors (400 Bad Request)**

#### **Base ValidationError**
```python
class ValidationError(HTTPException):
    """Base class for validation errors (400 Bad Request)"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None
    ):
        error_detail = ErrorDetail(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            path=path,
            method=method
        )
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_detail.model_dump()
        )
```

#### **Specific Validation Errors**

**ContentValidationError**
```python
# Usage
raise ContentValidationError(
    message="Content description must be at least 5 characters long",
    details={"field": "content_description", "value": "abc", "min_length": 5},
    request_id="req-123",
    path="/api/v14/generate",
    method="POST"
)

# Response
{
    "error_code": "CONTENT_VALIDATION_ERROR",
    "message": "Content description must be at least 5 characters long",
    "details": {
        "field": "content_description",
        "value": "abc",
        "min_length": 5
    },
    "request_id": "req-123",
    "path": "/api/v14/generate",
    "method": "POST",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**StyleValidationError**
```python
# Usage
raise StyleValidationError(
    message="Invalid caption style",
    details={"field": "style", "value": "invalid", "valid_styles": ["casual", "professional", "inspirational", "playful"]},
    request_id="req-123"
)

# Response
{
    "error_code": "STYLE_VALIDATION_ERROR",
    "message": "Invalid caption style",
    "details": {
        "field": "style",
        "value": "invalid",
        "valid_styles": ["casual", "professional", "inspirational", "playful"]
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**HashtagCountError**
```python
# Usage
raise HashtagCountError(
    message="Invalid hashtag count",
    details={"field": "hashtag_count", "value": 50, "min": 5, "max": 30},
    request_id="req-123"
)

# Response
{
    "error_code": "HASHTAG_COUNT_ERROR",
    "message": "Invalid hashtag count",
    "details": {
        "field": "hashtag_count",
        "value": 50,
        "min": 5,
        "max": 30
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**BatchSizeError**
```python
# Usage
raise BatchSizeError(
    message="Batch size exceeds maximum allowed size",
    details={"batch_size": 150, "max_batch_size": 100},
    request_id="batch-123"
)

# Response
{
    "error_code": "BATCH_SIZE_ERROR",
    "message": "Batch size exceeds maximum allowed size",
    "details": {
        "batch_size": 150,
        "max_batch_size": 100
    },
    "request_id": "batch-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **2. Authentication Errors**

#### **UnauthorizedError (401)**
```python
# Usage
raise UnauthorizedError(
    message="API key required",
    details={"header": "Authorization", "error_type": "missing_api_key"},
    request_id="req-123",
    path="/api/v14/generate",
    method="POST"
)

# Response
{
    "error_code": "UNAUTHORIZED",
    "message": "API key required",
    "details": {
        "header": "Authorization",
        "error_type": "missing_api_key"
    },
    "request_id": "req-123",
    "path": "/api/v14/generate",
    "method": "POST",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### **ForbiddenError (403)**
```python
# Usage
raise ForbiddenError(
    message="Invalid API key",
    details={"header": "Authorization", "error_type": "invalid_api_key"},
    request_id="req-123"
)

# Response
{
    "error_code": "FORBIDDEN",
    "message": "Invalid API key",
    "details": {
        "header": "Authorization",
        "error_type": "invalid_api_key"
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **3. Resource Errors**

#### **NotFoundError (404)**
```python
# Usage
raise NotFoundError(
    message="Resource not found",
    details={"resource_type": "caption", "resource_id": "cap-123"},
    request_id="req-123"
)

# Response
{
    "error_code": "NOT_FOUND",
    "message": "Resource not found",
    "details": {
        "resource_type": "caption",
        "resource_id": "cap-123"
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **4. Rate Limiting (429)**
```python
# Usage
raise TooManyRequestsError(
    message="Rate limit exceeded",
    retry_after=60,
    details={"rate_limit": 100, "window_seconds": 3600},
    request_id="req-123"
)

# Response
{
    "error_code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
        "rate_limit": 100,
        "window_seconds": 3600,
        "retry_after": 60
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔴 **Server Errors (5xx)**

### **1. AI Generation Errors**

#### **AIGenerationError (500)**
```python
# Usage
raise AIGenerationError(
    message="AI caption generation failed",
    details={"operation": "caption_generation", "error_type": "ModelError"},
    request_id="req-123"
)

# Response
{
    "error_code": "AI_GENERATION_ERROR",
    "message": "AI caption generation failed",
    "details": {
        "operation": "caption_generation",
        "error_type": "ModelError"
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **2. Cache Errors**

#### **CacheError (500)**
```python
# Usage
raise CacheError(
    message="Cache operation failed",
    details={"operation": "get", "cache_key": "key-123", "error_type": "ConnectionError"},
    request_id="req-123"
)

# Response
{
    "error_code": "CACHE_ERROR",
    "message": "Cache operation failed",
    "details": {
        "operation": "get",
        "cache_key": "key-123",
        "error_type": "ConnectionError"
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **3. Model Loading Errors**

#### **ModelLoadingError (500)**
```python
# Usage
raise ModelLoadingError(
    message="AI model loading failed",
    details={"model_name": "distilgpt2", "device": "cuda", "error": "CUDA out of memory"},
    request_id="req-123"
)

# Response
{
    "error_code": "MODEL_LOADING_ERROR",
    "message": "AI model loading failed",
    "details": {
        "model_name": "distilgpt2",
        "device": "cuda",
        "error": "CUDA out of memory"
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### **4. Service Unavailable (503)**
```python
# Usage
raise ServiceUnavailableError(
    message="Service temporarily unavailable",
    retry_after=300,
    details={"maintenance_window": "2024-01-15T12:00:00Z"},
    request_id="req-123"
)

# Response
{
    "error_code": "SERVICE_UNAVAILABLE",
    "message": "Service temporarily unavailable",
    "details": {
        "maintenance_window": "2024-01-15T12:00:00Z",
        "retry_after": 300
    },
    "request_id": "req-123",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🛠️ **Utility Functions**

### **1. Error Creation Helpers**

#### **handle_validation_error**
```python
def handle_validation_error(
    field: str,
    value: Any,
    message: str,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> ValidationError:
    """Create validation error with field details"""
    details = {
        "field": field,
        "value": str(value),
        "error_type": "validation"
    }
    
    return ValidationError(
        message=message,
        details=details,
        request_id=request_id,
        path=path,
        method=method
    )

# Usage
raise handle_validation_error(
    field="content_description",
    value="abc",
    message="Content description must be at least 5 characters long",
    request_id="req-123"
)
```

#### **handle_rate_limit_error**
```python
def handle_rate_limit_error(
    limit: int,
    window: int,
    retry_after: Optional[int] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> TooManyRequestsError:
    """Create rate limit error with details"""
    details = {
        "rate_limit": limit,
        "window_seconds": window,
        "error_type": "rate_limit"
    }
    
    return TooManyRequestsError(
        message=f"Rate limit exceeded: {limit} requests per {window} seconds",
        details=details,
        retry_after=retry_after,
        request_id=request_id,
        path=path,
        method=method
    )

# Usage
raise handle_rate_limit_error(
    limit=100,
    window=3600,
    retry_after=60,
    request_id="req-123"
)
```

#### **handle_ai_error**
```python
def handle_ai_error(
    operation: str,
    error: Exception,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> AIGenerationError:
    """Create AI generation error with operation details"""
    details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    return AIGenerationError(
        message=f"AI {operation} failed: {str(error)}",
        details=details,
        request_id=request_id,
        path=path,
        method=method
    )

# Usage
try:
    result = await ai_model.generate(prompt)
except Exception as e:
    raise handle_ai_error(
        operation="caption_generation",
        error=e,
        request_id="req-123"
    )
```

## 🔧 **FastAPI Integration**

### **1. Exception Handlers**

```python
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(AIGenerationError)
async def ai_generation_error_handler(request: Request, exc: AIGenerationError):
    """Handle AI generation errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(TooManyRequestsError)
async def rate_limit_error_handler(request: Request, exc: TooManyRequestsError):
    """Handle rate limit errors"""
    headers = {}
    if "retry_after" in exc.detail.get("details", {}):
        headers["Retry-After"] = str(exc.detail["details"]["retry_after"])
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
        headers=headers
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = create_error_response(
        error_code="INTERNAL_ERROR",
        message="An unexpected error occurred",
        status_code=500,
        details={"error_type": type(exc).__name__, "error_message": str(exc)},
        request_id=request_id,
        path=str(request.url.path),
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )
```

### **2. Dependency Injection**

```python
async def validate_api_key_dependency(request: Request) -> str:
    """Validate API key from request headers"""
    api_key = request.headers.get("Authorization")
    if not api_key:
        raise UnauthorizedError(
            message="API key required",
            details={"header": "Authorization", "error_type": "missing_api_key"},
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )
    
    if not validate_api_key(api_key):
        raise ForbiddenError(
            message="Invalid API key",
            details={"header": "Authorization", "error_type": "invalid_api_key"},
            request_id=getattr(request.state, "request_id", None),
            path=str(request.url.path),
            method=request.method
        )
    
    return api_key
```

## 📊 **Error Response Examples**

### **1. Validation Error Response**
```json
{
    "error_code": "CONTENT_VALIDATION_ERROR",
    "message": "Content description must be at least 5 characters long",
    "details": {
        "field": "content_description",
        "value": "abc",
        "min_length": 5,
        "error_type": "validation"
    },
    "request_id": "req-123456",
    "path": "/api/v14/generate",
    "method": "POST",
    "timestamp": "2024-01-15T10:30:00.123Z"
}
```

### **2. AI Generation Error Response**
```json
{
    "error_code": "AI_GENERATION_ERROR",
    "message": "AI caption generation failed: CUDA out of memory",
    "details": {
        "operation": "caption_generation",
        "error_type": "RuntimeError",
        "error_message": "CUDA out of memory"
    },
    "request_id": "req-123456",
    "path": "/api/v14/generate",
    "method": "POST",
    "timestamp": "2024-01-15T10:30:00.123Z"
}
```

### **3. Rate Limit Error Response**
```json
{
    "error_code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded: 100 requests per 3600 seconds",
    "details": {
        "rate_limit": 100,
        "window_seconds": 3600,
        "retry_after": 60,
        "error_type": "rate_limit"
    },
    "request_id": "req-123456",
    "path": "/api/v14/generate",
    "method": "POST",
    "timestamp": "2024-01-15T10:30:00.123Z"
}
```

## 🎯 **Best Practices**

### **1. Error Handling Patterns**

#### **Do:**
```python
# Use specific exception types
try:
    result = await ai_model.generate(prompt)
except torch.cuda.OutOfMemoryError as e:
    raise AIGenerationError(
        message="GPU memory exhausted",
        details={"operation": "caption_generation", "error_type": "CUDA_OOM"},
        request_id=request_id
    )

# Include request context
raise ValidationError(
    message="Invalid input",
    details={"field": "content", "value": user_input},
    request_id=request_id,
    path=str(request.url.path),
    method=request.method
)

# Use utility functions for common errors
raise handle_validation_error(
    field="hashtag_count",
    value=user_value,
    message="Hashtag count must be between 5 and 30",
    request_id=request_id
)
```

#### **Don't:**
```python
# Don't use generic exceptions
raise HTTPException(status_code=400, detail="Bad request")  # ❌

# Don't expose internal errors
raise HTTPException(status_code=500, detail=str(internal_error))  # ❌

# Don't forget request context
raise ValidationError("Invalid input")  # ❌ Missing request_id, path, method
```

### **2. Error Logging**

```python
# Log errors with context
logger.error(
    f"AI generation failed: {e}",
    extra={
        "request_id": request_id,
        "operation": "caption_generation",
        "error_type": type(e).__name__,
        "path": str(request.url.path),
        "method": request.method
    },
    exc_info=True
)
```

### **3. Error Monitoring**

```python
# Track error metrics
def record_error_metrics(error_type: str, request_id: str):
    """Record error metrics for monitoring"""
    # Increment error counter
    error_counter.labels(error_type=error_type).inc()
    
    # Record error details
    error_details.observe({
        "error_type": error_type,
        "request_id": request_id,
        "timestamp": time.time()
    })
```

## 📈 **Error Codes Reference**

| Error Code | HTTP Status | Description | Usage |
|------------|-------------|-------------|-------|
| `VALIDATION_ERROR` | 400 | General validation error | Base class for validation errors |
| `CONTENT_VALIDATION_ERROR` | 400 | Content validation failed | Invalid content description |
| `STYLE_VALIDATION_ERROR` | 400 | Style validation failed | Invalid caption style |
| `HASHTAG_COUNT_ERROR` | 400 | Hashtag count invalid | Invalid hashtag count |
| `BATCH_SIZE_ERROR` | 400 | Batch size invalid | Batch size exceeds limit |
| `UNAUTHORIZED` | 401 | Authentication required | Missing API key |
| `FORBIDDEN` | 403 | Access forbidden | Invalid API key |
| `NOT_FOUND` | 404 | Resource not found | Resource doesn't exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error | Unexpected errors |
| `AI_GENERATION_ERROR` | 500 | AI generation failed | AI model errors |
| `CACHE_ERROR` | 500 | Cache operation failed | Cache system errors |
| `MODEL_LOADING_ERROR` | 500 | Model loading failed | AI model loading errors |
| `SERVICE_UNAVAILABLE` | 503 | Service unavailable | Maintenance or overload |

---

This comprehensive HTTP exception handling system ensures consistent, informative, and properly structured error responses across the Instagram Captions API v14.0, providing clear feedback to clients and enabling effective error monitoring and debugging. 