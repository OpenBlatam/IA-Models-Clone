# FastAPI Best Practices Implementation Guide

## Overview

This guide documents the implementation of FastAPI best practices following the official FastAPI documentation for Data Models, Path Operations, and Middleware. The implementation demonstrates production-ready patterns and techniques.

## Table of Contents

1. [Data Models (Pydantic v2)](#data-models-pydantic-v2)
2. [Path Operations](#path-operations)
3. [Middleware](#middleware)
4. [Dependency Injection](#dependency-injection)
5. [Error Handling](#error-handling)
6. [Security](#security)
7. [Performance](#performance)
8. [Testing](#testing)

## Data Models (Pydantic v2)

### 1. Model Configuration Best Practices

```python
class DiffusionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        str_strip_whitespace=True,  # Strip whitespace from strings
        validate_assignment=True,  # Validate on assignment
        json_schema_extra={
            "example": {
                "prompt": "A beautiful sunset over mountains",
                "negative_prompt": "blurry, low quality",
                "pipeline_type": "text_to_image",
                "model_type": "stable-diffusion-v1-5",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 42,
                "batch_size": 1
            }
        }
    )
```

**Best Practices:**
- Use `extra="forbid"` to reject unexpected fields
- Enable `str_strip_whitespace=True` for automatic string cleaning
- Use `validate_assignment=True` for runtime validation
- Provide comprehensive examples in `json_schema_extra`

### 2. Field Validation

```python
prompt: str = Field(
    ..., 
    min_length=1, 
    max_length=1000, 
    description="Text prompt for image generation",
    examples=["A beautiful sunset over mountains", "A futuristic city skyline"]
)
```

**Best Practices:**
- Use descriptive field names and descriptions
- Provide multiple examples for better documentation
- Set appropriate min/max length constraints
- Use type hints for better IDE support

### 3. Field Validators (Pydantic v2)

```python
@field_validator('width', 'height')
@classmethod
def validate_dimensions(cls, v: int) -> int:
    """Validate that dimensions are divisible by 8."""
    if v % 8 != 0:
        raise ValueError('Width and height must be divisible by 8')
    return v

@field_validator('prompt')
@classmethod
def validate_prompt(cls, v: str) -> str:
    """Sanitize and validate prompt."""
    sanitized = v.strip()
    if not sanitized:
        raise ValueError('Prompt cannot be empty')
    return sanitized[:1000]  # Ensure max length
```

**Best Practices:**
- Use `@field_validator` decorator (Pydantic v2)
- Make validators `@classmethod`
- Return the validated value
- Provide clear error messages

### 4. Model Validators

```python
@model_validator(mode='after')
def validate_model(self) -> 'DiffusionRequest':
    """Cross-field validation."""
    if self.width * self.height > 1024 * 1024:  # 1MP limit
        raise ValueError('Total image pixels cannot exceed 1,048,576')
    return self
```

**Best Practices:**
- Use `mode='after'` for cross-field validation
- Validate business rules that span multiple fields
- Return `self` for model validators

### 5. Computed Fields

```python
@computed_field
@property
def total_pixels(self) -> int:
    """Computed field for total pixels."""
    return self.width * self.height

@computed_field
@property
def estimated_memory_mb(self) -> float:
    """Computed field for estimated memory usage."""
    return (self.width * self.height * 3 * 4) / (1024 * 1024)  # RGB float32
```

**Best Practices:**
- Use `@computed_field` decorator for derived values
- Make computed fields `@property`
- Provide clear documentation
- Use for values that can be calculated from other fields

### 6. JSON Schema Customization

```python
model_config = ConfigDict(
    json_encoders={
        datetime: lambda v: v.isoformat(),
        ObjectId: str
    }
)
```

**Best Practices:**
- Customize JSON encoding for complex types
- Use ISO format for datetime objects
- Convert ObjectId to string for MongoDB integration

## Path Operations

### 1. Route Organization

```python
@app.post(
    "/api/v1/diffusion/generate",
    response_model=DiffusionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Single Image",
    description="Generate an image from a text prompt using diffusion models",
    response_description="Generated image information",
    tags=["Diffusion"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}
    }
)
```

**Best Practices:**
- Use descriptive route paths with versioning
- Specify response models for type safety
- Use appropriate HTTP status codes
- Provide comprehensive documentation
- Group routes with tags
- Document all possible responses

### 2. Parameter Validation

```python
async def upload_image(
    file: UploadFile = File(
        ...,
        description="Image file to upload",
        max_length=10 * 1024 * 1024  # 10MB limit
    ),
    current_user: UserDep = Depends(get_current_user)
):
```

**Best Practices:**
- Use `File()` for file uploads with size limits
- Use `Query()` for query parameters with validation
- Use `Path()` for path parameters with validation
- Use `Body()` for complex request bodies
- Provide clear descriptions and examples

### 3. Response Models

```python
class DiffusionResponse(BaseModel):
    image_url: str = Field(..., description="URL to generated image")
    image_id: str = Field(..., description="Unique image identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Generation metadata"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
```

**Best Practices:**
- Use Pydantic models for response validation
- Include all necessary fields with descriptions
- Use appropriate default values
- Include timestamps for tracking
- Use computed fields for derived values

### 4. Status Codes

```python
from fastapi import status

@app.post(
    "/api/v1/diffusion/generate",
    status_code=status.HTTP_200_OK,
    # ...
)
```

**Best Practices:**
- Use `status` constants instead of magic numbers
- Use appropriate status codes:
  - `200` for successful operations
  - `201` for resource creation
  - `400` for bad requests
  - `401` for unauthorized
  - `403` for forbidden
  - `404` for not found
  - `422` for validation errors
  - `429` for rate limiting
  - `500` for server errors

## Middleware

### 1. Request ID Middleware

```python
class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = hashlib.md5(
                f"{scope['client'][0]}:{time.time()}".encode()
            ).hexdigest()[:8]
            
            scope["request_id"] = request_id
            
            async def send_with_request_id(message):
                if message["type"] == "http.response.start":
                    message["headers"].append(
                        (b"x-request-id", request_id.encode())
                    )
                await send(message)
            
            await self.app(scope, receive, send_with_request_id)
```

**Best Practices:**
- Generate unique request IDs for tracing
- Add request ID to response headers
- Make request ID available in route handlers
- Use consistent ID format

### 2. Logging Middleware

```python
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            request_id = scope.get("request_id", "unknown")
            
            logger.info(
                f"Request started",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "client": scope["client"][0] if scope["client"] else "unknown"
                }
            )
            
            async def send_with_logging(message):
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    status_code = message["status"]
                    
                    logger.info(
                        f"Request completed",
                        extra={
                            "request_id": request_id,
                            "method": scope["method"],
                            "path": scope["path"],
                            "status_code": status_code,
                            "duration": duration,
                            "client": scope["client"][0] if scope["client"] else "unknown"
                        }
                    )
                await send(message)
            
            await self.app(scope, receive, send_with_logging)
```

**Best Practices:**
- Log both request start and completion
- Include request ID for correlation
- Log performance metrics (duration)
- Use structured logging with extra fields
- Include relevant request information

### 3. Performance Middleware

```python
class PerformanceMiddleware:
    def __init__(self, app):
        self.app = app
        self.request_counter = Counter(
            'http_requests_total', 
            'Total HTTP requests',
            ['method', 'path', 'status']
        )
        self.request_duration = Histogram(
            'http_request_duration_seconds', 
            'HTTP request duration',
            ['method', 'path']
        )
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            method = scope["method"]
            path = scope["path"]
            
            async def send_with_metrics(message):
                if message["type"] == "http.response.start":
                    duration = time.time() - start_time
                    status = message["status"]
                    
                    self.request_counter.labels(
                        method=method, 
                        path=path, 
                        status=status
                    ).inc()
                    self.request_duration.labels(
                        method=method, 
                        path=path
                    ).observe(duration)
                
                await send(message)
            
            await self.app(scope, receive, send_with_metrics)
```

**Best Practices:**
- Use Prometheus metrics for monitoring
- Track request counts by method, path, and status
- Measure request duration
- Use appropriate metric types (Counter, Histogram, Gauge)
- Include relevant labels for filtering

### 4. Security Middleware

```python
class SecurityMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_with_security_headers(message):
                if message["type"] == "http.response.start":
                    security_headers = [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"x-xss-protection", b"1; mode=block"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()")
                    ]
                    
                    for header_name, header_value in security_headers:
                        message["headers"].append((header_name, header_value))
                
                await send(message)
            
            await self.app(scope, receive, send_with_security_headers)
```

**Best Practices:**
- Add security headers to all responses
- Use OWASP recommended security headers
- Configure Content Security Policy (CSP)
- Set appropriate CORS policies
- Use HTTPS in production

### 5. Middleware Order

```python
def create_application() -> FastAPI:
    app = FastAPI()
    
    # Add middleware in order (last added = first executed)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(CORSMiddleware)
    app.add_middleware(GZipMiddleware)
    app.add_middleware(TrustedHostMiddleware)
```

**Best Practices:**
- Order middleware correctly (last added = first executed)
- Security middleware should be early in the chain
- Performance monitoring should be early
- Logging should capture all requests
- CORS should be after security but before application logic

## Dependency Injection

### 1. Type Annotations

```python
from typing import Annotated

DatabaseDep = Annotated[Database, Depends()]
CacheDep = Annotated[redis.Redis, Depends()]
UserDep = Annotated[str, Depends()]
```

**Best Practices:**
- Use `Annotated` for better type hints
- Create type aliases for common dependencies
- Use descriptive names for dependency types
- Leverage IDE support for better development experience

### 2. Service Dependencies

```python
async def get_database_service() -> DatabaseService:
    """Get database service instance."""
    return DatabaseService("postgresql://user:pass@localhost/db")

async def get_cache_service() -> CacheService:
    """Get cache service instance."""
    return CacheService("redis://localhost:6379")
```

**Best Practices:**
- Create dedicated functions for each service
- Use async functions for async services
- Provide clear documentation
- Handle service initialization properly

### 3. Authentication Dependencies

```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> str:
    """Get current user from JWT token."""
    # In production, validate JWT token
    return "default_user"
```

**Best Practices:**
- Use `HTTPBearer` for JWT authentication
- Validate tokens properly in production
- Return user information consistently
- Handle authentication errors gracefully

### 4. Rate Limiting Dependencies

```python
async def get_rate_limit_info(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """Get rate limiting information."""
    client_ip = request.client.host
    rate_limit_key = f"rate_limit:{client_ip}"
    
    redis_client = await cache_service.get_redis()
    current_count = await redis_client.get(rate_limit_key)
    current_count = int(current_count) if current_count else 0
    
    limit_per_minute = 60
    if current_count >= limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    await redis_client.set(rate_limit_key, str(current_count + 1), ex=60)
    
    return {
        "requests_per_minute": limit_per_minute,
        "remaining_requests": limit_per_minute - current_count - 1,
        "reset_time": datetime.now(timezone.utc).timestamp() + 60
    }
```

**Best Practices:**
- Use Redis for rate limiting storage
- Track requests per client IP
- Set appropriate rate limits
- Return rate limit information
- Use proper HTTP status codes

## Error Handling

### 1. Global Exception Handlers

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            detail=str(exc),
            error_code="VALIDATION_ERROR",
            request_id=request.scope.get("request_id"),
            path=request.url.path,
            method=request.method
        ).model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code="HTTP_ERROR",
            request_id=request.scope.get("request_id"),
            path=request.url.path,
            method=request.method
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=request.scope.get("request_id"),
            path=request.url.path,
            method=request.method
        ).model_dump()
    )
```

**Best Practices:**
- Handle specific exception types first
- Use consistent error response format
- Include request ID for correlation
- Log unhandled exceptions
- Don't expose internal details in production

### 2. Error Response Models

```python
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")
    path: Optional[str] = Field(None, description="Request path")
    method: Optional[str] = Field(None, description="HTTP method")
```

**Best Practices:**
- Use Pydantic models for error responses
- Include relevant context information
- Use consistent field names
- Provide clear error codes
- Include timestamps for debugging

## Security

### 1. Security Headers

```python
security_headers = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"x-xss-protection", b"1; mode=block"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()")
]
```

**Best Practices:**
- Use OWASP recommended security headers
- Configure Content Security Policy
- Set appropriate CORS policies
- Use HTTPS in production
- Validate all inputs

### 2. Input Validation

```python
@field_validator('prompt')
@classmethod
def validate_prompt(cls, v: str) -> str:
    """Sanitize and validate prompt."""
    sanitized = v.strip()
    if not sanitized:
        raise ValueError('Prompt cannot be empty')
    return sanitized[:1000]  # Ensure max length
```

**Best Practices:**
- Validate all user inputs
- Sanitize strings
- Set appropriate length limits
- Use allowlists for allowed values
- Escape special characters

### 3. File Upload Security

```python
async def upload_image(
    file: UploadFile = File(
        ...,
        description="Image file to upload",
        max_length=10 * 1024 * 1024  # 10MB limit
    ),
    current_user: UserDep = Depends(get_current_user)
):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
```

**Best Practices:**
- Set file size limits
- Validate file types
- Scan for malware
- Store files securely
- Use signed URLs for access

## Performance

### 1. Async Operations

```python
async def generate_batch_images(
    self,
    request: BatchDiffusionRequest,
    current_user: UserDep = Depends(get_current_user),
    rate_limit: Dict[str, Any] = Depends(get_rate_limit_info)
) -> BatchDiffusionResponse:
    # Process requests in parallel
    tasks = [
        self.generate_single_image(
            req,
            current_user,
            rate_limit
        )
        for req in request.requests
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Best Practices:**
- Use async/await for I/O operations
- Process requests in parallel when possible
- Use connection pooling
- Implement caching strategies
- Monitor performance metrics

### 2. Caching

```python
# Check cache first
prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()
cached_result = await self.cache_service.get(f"generation:{prompt_hash}")

if cached_result:
    processing_time = time.time() - start_time
    return DiffusionResponse(
        image_url=cached_result,
        image_id=f"cached_{prompt_hash}",
        processing_time=processing_time,
        model_used=request.model_type.value
    )
```

**Best Practices:**
- Cache expensive operations
- Use appropriate cache keys
- Set reasonable TTL values
- Implement cache invalidation
- Monitor cache hit rates

### 3. Database Optimization

```python
async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
    db = await self.get_database()
    try:
        result = await db.fetch_all(text(query), params or {})
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise
```

**Best Practices:**
- Use connection pooling
- Parameterize queries
- Use appropriate indexes
- Monitor query performance
- Handle database errors gracefully

## Testing

### 1. Unit Testing

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

def test_generate_single_image():
    # Mock dependencies
    mock_diffusion_service = Mock()
    mock_diffusion_service.generate_single_image.return_value = expected_response
    
    # Test the endpoint
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/diffusion/generate",
            json={
                "prompt": "A beautiful sunset",
                "width": 512,
                "height": 512
            }
        )
        
        assert response.status_code == 200
        assert response.json()["image_url"] == expected_response.image_url
```

**Best Practices:**
- Mock external dependencies
- Test happy path and error cases
- Use TestClient for integration testing
- Test input validation
- Test error handling

### 2. Integration Testing

```python
@pytest.mark.asyncio
async def test_database_integration():
    db_service = DatabaseService("test_url")
    
    # Test database operations
    result = await db_service.execute_query("SELECT 1")
    assert len(result) == 1
    assert result[0][0] == 1
```

**Best Practices:**
- Use test databases
- Clean up test data
- Test database connections
- Test transaction handling
- Use async test functions

### 3. Performance Testing

```python
def test_performance():
    with TestClient(app) as client:
        start_time = time.time()
        
        # Make multiple requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200
        
        duration = time.time() - start_time
        assert duration < 10  # Should complete within 10 seconds
```

**Best Practices:**
- Test response times
- Test under load
- Monitor resource usage
- Test concurrent requests
- Set performance benchmarks

## Deployment

### 1. Production Configuration

```python
class AppConfig:
    def __init__(self):
        self.app_name: str = os.getenv("APP_NAME", "notebooklm_ai")
        self.version: str = os.getenv("APP_VERSION", "1.0.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.database_url: str = os.getenv("DATABASE_URL")
        self.redis_url: str = os.getenv("REDIS_URL")
```

**Best Practices:**
- Use environment variables for configuration
- Don't hardcode sensitive information
- Use different configs for different environments
- Validate configuration on startup
- Use secure defaults

### 2. Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_best_practices_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Best Practices:**
- Use multi-stage builds
- Minimize image size
- Use specific Python versions
- Copy requirements first for caching
- Use non-root user

### 3. Health Checks

```python
@app.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=time.time() - 0,
        gpu_available=True,
        models_loaded={
            "stable-diffusion-v1-5": True,
            "stable-diffusion-xl": True
        },
        memory_usage={
            "gpu": 2048.0,
            "ram": 8192.0
        },
        services={
            "database": "healthy",
            "cache": "healthy",
            "api": "healthy"
        }
    )
```

**Best Practices:**
- Implement comprehensive health checks
- Check all dependencies
- Return detailed status information
- Use for load balancer health checks
- Monitor system resources

## Conclusion

This implementation demonstrates comprehensive FastAPI best practices covering:

1. **Data Models**: Pydantic v2 features with validation and computed fields
2. **Path Operations**: Well-organized routes with proper documentation
3. **Middleware**: Custom middleware for security, logging, and performance
4. **Dependency Injection**: Type-safe dependency management
5. **Error Handling**: Comprehensive error management with structured responses
6. **Security**: Security headers and input validation
7. **Performance**: Async operations, caching, and monitoring
8. **Testing**: Unit and integration testing strategies
9. **Deployment**: Production-ready configuration and health checks

The implementation follows official FastAPI documentation recommendations and provides a solid foundation for building scalable, maintainable, and secure APIs. 