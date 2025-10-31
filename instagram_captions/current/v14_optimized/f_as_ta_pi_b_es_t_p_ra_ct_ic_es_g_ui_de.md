# FastAPI Best Practices Guide

## Overview

This guide documents the implementation of FastAPI best practices for the Instagram Captions API v14.0, covering Data Models, Path Operations, and Middleware based on the official FastAPI documentation and industry standards.

## Table of Contents

1. [Data Models Best Practices](#data-models-best-practices)
2. [Path Operations Best Practices](#path-operations-best-practices)
3. [Middleware Best Practices](#middleware-best-practices)
4. [Security Best Practices](#security-best-practices)
5. [Performance Best Practices](#performance-best-practices)
6. [Testing Best Practices](#testing-best-practices)
7. [Documentation Best Practices](#documentation-best-practices)

## Data Models Best Practices

### 1. Pydantic v2 Models

#### Base Model Configuration
```python
class BaseModelWithConfig(BaseModel):
    """Base model with common configuration"""
    
    model_config = ConfigDict(
        # Use alias for field names (camelCase in JSON, snake_case in Python)
        populate_by_name=True,
        # Allow extra fields (useful for API versioning)
        extra="ignore",
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # JSON serialization options
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
            uuid.UUID: lambda v: str(v)
        },
        # Example generation
        json_schema_extra={
            "examples": [
                {
                    "name": "Example Model",
                    "description": "This is an example model"
                }
            ]
        }
    )
```

#### Field Validation
```python
class CaptionGenerationRequest(BaseModelWithConfig):
    content_description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed description of the content for caption generation",
        examples=[
            "Beautiful sunset over mountains with golden light reflecting on a calm lake",
            "Delicious homemade pizza with melted cheese and fresh basil"
        ]
    )
    
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Desired caption style",
        examples=["casual", "formal", "creative"]
    )
    
    hashtag_count: int = Field(
        default=15,
        ge=0,
        le=30,
        description="Number of hashtags to generate",
        examples=[10, 15, 20]
    )
```

#### Custom Validation
```python
@field_validator('content_description')
@classmethod
def validate_content_description(cls, v: str) -> str:
    """Validate content description"""
    if not v.strip():
        raise ValueError("Content description cannot be empty")
    if len(v.split()) < 3:
        raise ValueError("Content description must have at least 3 words")
    return v.strip()

@model_validator(mode='after')
def validate_model(self) -> 'CaptionGenerationRequest':
    """Validate the entire model"""
    if self.max_length and len(self.content_description) > self.max_length:
        raise ValueError("Content description exceeds maximum length")
    return self
```

#### Computed Fields
```python
class CaptionGenerationResponse(BaseModelWithConfig):
    caption: str = Field(..., description="Generated caption text")
    hashtags: List[str] = Field(..., description="Generated hashtags")
    
    @computed_field
    @property
    def total_length(self) -> int:
        """Total length including hashtags"""
        caption_length = len(self.caption)
        hashtags_length = sum(len(hashtag) for hashtag in self.hashtags)
        return caption_length + hashtags_length
    
    @computed_field
    @property
    def is_within_limits(self) -> bool:
        """Check if caption is within Instagram limits"""
        return self.total_length <= 2200
```

### 2. Enum Usage for Type Safety

```python
class CaptionStyle(str, Enum):
    """Caption style enumeration"""
    CASUAL = "casual"
    FORMAL = "formal"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    FUNNY = "funny"
    INSPIRATIONAL = "inspirational"
    MINIMAL = "minimal"
    DETAILED = "detailed"

class LanguageCode(str, Enum):
    """Language code enumeration"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
```

### 3. Error Response Models

```python
class ErrorDetail(BaseModelWithConfig):
    """Error detail model"""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")

class ErrorResponse(BaseModelWithConfig):
    """Standard error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")
```

## Path Operations Best Practices

### 1. HTTP Methods and Status Codes

#### POST for Creation
```python
@router.post(
    "/captions/generate",
    response_model=CaptionGenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Instagram Caption",
    description="Generate a high-quality Instagram caption based on content description",
    response_description="Successfully generated caption with metadata"
)
async def generate_caption(
    request: CaptionGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: CoreDependencies = Depends()
) -> CaptionGenerationResponse:
    """Generate Instagram caption with best practices implementation."""
    # Implementation
```

#### GET for Retrieval
```python
@router.get(
    "/captions/{caption_id}",
    response_model=CaptionGenerationResponse,
    summary="Get Caption by ID",
    description="Retrieve a previously generated caption by its ID"
)
async def get_caption_by_id(
    caption_id: str = Path(..., description="Unique caption identifier", example="cap_123456789"),
    deps: CoreDependencies = Depends()
) -> CaptionGenerationResponse:
    """Retrieve caption by ID with caching and validation."""
    # Implementation
```

#### PUT for Updates
```python
@router.put(
    "/captions/{caption_id}",
    response_model=CaptionGenerationResponse,
    summary="Update Caption",
    description="Update an existing caption (regenerate with new parameters)"
)
async def update_caption(
    caption_id: str = Path(..., description="Caption ID to update"),
    request: CaptionGenerationRequest = ...,
    deps: CoreDependencies = Depends()
) -> CaptionGenerationResponse:
    """Update caption by regenerating with new parameters."""
    # Implementation
```

#### DELETE for Removal
```python
@router.delete(
    "/captions/{caption_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Caption",
    description="Delete a caption by ID"
)
async def delete_caption(
    caption_id: str = Path(..., description="Caption ID to delete"),
    deps: CoreDependencies = Depends()
) -> None:
    """Delete caption by ID."""
    # Implementation
```

### 2. Query Parameters and Pagination

```python
@router.get(
    "/captions",
    response_model=List[CaptionGenerationResponse],
    summary="List User Captions",
    description="Retrieve paginated list of user's generated captions"
)
async def list_user_captions(
    skip: int = Query(default=0, ge=0, description="Number of records to skip", example=0),
    limit: int = Query(default=10, ge=1, le=100, description="Number of records to return", example=10),
    style: Optional[str] = Query(default=None, description="Filter by caption style", example="casual"),
    language: Optional[str] = Query(default=None, description="Filter by language", example="en"),
    deps: CoreDependencies = Depends()
) -> List[CaptionGenerationResponse]:
    """List user captions with pagination and filtering."""
    # Implementation
```

### 3. Request Headers and Authentication

```python
async def generate_caption(
    request: CaptionGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID"),
    user_agent: Optional[str] = Header(default=None, alias="User-Agent")
) -> CaptionGenerationResponse:
    """Generate Instagram caption with request tracking."""
    # Implementation
```

### 4. Background Tasks

```python
async def generate_caption(
    request: CaptionGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: CoreDependencies = Depends()
) -> CaptionGenerationResponse:
    """Generate caption with background analytics."""
    
    # Generate caption
    response = CaptionGenerationResponse(...)
    
    # Add background task for analytics
    background_tasks.add_task(
        log_caption_generation_analytics,
        user_id=deps.user["id"],
        request=request,
        response=response,
        processing_time=processing_time
    )
    
    return response
```

### 5. Comprehensive Documentation

```python
@router.post(
    "/captions/generate",
    response_model=CaptionGenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Instagram Caption",
    description="""
    Generate a high-quality Instagram caption based on content description.
    
    **Features:**
    - Multiple caption styles and tones
    - Customizable hashtag count
    - Multi-language support
    - Emoji integration
    - Performance metrics
    
    **Rate Limits:**
    - 100 requests per minute per user
    - 1000 requests per hour per user
    """,
    response_description="Successfully generated caption with metadata",
    openapi_extra={
        "examples": {
            "casual_caption": {
                "summary": "Casual Caption Example",
                "description": "Generate a casual, friendly caption",
                "value": {
                    "content_description": "Beautiful sunset over mountains with golden light",
                    "style": "casual",
                    "tone": "friendly",
                    "hashtag_count": 15,
                    "language": "en",
                    "include_emoji": True
                }
            }
        }
    }
)
```

## Middleware Best Practices

### 1. Request ID Tracking

```python
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add unique request ID to all requests."""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response
```

### 2. Comprehensive Logging

```python
class LoggingMiddleware(BaseHTTPMiddleware):
    """Comprehensive logging middleware."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            await self._log_response(request, response, request_id, processing_time)
            return response
        except Exception as e:
            processing_time = time.time() - start_time
            await self._log_error(request, e, request_id, processing_time)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log request details"""
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_length": request.headers.get("content-length"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Request: {json.dumps(log_data)}")
```

### 3. Performance Monitoring

```python
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware."""
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log slow requests
            if processing_time > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected - "
                    f"Method: {request.method}, "
                    f"URL: {request.url}, "
                    f"Time: {processing_time:.3f}s"
                )
            
            # Add performance headers
            response.headers["X-Processing-Time"] = str(round(processing_time, 3))
            
            return response
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request failed - Time: {processing_time:.3f}s, Error: {str(e)}")
            raise
```

### 4. Security Headers

```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response
```

### 5. Rate Limiting

```python
class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 100, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_counts: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        user_id = self._get_user_id(request)
        
        if not self._check_rate_limit(user_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        response = await call_next(request)
        self._update_rate_limit(user_id)
        return response
```

### 6. Middleware Stack Configuration

```python
def create_middleware_stack() -> list:
    """Create middleware stack following FastAPI best practices."""
    
    return [
        # Security first
        TrustedHostMiddleware(allowed_hosts=["*"]),
        
        # Request tracking
        RequestIDMiddleware,
        
        # Monitoring and logging
        LoggingMiddleware,
        PerformanceMonitoringMiddleware,
        
        # Protection
        RateLimitingMiddleware,
        
        # Security headers
        SecurityHeadersMiddleware,
        
        # Performance
        CacheControlMiddleware,
        
        # Error handling last
        ErrorHandlingMiddleware,
    ]
```

## Security Best Practices

### 1. Input Validation
- Use Pydantic models for all input validation
- Implement custom validators for business logic
- Sanitize user inputs
- Use type hints for all parameters

### 2. Authentication and Authorization
- Implement proper authentication middleware
- Use JWT tokens or API keys
- Validate permissions for each endpoint
- Log authentication events

### 3. Rate Limiting
- Implement per-user rate limiting
- Use sliding window algorithm
- Return proper 429 status codes
- Include retry-after headers

### 4. Security Headers
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection
- Referrer-Policy

## Performance Best Practices

### 1. Async Operations
- Use async/await for all I/O operations
- Implement connection pooling
- Use background tasks for non-critical operations
- Optimize database queries

### 2. Caching
- Implement smart caching strategies
- Use ETags for cache validation
- Set appropriate cache headers
- Monitor cache hit rates

### 3. Response Optimization
- Use GZip compression
- Implement pagination for large datasets
- Optimize JSON serialization
- Use computed fields for derived data

### 4. Monitoring
- Track response times
- Monitor error rates
- Log performance metrics
- Set up alerts for slow requests

## Testing Best Practices

### 1. Unit Testing
```python
@pytest.mark.asyncio
async def test_generate_caption():
    """Test caption generation endpoint."""
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    response = client.post(
        "/api/v14/captions/generate",
        json={
            "content_description": "Beautiful sunset over mountains",
            "style": "casual",
            "tone": "friendly"
        },
        headers={"Authorization": "Bearer test_token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert "caption" in data
    assert "hashtags" in data
```

### 2. Integration Testing
```python
@pytest.mark.asyncio
async def test_caption_generation_integration():
    """Test complete caption generation flow."""
    # Test with real dependencies
    # Mock external services
    # Verify end-to-end functionality
```

### 3. Performance Testing
```python
def test_caption_generation_performance():
    """Test caption generation performance."""
    # Measure response times
    # Test under load
    # Verify rate limiting
```

## Documentation Best Practices

### 1. OpenAPI Documentation
- Use comprehensive descriptions
- Provide examples for all endpoints
- Document error responses
- Include rate limit information

### 2. Code Documentation
- Use docstrings for all functions
- Document complex business logic
- Include type hints
- Provide usage examples

### 3. API Documentation
- Create comprehensive API guides
- Include authentication examples
- Document rate limits
- Provide SDK examples

## Conclusion

Following these FastAPI best practices ensures:

1. **Maintainability**: Clean, well-structured code that's easy to maintain
2. **Security**: Proper validation, authentication, and security measures
3. **Performance**: Optimized operations with caching and monitoring
4. **Reliability**: Comprehensive error handling and testing
5. **Usability**: Clear documentation and consistent API design

These practices align with the official FastAPI documentation and industry standards, providing a solid foundation for building production-ready APIs. 