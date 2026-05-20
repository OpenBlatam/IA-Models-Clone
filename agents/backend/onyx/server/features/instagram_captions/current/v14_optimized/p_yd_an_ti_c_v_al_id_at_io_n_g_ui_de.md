# 🎯 Pydantic Validation Guide - Instagram Captions API v14.0

## 📋 Overview

This guide documents the comprehensive Pydantic validation system implemented in v14.0, featuring consistent input/output validation, response schemas, and advanced validation patterns using Pydantic's BaseModel.

## 🏗️ **Schema Architecture**

### **1. Base Configuration**

All schemas inherit from a common configuration that ensures consistency:

```python
class BaseSchemaConfig:
    """Base configuration for all schemas"""
    model_config = ConfigDict(
        str_strip_whitespace=True,      # Auto-strip whitespace
        validate_assignment=True,       # Validate on assignment
        use_enum_values=True,          # Use enum values
        extra="forbid",                # Reject extra fields
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        populate_by_name=True,         # Allow field aliases
        validate_default=True          # Validate default values
    )
```

### **2. Enumeration Types**

Comprehensive enums for type safety and validation:

```python
class CaptionStyle(str, Enum):
    """Caption writing styles"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    INSPIRATIONAL = "inspirational"
    PLAYFUL = "playful"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"

class AudienceType(str, Enum):
    """Target audience types"""
    GENERAL = "general"
    BUSINESS = "business"
    MILLENNIALS = "millennials"
    GEN_Z = "gen_z"
    CREATORS = "creators"
    LIFESTYLE = "lifestyle"
```

## 📥 **Request Schemas**

### **1. CaptionGenerationRequest**

Comprehensive request validation with advanced features:

```python
class CaptionGenerationRequest(BaseModel):
    """Comprehensive caption generation request schema"""
    
    # Core content with validation
    content_description: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Content description for caption generation",
        examples=["Beautiful sunset at the beach with golden colors"]
    )
    
    # Enum-based style selection
    style: CaptionStyle = Field(
        default=CaptionStyle.CASUAL,
        description="Caption writing style"
    )
    
    # Performance settings
    optimization_level: OptimizationLevel = Field(
        default=OptimizationLevel.BALANCED,
        description="Performance optimization level"
    )
    
    # Advanced validation with custom validators
    @field_validator('content_description')
    @classmethod
    def validate_content_description(cls, v: str) -> str:
        """Validate and sanitize content description"""
        if not v or not v.strip():
            raise ValueError("Content description cannot be empty")
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Potentially harmful content detected: {pattern}")
        
        return v.strip()
    
    # Computed fields for caching
    @computed_field
    @property
    def request_hash(self) -> str:
        """Generate unique hash for request caching"""
        import hashlib
        content = f"{self.content_description}:{self.style}:{self.audience}:{self.hashtag_count}"
        return hashlib.md5(content.encode()).hexdigest()
```

### **2. BatchCaptionRequest**

Batch processing with duplicate detection:

```python
class BatchCaptionRequest(BaseModel):
    """Batch caption generation request schema"""
    
    requests: List[CaptionGenerationRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of caption generation requests"
    )
    
    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v: List[CaptionGenerationRequest]) -> List[CaptionGenerationRequest]:
        """Validate batch requests"""
        if not v:
            raise ValueError("Batch must contain at least one request")
        
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        
        # Check for duplicate requests
        request_hashes = [req.request_hash for req in v]
        if len(request_hashes) != len(set(request_hashes)):
            raise ValueError("Duplicate requests detected in batch")
        
        return v
```

### **3. CaptionOptimizationRequest**

Optimization-specific validation:

```python
class CaptionOptimizationRequest(BaseModel):
    """Caption optimization request schema"""
    
    caption: str = Field(
        ...,
        min_length=5,
        max_length=2200,
        description="Caption text to optimize"
    )
    
    enhancement_level: Literal["light", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Level of optimization to apply"
    )
    
    @field_validator('caption')
    @classmethod
    def validate_caption(cls, v: str) -> str:
        """Validate caption text"""
        if not v or not v.strip():
            raise ValueError("Caption cannot be empty")
        
        if len(v) > 2200:
            raise ValueError("Caption too long (max 2200 characters)")
        
        return v.strip()
```

## 📤 **Response Schemas**

### **1. CaptionGenerationResponse**

Comprehensive response with quality metrics:

```python
class CaptionGenerationResponse(BaseModel):
    """Comprehensive caption generation response schema"""
    
    # Core response data
    request_id: str = Field(description="Unique request identifier")
    caption: str = Field(description="Generated Instagram caption")
    hashtags: List[str] = Field(description="Generated hashtags")
    
    # Quality metrics with validation
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Caption quality score (0-100)"
    )
    
    engagement_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Predicted engagement score"
    )
    
    # Performance metrics
    processing_time: float = Field(
        ge=0.0,
        description="Processing time in seconds"
    )
    
    cache_hit: bool = Field(description="Whether response was served from cache")
    
    # Computed fields
    @computed_field
    @property
    def total_hashtags(self) -> int:
        """Get total number of hashtags"""
        return len(self.hashtags)
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if response was optimized"""
        return self.optimization_level != OptimizationLevel.ULTRA_FAST
```

### **2. BatchCaptionResponse**

Batch processing results with error tracking:

```python
class BatchCaptionResponse(BaseModel):
    """Batch caption generation response schema"""
    
    batch_id: str = Field(description="Unique batch identifier")
    total_requests: int = Field(ge=1, description="Total number of requests")
    successful_requests: int = Field(ge=0, description="Number of successful requests")
    failed_requests: int = Field(ge=0, description="Number of failed requests")
    
    responses: List[CaptionGenerationResponse] = Field(description="Generated responses")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    
    # Cross-field validation
    @field_validator('successful_requests')
    @classmethod
    def validate_successful_requests(cls, v: int, info) -> int:
        """Validate successful requests count"""
        total_requests = info.data.get('total_requests', 0)
        if v > total_requests:
            raise ValueError("Successful requests cannot exceed total requests")
        return v
    
    @field_validator('failed_requests')
    @classmethod
    def validate_failed_requests(cls, v: int, info) -> int:
        """Validate failed requests count"""
        total_requests = info.data.get('total_requests', 0)
        successful_requests = info.data.get('successful_requests', 0)
        if v != (total_requests - successful_requests):
            raise ValueError("Failed requests count must equal total minus successful")
        return v
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
```

## 🚨 **Error Schemas**

### **1. Standardized Error Responses**

```python
class ErrorDetail(BaseModel):
    """Detailed error information schema"""
    
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")

class APIErrorResponse(BaseModel):
    """Standardized API error response schema"""
    
    error: bool = Field(default=True, description="Error flag")
    error_code: str = Field(description="Unique error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")
    status_code: int = Field(description="HTTP status code")
```

## 📊 **Monitoring Schemas**

### **1. Performance Metrics**

```python
class PerformanceMetrics(BaseModel):
    """Performance metrics schema"""
    
    total_requests: int = Field(ge=0, description="Total requests processed")
    successful_requests: int = Field(ge=0, description="Successful requests")
    failed_requests: int = Field(ge=0, description="Failed requests")
    cache_hits: int = Field(ge=0, description="Number of cache hits")
    cache_misses: int = Field(ge=0, description="Number of cache misses")
    
    average_response_time: float = Field(ge=0.0, description="Average response time")
    p95_response_time: float = Field(ge=0.0, description="95th percentile response time")
    p99_response_time: float = Field(ge=0.0, description="99th percentile response time")
    
    error_rate: float = Field(ge=0.0, le=100.0, description="Error rate percentage")
    cache_hit_rate: float = Field(ge=0.0, le=100.0, description="Cache hit rate percentage")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
```

### **2. Health Check Response**

```python
class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(description="System health status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime: float = Field(ge=0.0, description="System uptime in seconds")
    
    components: Dict[str, Literal["healthy", "degraded", "unhealthy"]] = Field(
        description="Individual component health status"
    )
    
    performance: PerformanceMetrics = Field(description="Current performance metrics")
```

## 🔧 **Utility Functions**

### **1. Error Response Creation**

```python
def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None
) -> APIErrorResponse:
    """Create standardized error response"""
    return APIErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        path=path,
        method=method,
        status_code=status_code
    )
```

### **2. Request Validation**

```python
def validate_request_data(data: Dict[str, Any], schema_class: type) -> BaseModel:
    """Validate request data against schema"""
    try:
        return schema_class(**data)
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")
```

### **3. Input Sanitization**

```python
def sanitize_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input data"""
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove null bytes and normalize whitespace
            sanitized_value = value.replace('\x00', '').strip()
            # Basic XSS protection
            sanitized_value = re.sub(r'<script[^>]*>.*?</script>', '', sanitized_value, flags=re.IGNORECASE)
            sanitized_value = re.sub(r'javascript:', '', sanitized_value, flags=re.IGNORECASE)
            sanitized[key] = sanitized_value
        else:
            sanitized[key] = value
    
    return sanitized
```

## 🚀 **FastAPI Integration**

### **1. Endpoint Validation**

```python
@app.post("/api/v14/generate", response_model=CaptionGenerationResponse)
async def generate_caption(
    request: CaptionGenerationRequest,
    api_key: str = Depends(validate_api_key_dependency),
    request_id: str = Depends(get_request_id),
    rate_limit: None = Depends(rate_limit_check)
):
    """Generate Instagram caption with comprehensive validation"""
    start_time = time.time()
    
    try:
        # Additional validation using our validation engine
        is_valid, validation_errors = await validation_engine.validate_request(
            request.model_dump(), request_id
        )
        
        if not is_valid:
            raise ValidationError(
                message="Request validation failed",
                details={"validation_errors": validation_errors},
                request_id=request_id,
                path="/api/v14/generate",
                method="POST"
            )
        
        # Generate caption using optimized engine
        response = await optimized_engine.generate_caption(request, request_id)
        
        return response
        
    except Exception as e:
        # Handle errors with structured responses
        error_response = create_error_response(
            error_code="GENERATION_ERROR",
            message="Caption generation failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__},
            request_id=request_id,
            path="/api/v14/generate",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )
```

### **2. Exception Handlers**

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    request_id = str(uuid.uuid4())
    
    error_response = create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"validation_errors": exc.errors()},
        request_id=request_id,
        path=str(request.url.path),
        method=request.method
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )
```

## 📝 **Validation Examples**

### **1. Valid Request**

```json
{
  "content_description": "Beautiful sunset at the beach with golden colors",
  "style": "inspirational",
  "audience": "lifestyle",
  "hashtag_count": 15,
  "optimization_level": "balanced",
  "language": "en",
  "client_id": "test-client-123"
}
```

### **2. Invalid Request (Validation Error)**

```json
{
  "content_description": "abc",
  "style": "invalid_style",
  "hashtag_count": 100
}
```

**Response:**
```json
{
  "error": true,
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": {
    "validation_errors": [
      {
        "field": "content_description",
        "message": "String should have at least 5 characters"
      },
      {
        "field": "style",
        "message": "Input should be 'casual', 'professional', 'inspirational', 'playful', 'educational' or 'promotional'"
      },
      {
        "field": "hashtag_count",
        "message": "Input should be less than or equal to 50"
      }
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req-123",
  "path": "/api/v14/generate",
  "method": "POST",
  "status_code": 422
}
```

### **3. Successful Response**

```json
{
  "request_id": "req-456",
  "caption": "Golden hour magic ✨ There's something truly special about watching the sun paint the sky in hues of amber and gold. Nature's daily masterpiece never fails to take my breath away. 🌅 #GoldenHour #SunsetVibes #NaturePhotography #BeachLife #GoldenMoment #SunsetMagic #CoastalVibes #Photography #NaturalBeauty #SunsetLover #BeachSunset #GoldenSky #MomentOfPeace #SunsetGoals #BeachVibes",
  "hashtags": ["#GoldenHour", "#SunsetVibes", "#NaturePhotography", "#BeachLife", "#GoldenMoment", "#SunsetMagic", "#CoastalVibes", "#Photography", "#NaturalBeauty", "#SunsetLover", "#BeachSunset", "#GoldenSky", "#MomentOfPeace", "#SunsetGoals", "#BeachVibes"],
  "quality_score": 92.5,
  "engagement_score": 88.3,
  "readability_score": 94.1,
  "processing_time": 0.245,
  "cache_hit": false,
  "optimization_level": "balanced",
  "api_version": "14.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "sentiment": "positive",
  "tone": "inspirational",
  "word_count": 28,
  "character_count": 456,
  "total_hashtags": 15,
  "is_optimized": true
}
```

## 🎯 **Best Practices**

### **1. Schema Design**

- **Use Enums**: For categorical data to ensure type safety
- **Comprehensive Validation**: Include min/max lengths, patterns, and custom validators
- **Computed Fields**: Use `@computed_field` for derived values
- **Cross-field Validation**: Validate relationships between fields
- **Default Values**: Provide sensible defaults for optional fields

### **2. Error Handling**

- **Structured Errors**: Use consistent error response schemas
- **Detailed Messages**: Provide clear, actionable error messages
- **Error Codes**: Use unique error codes for categorization
- **Request Tracking**: Include request IDs for debugging

### **3. Performance**

- **Lazy Validation**: Validate only when needed
- **Caching**: Use computed fields for expensive calculations
- **Batch Processing**: Validate batch requests efficiently
- **Sanitization**: Clean input data before processing

### **4. Security**

- **Input Sanitization**: Remove harmful content
- **Field Validation**: Validate all input fields
- **Type Safety**: Use strict typing with enums
- **Extra Field Rejection**: Reject unexpected fields

## 🔍 **Testing Validation**

### **1. Unit Tests**

```python
import pytest
from pydantic import ValidationError
from ..types.schemas import CaptionGenerationRequest

def test_valid_request():
    """Test valid request validation"""
    request_data = {
        "content_description": "Beautiful sunset",
        "style": "casual",
        "hashtag_count": 15
    }
    
    request = CaptionGenerationRequest(**request_data)
    assert request.content_description == "Beautiful sunset"
    assert request.style == "casual"
    assert request.hashtag_count == 15

def test_invalid_request():
    """Test invalid request validation"""
    request_data = {
        "content_description": "abc",  # Too short
        "style": "invalid_style",      # Invalid enum
        "hashtag_count": 100           # Too high
    }
    
    with pytest.raises(ValidationError) as exc_info:
        CaptionGenerationRequest(**request_data)
    
    errors = exc_info.value.errors()
    assert len(errors) == 3

def test_harmful_content_detection():
    """Test harmful content detection"""
    request_data = {
        "content_description": "<script>alert('xss')</script>",
        "style": "casual"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        CaptionGenerationRequest(**request_data)
    
    assert "harmful content detected" in str(exc_info.value)
```

### **2. Integration Tests**

```python
async def test_api_validation(client):
    """Test API endpoint validation"""
    # Valid request
    valid_data = {
        "content_description": "Beautiful sunset",
        "style": "casual",
        "hashtag_count": 15
    }
    
    response = await client.post("/api/v14/generate", json=valid_data)
    assert response.status_code == 200
    
    # Invalid request
    invalid_data = {
        "content_description": "abc",
        "style": "invalid_style"
    }
    
    response = await client.post("/api/v14/generate", json=invalid_data)
    assert response.status_code == 422
    
    error_data = response.json()
    assert error_data["error_code"] == "VALIDATION_ERROR"
    assert "validation_errors" in error_data["details"]
```

## 📈 **Benefits**

### **1. Type Safety**
- **Compile-time Validation**: Catch errors before runtime
- **IDE Support**: Better autocomplete and error detection
- **Refactoring Safety**: Safe field renaming and restructuring

### **2. Data Integrity**
- **Input Validation**: Ensure data quality at the boundary
- **Output Consistency**: Guaranteed response structure
- **Error Handling**: Structured error responses

### **3. Developer Experience**
- **Auto-generated Documentation**: OpenAPI/Swagger docs
- **Clear Contracts**: Explicit input/output schemas
- **Testing Support**: Easy to test validation logic

### **4. Performance**
- **Efficient Validation**: Fast validation with Pydantic
- **Caching Support**: Computed fields for expensive operations
- **Batch Processing**: Optimized batch validation

## 🔮 **Future Enhancements**

### **1. Advanced Validation**
- **Custom Validators**: More sophisticated validation rules
- **Async Validation**: Database-based validation
- **Conditional Validation**: Context-dependent validation

### **2. Schema Evolution**
- **Versioning**: Backward-compatible schema changes
- **Migration Tools**: Automated schema migration
- **Deprecation Handling**: Graceful field deprecation

### **3. Performance Optimization**
- **Lazy Loading**: Load schemas on demand
- **Caching**: Cache validation results
- **Parallel Validation**: Validate multiple fields concurrently

This comprehensive Pydantic validation system ensures data integrity, provides excellent developer experience, and maintains high performance while offering robust error handling and monitoring capabilities. 