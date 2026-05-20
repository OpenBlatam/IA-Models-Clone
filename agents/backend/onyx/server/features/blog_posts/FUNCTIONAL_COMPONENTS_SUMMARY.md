# 🚀 Functional FastAPI Components Summary

## Overview

This implementation demonstrates a **functional approach** to FastAPI development using **Pydantic models** for validation and **pure functions** for business logic. The approach emphasizes:

- ✅ **Type Safety** - Comprehensive type hints and validation
- ✅ **Immutability** - Predictable data structures
- ✅ **Composability** - Reusable function components
- ✅ **Testability** - Easy to unit test and mock
- ✅ **Performance** - Efficient caching and processing
- ✅ **Maintainability** - Clear separation of concerns

## 🎯 Key Components

### 1. Pydantic Models for Input Validation

#### TextAnalysisRequest
```python
class TextAnalysisRequest(BaseModel):
    text_content: constr(min_length=1, max_length=10000)
    analysis_type: AnalysisTypeEnum
    optimization_tier: OptimizationTierEnum = OptimizationTierEnum.STANDARD
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text_content')
    def validate_text_content(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Text content cannot be empty')
        return cleaned
```

**Benefits:**
- **Automatic validation** with length constraints
- **Enum type safety** for analysis types
- **Custom validators** for business logic
- **JSON schema generation** for API docs

#### BatchAnalysisRequest
```python
class BatchAnalysisRequest(BaseModel):
    batch_name: constr(min_length=1, max_length=200)
    texts: List[constr(min_length=1, max_length=10000)] = Field(
        ..., min_items=1, max_items=1000
    )
    analysis_type: AnalysisTypeEnum
    priority: conint(ge=1, le=10) = Field(default=5)
```

**Benefits:**
- **List validation** with size limits
- **Nested validation** for text content
- **Range validation** for priority scores
- **Batch processing** support

### 2. Response Models

#### AnalysisResponse
```python
class AnalysisResponse(BaseModel):
    id: int = Field(description="Analysis ID")
    text_content: str = Field(description="Analyzed text content")
    analysis_type: AnalysisTypeEnum = Field(description="Analysis type")
    status: AnalysisStatusEnum = Field(description="Analysis status")
    sentiment_score: Optional[float] = Field(description="Sentiment score")
    processing_time_ms: Optional[float] = Field(description="Processing time")
    created_at: datetime = Field(description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True  # Enable ORM model conversion
```

**Benefits:**
- **ORM integration** with SQLAlchemy models
- **Comprehensive metadata** for debugging
- **Timestamp tracking** for audit trails
- **Optional fields** for incomplete analyses

#### PaginatedResponse
```python
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    size: int = Field(description="Page size")
    pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Has next page")
    has_prev: bool = Field(description="Has previous page")
    
    @classmethod
    def create(cls, items: List[T], total: int, page: int, size: int) -> 'PaginatedResponse[T]':
        pages = (total + size - 1) // size
        return cls(
            items=items, total=total, page=page, size=size,
            pages=pages, has_next=page < pages, has_prev=page > 1
        )
```

**Benefits:**
- **Generic typing** for type safety
- **Factory method** for easy creation
- **Pagination metadata** for UI navigation
- **Calculated fields** for convenience

### 3. Pure Functions

#### Validation Functions
```python
def validate_text_content(text: str) -> ValidationResult:
    """Pure function to validate text content."""
    errors = []
    warnings = []
    
    if not text:
        errors.append("Text content cannot be empty")
        return ValidationResult(is_valid=False, errors=errors)
    
    cleaned = text.strip()
    if not cleaned:
        errors.append("Text content cannot be whitespace only")
        return ValidationResult(is_valid=False, errors=errors)
    
    if len(cleaned) > 10000:
        errors.append("Text content too long (max 10000 characters)")
        return ValidationResult(is_valid=False, errors=errors)
    
    if len(cleaned) < 10:
        warnings.append("Text content is very short, analysis may be less accurate")
    
    return ValidationResult(is_valid=True, errors=errors, warnings=warnings)
```

**Benefits:**
- **No side effects** - predictable behavior
- **Immutable return** - safe to use anywhere
- **Comprehensive validation** - multiple checks
- **Warning system** - non-blocking issues

#### Business Logic Functions
```python
def calculate_processing_priority(
    optimization_tier: OptimizationTierEnum,
    text_length: int,
    analysis_type: AnalysisTypeEnum
) -> int:
    """Pure function to calculate processing priority."""
    base_priority = 5
    
    tier_multipliers = {
        OptimizationTierEnum.BASIC: 0.8,
        OptimizationTierEnum.STANDARD: 1.0,
        OptimizationTierEnum.ADVANCED: 1.2,
        OptimizationTierEnum.ULTRA: 1.5
    }
    
    length_factor = min(text_length / 1000, 2.0)
    
    complexity_factors = {
        AnalysisTypeEnum.SENTIMENT: 1.0,
        AnalysisTypeEnum.QUALITY: 1.2,
        AnalysisTypeEnum.EMOTION: 1.3,
        AnalysisTypeEnum.LANGUAGE: 0.8,
        AnalysisTypeEnum.KEYWORDS: 1.1,
        AnalysisTypeEnum.READABILITY: 1.0,
        AnalysisTypeEnum.ENTITIES: 1.4,
        AnalysisTypeEnum.TOPICS: 1.5
    }
    
    priority = int(
        base_priority * 
        tier_multipliers[optimization_tier] * 
        length_factor * 
        complexity_factors[analysis_type]
    )
    
    return max(1, min(10, priority))
```

**Benefits:**
- **Deterministic output** - same input always produces same output
- **Configurable logic** - easy to adjust multipliers
- **Type safety** - enum-based parameters
- **Bounded output** - always returns 1-10

### 4. Functional Decorators

#### Error Handling Decorator
```python
def with_error_handling(error_handler: Optional[Callable[[Exception], Dict[str, Any]]] = None):
    """Functional decorator for error handling."""
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[ProcessingResult[R]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ProcessingResult[R]:
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ProcessingResult(
                    success=True,
                    data=result,
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if error_handler:
                    error_data = error_handler(e)
                    return ProcessingResult(
                        success=False,
                        error=str(e),
                        processing_time_ms=processing_time,
                        metadata=error_data
                    )
                
                return ProcessingResult(
                    success=False,
                    error=str(e),
                    processing_time_ms=processing_time
                )
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_error_handling()
async def create_analysis_service(
    request: TextAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    # Function implementation
    pass
```

**Benefits:**
- **Automatic error wrapping** - consistent error handling
- **Performance tracking** - built-in timing
- **Custom error handlers** - flexible error processing
- **Type safety** - preserves function signatures

#### Validation Decorator
```python
def with_validation(validator_func: Callable[[Any], ValidationResult]):
    """Functional decorator for input validation."""
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[ProcessingResult[R]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ProcessingResult[R]:
            if args:
                validation_result = validator_func(args[0])
                if not validation_result.is_valid:
                    return ProcessingResult(
                        success=False,
                        error="Validation failed",
                        metadata={"validation_errors": validation_result.errors}
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_validation(validate_text_content)
async def process_text(text: str) -> str:
    # Process validated text
    pass
```

**Benefits:**
- **Separation of concerns** - validation separate from logic
- **Reusable validation** - same validator for multiple functions
- **Early failure** - fail fast on invalid input
- **Detailed error reporting** - specific validation errors

#### Caching Decorator
```python
def with_caching(cache_key_func: Callable[..., str], ttl_seconds: int = 3600):
    """Functional decorator for caching."""
    cache = {}  # In production, use Redis or similar
    
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            cache_key = cache_key_func(*args, **kwargs)
            
            # Check cache
            if cache_key in cache:
                cached_data, timestamp = cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                    return cached_data
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, datetime.now())
            
            return result
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_caching(generate_cache_key, ttl_seconds=1800)
async def analyze_text(text: str, analysis_type: AnalysisTypeEnum) -> AnalysisResult:
    # Expensive analysis operation
    pass
```

**Benefits:**
- **Performance optimization** - avoid repeated expensive operations
- **Configurable TTL** - flexible cache expiration
- **Custom key generation** - flexible caching strategies
- **Transparent caching** - no changes to function logic

### 5. Service Layer

#### Pure Service Functions
```python
@with_logging("analysis_service")
@with_error_handling()
async def create_analysis_service(
    request: TextAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Pure function to create analysis."""
    # Validate input
    validation_result = validate_text_content(request.text_content)
    if not validation_result.is_valid:
        raise ValidationError("Invalid text content")
    
    # Calculate processing priority
    priority = calculate_processing_priority(
        request.optimization_tier,
        len(request.text_content),
        request.analysis_type
    )
    
    # Create analysis data
    analysis_data = TextAnalysisCreate(
        text_content=request.text_content,
        analysis_type=request.analysis_type,
        optimization_tier=request.optimization_tier
    )
    
    # Save to database
    analysis = await db_manager.create_text_analysis(analysis_data)
    
    # Transform to response
    return transform_analysis_to_response(analysis)
```

**Benefits:**
- **Composition of pure functions** - building blocks approach
- **Clear separation** - validation, calculation, persistence, transformation
- **Error handling** - automatic error wrapping
- **Logging** - automatic performance and error tracking

### 6. API Handlers

#### Functional API Handlers
```python
async def create_analysis_handler(
    request: TextAnalysisRequest,
    db_manager: SQLAlchemy2Manager
) -> AnalysisResponse:
    """Functional API handler for creating analysis."""
    result = await create_analysis_service(request, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error
        )
    
    return result.data
```

**Benefits:**
- **Thin handlers** - minimal logic in API layer
- **Error translation** - convert service errors to HTTP responses
- **Type safety** - Pydantic models throughout
- **Consistent patterns** - same structure for all handlers

#### Dependency Injection
```python
async def list_analyses_handler(
    pagination: PaginationRequest = Depends(),
    filters: AnalysisFilterRequest = Depends(),
    db_manager: SQLAlchemy2Manager = Depends()
) -> PaginatedResponse[AnalysisResponse]:
    """Functional API handler for listing analyses."""
    result = await list_analyses_service(pagination, filters, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )
    
    return result.data
```

**Benefits:**
- **Automatic validation** - Pydantic models as dependencies
- **Clean separation** - business logic in service layer
- **Reusable dependencies** - same patterns across endpoints
- **Type safety** - compile-time validation

## 🧪 Testing Strategy

### Pydantic Model Testing
```python
class TestTextAnalysisRequest:
    def test_valid_request(self, sample_text_analysis_request):
        assert sample_text_analysis_request.text_content == "This is a positive text for sentiment analysis."
        assert sample_text_analysis_request.analysis_type == AnalysisTypeEnum.SENTIMENT
    
    def test_empty_text_content(self):
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            TextAnalysisRequest(
                text_content="",
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_text_content_cleaning(self):
        request = TextAnalysisRequest(
            text_content="  Test text with spaces  ",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        assert request.text_content == "Test text with spaces"
```

### Pure Function Testing
```python
class TestValidateTextContent:
    def test_valid_text(self):
        result = validate_text_content("This is a valid text.")
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_text(self):
        result = validate_text_content("")
        assert result.is_valid is False
        assert "Text content cannot be empty" in result.errors[0]
    
    def test_short_text_warning(self):
        result = validate_text_content("Short")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "very short" in result.warnings[0]
```

### Service Function Testing
```python
class TestCreateAnalysisService:
    @pytest.mark.asyncio
    async def test_successful_creation(self, sample_text_analysis_request, mock_db_manager, sample_analysis_model):
        mock_db_manager.create_text_analysis.return_value = sample_analysis_model
        
        result = await create_analysis_service(sample_text_analysis_request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        mock_db_manager.create_text_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_error(self, mock_db_manager):
        invalid_request = TextAnalysisRequest(
            text_content="",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        result = await create_analysis_service(invalid_request, mock_db_manager)
        
        assert result.success is False
        assert "Invalid text content" in result.error
```

## 🚀 Best Practices

### 1. Use Pure Functions
- **No side effects** - functions should not modify external state
- **Deterministic output** - same input always produces same output
- **Easy to test** - no complex setup or teardown needed
- **Composable** - functions can be combined safely

### 2. Leverage Pydantic Models
- **Automatic validation** - built-in type checking and constraints
- **Serialization** - automatic JSON conversion
- **Documentation** - automatic OpenAPI schema generation
- **Type safety** - compile-time error detection

### 3. Use Functional Decorators
- **Cross-cutting concerns** - logging, error handling, caching
- **Reusable patterns** - same decorators across functions
- **Separation of concerns** - business logic separate from infrastructure
- **Composable** - multiple decorators can be combined

### 4. Immutable Data Structures
- **Thread safety** - safe for concurrent access
- **Predictable state** - no unexpected mutations
- **Debugging** - easier to trace data flow
- **Performance** - can be optimized by the runtime

### 5. Type Safety Throughout
- **Type hints** - everywhere possible
- **Generic types** - for reusable components
- **Enum usage** - for constrained values
- **Pydantic validation** - for runtime type checking

### 6. Error Handling
- **Result types** - use `ProcessingResult` for consistent error handling
- **Early returns** - fail fast on validation errors
- **Detailed errors** - provide specific error messages
- **Error translation** - convert service errors to HTTP responses

### 7. Testing Strategy
- **Unit tests** - for pure functions
- **Integration tests** - for service functions
- **Mock dependencies** - for external services
- **Property-based testing** - for complex validation logic

## 📊 Performance Considerations

### 1. Caching Strategy
```python
@with_caching(generate_cache_key, ttl_seconds=1800)
async def expensive_analysis(text: str) -> AnalysisResult:
    # Expensive operation
    pass
```

### 2. Batch Processing
```python
async def process_batch(texts: List[str]) -> List[AnalysisResult]:
    # Use asyncio.gather for concurrent processing
    tasks = [analyze_text(text) for text in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Lazy Evaluation
```python
def process_large_dataset(items: Iterator[str]) -> Iterator[AnalysisResult]:
    for item in items:
        yield analyze_item(item)
```

## 🔧 Configuration

### Environment-based Configuration
```python
class AppConfig(BaseModel):
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(..., env="REDIS_URL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
```

## 📈 Monitoring and Observability

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "Analysis completed",
    analysis_id=analysis.id,
    processing_time_ms=processing_time,
    analysis_type=analysis.analysis_type,
    success=True
)
```

### Health Checks
```python
async def health_check() -> HealthResponse:
    """Functional health check."""
    start_time = datetime.now()
    
    # Check database
    db_healthy = await check_database_health()
    
    # Check cache
    cache_healthy = await check_cache_health()
    
    # Calculate uptime
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if db_healthy and cache_healthy else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        database={"status": "healthy" if db_healthy else "unhealthy"},
        performance={"response_time_ms": uptime * 1000}
    )
```

## 🎯 Benefits Summary

### 1. **Predictable Behavior**
- Pure functions with no side effects
- Immutable data structures
- Deterministic outputs

### 2. **Type Safety**
- Comprehensive type hints
- Pydantic validation
- Compile-time error detection

### 3. **Testability**
- Easy to unit test pure functions
- Simple mocking of dependencies
- Clear input/output contracts

### 4. **Composability**
- Functions can be combined safely
- Reusable components
- Building block approach

### 5. **Performance**
- Efficient caching strategies
- Batch processing capabilities
- Lazy evaluation support

### 6. **Maintainability**
- Clear separation of concerns
- Consistent patterns
- Easy to understand and modify

### 7. **Scalability**
- Functional patterns scale well
- Stateless operations
- Concurrent processing support

## 🎯 Use Cases

This functional approach is particularly well-suited for:

### 1. **Data Processing Pipelines**
- Text analysis and NLP
- ETL operations
- Data transformation workflows

### 2. **API Services**
- RESTful APIs
- GraphQL services
- Microservices

### 3. **High-Performance Applications**
- Real-time processing
- Batch operations
- Caching-intensive applications

### 4. **Complex Business Logic**
- Multi-step workflows
- Decision trees
- Rule engines

## 🚀 Getting Started

1. **Install Dependencies**
```bash
pip install fastapi pydantic sqlalchemy structlog pytest
```

2. **Import Components**
```python
from functional_fastapi_components import (
    TextAnalysisRequest, AnalysisResponse,
    create_analysis_service, validate_text_content
)
```

3. **Use in Your Application**
```python
# Create request
request = TextAnalysisRequest(
    text_content="Sample text for analysis",
    analysis_type=AnalysisTypeEnum.SENTIMENT
)

# Process with service
result = await create_analysis_service(request, db_manager)

# Handle response
if result.success:
    return result.data
else:
    raise HTTPException(status_code=400, detail=result.error)
```

## 📚 Additional Resources

- **Functional Programming Guide** - `FUNCTIONAL_FASTAPI_GUIDE.md`
- **Test Suite** - `test_functional_components.py`
- **SQLAlchemy 2.0 Integration** - `sqlalchemy_2_implementation.py`
- **FastAPI Best Practices** - `FASTAPI_BEST_PRACTICES.md`

---

This functional approach provides a robust, maintainable, and scalable foundation for building FastAPI applications with comprehensive type safety and validation. 