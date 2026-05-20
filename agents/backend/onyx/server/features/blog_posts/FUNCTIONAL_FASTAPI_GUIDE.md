# 🚀 Functional FastAPI Components with Pydantic Models

## Overview

This guide demonstrates a **functional approach** to building FastAPI applications using:
- **Pure functions** for business logic
- **Pydantic models** for validation and serialization
- **Immutable data structures** for predictable state
- **Declarative patterns** for clear intent
- **Composition over inheritance** for flexibility
- **Type-safe operations** throughout the stack

## 🎯 Key Principles

### 1. Pure Functions
Functions that have no side effects and always return the same output for the same input.

```python
def calculate_processing_priority(
    optimization_tier: OptimizationTierEnum,
    text_length: int,
    analysis_type: AnalysisTypeEnum
) -> int:
    """Pure function to calculate processing priority."""
    # No side effects, deterministic output
    return priority_score
```

### 2. Immutable Data Structures
Using `@dataclass(frozen=True)` and Pydantic models for immutable data.

```python
@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### 3. Declarative Validation
Pydantic models with built-in validation and serialization.

```python
class TextAnalysisRequest(BaseModel):
    text_content: constr(min_length=1, max_length=10000) = Field(
        ..., 
        description="Text content to analyze"
    )
    analysis_type: AnalysisTypeEnum = Field(..., description="Type of analysis")
    
    @validator('text_content')
    def validate_text_content(cls, v: str) -> str:
        """Validate and clean text content."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError('Text content cannot be empty')
        return cleaned
```

## 📋 Pydantic Models

### Input Validation Models

#### TextAnalysisRequest
```python
class TextAnalysisRequest(BaseModel):
    text_content: constr(min_length=1, max_length=10000)
    analysis_type: AnalysisTypeEnum
    optimization_tier: OptimizationTierEnum = OptimizationTierEnum.STANDARD
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text_content": "This is a positive text for sentiment analysis.",
                "analysis_type": "sentiment",
                "optimization_tier": "standard",
                "metadata": {"source": "user_input", "priority": "high"}
            }
        }
    )
```

**Features:**
- **Constrained strings** with length validation
- **Enum validation** for type safety
- **Default values** with sensible defaults
- **JSON schema examples** for API documentation
- **Custom validators** for business logic

#### BatchAnalysisRequest
```python
class BatchAnalysisRequest(BaseModel):
    batch_name: constr(min_length=1, max_length=200)
    texts: List[constr(min_length=1, max_length=10000)] = Field(
        ..., min_items=1, max_items=1000
    )
    analysis_type: AnalysisTypeEnum
    optimization_tier: OptimizationTierEnum = OptimizationTierEnum.STANDARD
    priority: conint(ge=1, le=10) = Field(default=5)
```

**Features:**
- **List validation** with min/max items
- **Nested validation** for text content
- **Range validation** for priority (1-10)
- **Batch processing** support

#### AnalysisUpdateRequest
```python
class AnalysisUpdateRequest(BaseModel):
    status: Optional[AnalysisStatusEnum] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: Optional[float] = Field(None, ge=0.0)
    model_used: Optional[constr(max_length=100)] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[constr(max_length=1000)] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Features:**
- **Optional fields** for partial updates
- **Range validation** for scores
- **Cross-field validation** with `@root_validator`
- **Error handling** support

### Response Models

#### AnalysisResponse
```python
class AnalysisResponse(BaseModel):
    id: int = Field(description="Analysis ID")
    text_content: str = Field(description="Analyzed text content")
    analysis_type: AnalysisTypeEnum = Field(description="Analysis type")
    status: AnalysisStatusEnum = Field(description="Analysis status")
    sentiment_score: Optional[float] = Field(description="Sentiment score")
    quality_score: Optional[float] = Field(description="Quality score")
    processing_time_ms: Optional[float] = Field(description="Processing time")
    model_used: Optional[str] = Field(description="Model used")
    confidence_score: Optional[float] = Field(description="Confidence score")
    optimization_tier: OptimizationTierEnum = Field(description="Optimization tier")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    processed_at: Optional[datetime] = Field(description="Processing completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        from_attributes = True  # Enable ORM model conversion
```

**Features:**
- **ORM integration** with `from_attributes = True`
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
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        size: int
    ) -> 'PaginatedResponse[T]':
        """Factory method for creating paginated responses."""
        pages = (total + size - 1) // size  # Ceiling division
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )
```

**Features:**
- **Generic typing** for type safety
- **Factory method** for easy creation
- **Pagination metadata** for UI navigation
- **Calculated fields** for convenience

## 🔧 Pure Functions

### Validation Functions
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

### Business Logic Functions
```python
def calculate_processing_priority(
    optimization_tier: OptimizationTierEnum,
    text_length: int,
    analysis_type: AnalysisTypeEnum
) -> int:
    """Pure function to calculate processing priority."""
    base_priority = 5
    
    # Adjust based on optimization tier
    tier_multipliers = {
        OptimizationTierEnum.BASIC: 0.8,
        OptimizationTierEnum.STANDARD: 1.0,
        OptimizationTierEnum.ADVANCED: 1.2,
        OptimizationTierEnum.ULTRA: 1.5
    }
    
    # Adjust based on text length
    length_factor = min(text_length / 1000, 2.0)
    
    # Adjust based on analysis type complexity
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

### Utility Functions
```python
def calculate_batch_progress(
    completed_count: int,
    error_count: int,
    total_count: int
) -> Dict[str, Any]:
    """Pure function to calculate batch progress."""
    if total_count == 0:
        return {
            "progress_percentage": 0.0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "remaining_count": 0
        }
    
    progress_percentage = ((completed_count + error_count) / total_count) * 100
    success_rate = (completed_count / total_count) * 100
    error_rate = (error_count / total_count) * 100
    remaining_count = total_count - completed_count - error_count
    
    return {
        "progress_percentage": round(progress_percentage, 2),
        "success_rate": round(success_rate, 2),
        "error_rate": round(error_rate, 2),
        "remaining_count": remaining_count
    }
```

**Benefits:**
- **Edge case handling** - zero division protection
- **Rounded output** - consistent decimal places
- **Multiple metrics** - comprehensive progress tracking
- **Immutable return** - safe for concurrent access

## 🎭 Functional Decorators

### Error Handling Decorator
```python
def with_error_handling(
    error_handler: Optional[Callable[[Exception], Dict[str, Any]]] = None
):
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

### Validation Decorator
```python
def with_validation(validator_func: Callable[[Any], ValidationResult]):
    """Functional decorator for input validation."""
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[ProcessingResult[R]]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ProcessingResult[R]:
            # Validate first argument (assuming it's the input)
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

### Caching Decorator
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

### Logging Decorator
```python
def with_logging(logger_name: str):
    """Functional decorator for logging."""
    logger = structlog.get_logger(logger_name)
    
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            start_time = datetime.now()
            
            logger.info(
                "Function started",
                function=func.__name__,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            try:
                result = await func(*args, **kwargs)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "Function completed",
                    function=func.__name__,
                    processing_time=processing_time,
                    success=True
                )
                
                return result
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    processing_time=processing_time,
                    error=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_logging("analysis_service")
async def create_analysis_service(request: TextAnalysisRequest) -> AnalysisResponse:
    # Service implementation
    pass
```

**Benefits:**
- **Automatic logging** - no manual log statements needed
- **Performance tracking** - built-in timing
- **Error logging** - automatic error capture
- **Structured logging** - consistent log format

## 🏗️ Service Layer

### Pure Service Functions
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

### Data Transformation
```python
def transform_analysis_to_response(
    analysis: Any,
    include_metadata: bool = True
) -> AnalysisResponse:
    """Pure function to transform database model to response."""
    response_data = {
        "id": analysis.id,
        "text_content": analysis.text_content,
        "analysis_type": analysis.analysis_type,
        "status": analysis.status,
        "sentiment_score": analysis.sentiment_score,
        "quality_score": analysis.quality_score,
        "processing_time_ms": analysis.processing_time_ms,
        "model_used": analysis.model_used,
        "confidence_score": analysis.confidence_score,
        "optimization_tier": analysis.optimization_tier,
        "created_at": analysis.created_at,
        "updated_at": analysis.updated_at,
        "processed_at": analysis.processed_at
    }
    
    if include_metadata:
        response_data["metadata"] = analysis.metadata or {}
    
    return AnalysisResponse(**response_data)
```

**Benefits:**
- **Pure transformation** - no side effects
- **Optional metadata** - flexible output
- **Type safety** - Pydantic validation
- **Reusable** - same transformer for multiple endpoints

## 🎯 API Handlers

### Functional API Handlers
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

### Dependency Injection
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

## 🔄 Background Tasks

### Functional Background Processing
```python
async def process_batch_texts_functional(
    batch_id: int,
    texts: List[str],
    analysis_type: AnalysisTypeEnum,
    db_manager: SQLAlchemy2Manager
):
    """Functional background task for processing batch texts."""
    logger = structlog.get_logger("batch_processor")
    
    completed_count = 0
    error_count = 0
    
    logger.info(f"Starting batch processing for batch {batch_id}")
    
    for i, text in enumerate(texts):
        try:
            # Create analysis request
            analysis_request = TextAnalysisRequest(
                text_content=text,
                analysis_type=analysis_type
            )
            
            # Process analysis
            analysis_result = await create_analysis_service(analysis_request, db_manager)
            
            if analysis_result.success:
                # Update with simulated results
                update_request = AnalysisUpdateRequest(
                    status=AnalysisStatusEnum.COMPLETED,
                    sentiment_score=0.5 + (i * 0.1),
                    processing_time_ms=100.0 + i,
                    model_used="functional-model"
                )
                
                await update_analysis_service(analysis_result.data.id, update_request, db_manager)
                completed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing text {i+1} in batch {batch_id}: {e}")
    
    # Update batch progress
    await db_manager.update_batch_progress(batch_id, completed_count, error_count)
    
    logger.info(
        f"Completed batch {batch_id}: {completed_count} successful, {error_count} errors"
    )
```

**Benefits:**
- **Composition of services** - reuse existing service functions
- **Error isolation** - individual text failures don't stop batch
- **Progress tracking** - real-time batch progress updates
- **Comprehensive logging** - detailed processing information

## 🧪 Testing

### Pydantic Model Testing
```python
class TestTextAnalysisRequest:
    """Test TextAnalysisRequest model."""
    
    def test_valid_request(self, sample_text_analysis_request):
        """Test valid request creation."""
        assert sample_text_analysis_request.text_content == "This is a positive text for sentiment analysis."
        assert sample_text_analysis_request.analysis_type == AnalysisTypeEnum.SENTIMENT
        assert sample_text_analysis_request.optimization_tier == OptimizationTierEnum.STANDARD
        assert sample_text_analysis_request.metadata == {"source": "test", "priority": "high"}
    
    def test_empty_text_content(self):
        """Test empty text content validation."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            TextAnalysisRequest(
                text_content="",
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_text_content_cleaning(self):
        """Test text content cleaning."""
        request = TextAnalysisRequest(
            text_content="  Test text with spaces  ",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        assert request.text_content == "Test text with spaces"
```

### Pure Function Testing
```python
class TestValidateTextContent:
    """Test validate_text_content function."""
    
    def test_valid_text(self):
        """Test valid text validation."""
        result = validate_text_content("This is a valid text.")
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_text(self):
        """Test empty text validation."""
        result = validate_text_content("")
        assert result.is_valid is False
        assert "Text content cannot be empty" in result.errors[0]
    
    def test_short_text_warning(self):
        """Test short text warning."""
        result = validate_text_content("Short")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "very short" in result.warnings[0]
```

### Service Function Testing
```python
class TestCreateAnalysisService:
    """Test create_analysis_service function."""
    
    @pytest.mark.asyncio
    async def test_successful_creation(self, sample_text_analysis_request, mock_db_manager, sample_analysis_model):
        """Test successful analysis creation."""
        mock_db_manager.create_text_analysis.return_value = sample_analysis_model
        
        result = await create_analysis_service(sample_text_analysis_request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        mock_db_manager.create_text_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_error(self, mock_db_manager):
        """Test validation error handling."""
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
# Use functional caching decorator
@with_caching(generate_cache_key, ttl_seconds=1800)
async def expensive_analysis(text: str) -> AnalysisResult:
    # Expensive operation
    pass
```

### 2. Batch Processing
```python
# Process multiple items efficiently
async def process_batch(texts: List[str]) -> List[AnalysisResult]:
    # Use asyncio.gather for concurrent processing
    tasks = [analyze_text(text) for text in texts]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Lazy Evaluation
```python
# Use generators for large datasets
def process_large_dataset(items: Iterator[str]) -> Iterator[AnalysisResult]:
    for item in items:
        yield analyze_item(item)
```

### 4. Memory Management
```python
# Use context managers for resource cleanup
async def process_with_resources():
    async with get_db_connection() as conn:
        async with get_cache_connection() as cache:
            return await process_data(conn, cache)
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

### Feature Flags
```python
class FeatureFlags(BaseModel):
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_batch_processing: bool = Field(default=True, env="ENABLE_BATCH_PROCESSING")
    enable_advanced_validation: bool = Field(default=False, env="ENABLE_ADVANCED_VALIDATION")
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
```

## 📈 Monitoring and Observability

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "Analysis completed",
    analysis_id=analysis.id,
    processing_time_ms=processing_time,
    analysis_type=analysis.analysis_type,
    success=True
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

analysis_counter = Counter('analysis_total', 'Total analyses', ['type', 'status'])
processing_time = Histogram('analysis_processing_seconds', 'Analysis processing time')

# In your service function
@processing_time.time()
async def create_analysis_service(request: TextAnalysisRequest) -> AnalysisResponse:
    # ... processing logic ...
    analysis_counter.labels(type=request.analysis_type, status='completed').inc()
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

## 🎯 Conclusion

The functional approach to FastAPI with Pydantic models provides:

1. **Predictable behavior** - pure functions and immutable data
2. **Type safety** - comprehensive type checking throughout
3. **Testability** - easy to unit test and mock
4. **Composability** - functions can be combined safely
5. **Performance** - efficient caching and batch processing
6. **Maintainability** - clear separation of concerns
7. **Scalability** - functional patterns scale well

This approach is particularly well-suited for:
- **Data processing pipelines** - where data flows through multiple transformations
- **API services** - where validation and serialization are critical
- **Microservices** - where clear interfaces and error handling are essential
- **High-performance applications** - where caching and optimization are important

By following these patterns, you can build robust, maintainable, and scalable FastAPI applications that are easy to test, debug, and extend. 