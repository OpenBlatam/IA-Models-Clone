# 🎯 Declarative Route Definitions with Clear Return Type Annotations

## Overview

This guide demonstrates a **declarative approach** to FastAPI route definitions using:
- **Clear return type annotations** for type safety
- **Functional route handlers** for predictable behavior
- **Pydantic models** for request/response validation
- **Dependency injection** for clean architecture
- **Type-safe operations** throughout the stack
- **OpenAPI documentation** generation

## 🎯 Key Principles

### 1. Clear Return Type Annotations
Every route handler has explicit return type annotations for maximum type safety.

```python
async def create_analysis(
    request: TextAnalysisRequest = Body(...),
    db_manager: DBManager = Depends(get_db_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> AnalysisDetailResponse:
    """Create a new text analysis."""
    # Implementation with guaranteed return type
    return AnalysisDetailResponse(...)
```

### 2. Declarative Route Definitions
Routes are defined using class-based organization with clear structure.

```python
class AnalysisRoutes:
    """Declarative route definitions for analysis endpoints."""
    
    def __init__(self, router: APIRouter):
        self.router = router
        self._register_routes()
    
    def _register_routes(self):
        """Register all analysis routes."""
        # Route definitions with decorators and type annotations
```

### 3. Consistent Response Wrapping
All responses are wrapped in consistent format with proper error handling.

```python
class RouteResponse(BaseModel):
    """Base response wrapper for consistent API responses."""
    success: bool = Field(description="Operation success status")
    data: Optional[Dict[str, Any]] = Field(description="Response data")
    message: Optional[str] = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(description="Request ID for tracking")
```

## 📋 Type Definitions and Annotations

### Type Aliases for Better Readability
```python
# Type aliases for better readability
AnalysisID = Annotated[int, Path(description="Analysis ID", ge=1)]
BatchID = Annotated[int, Path(description="Batch ID", ge=1)]
PageNumber = Annotated[int, Query(description="Page number", ge=1, default=1)]
PageSize = Annotated[int, Query(description="Page size", ge=1, le=100, default=20)]
OrderBy = Annotated[str, Query(description="Field to order by", default="created_at")]
OrderDesc = Annotated[bool, Query(description="Descending order", default=True)]

# Dependency type annotations
DBManager = Annotated[Any, Depends()]
AuthToken = Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())]
BackgroundTaskManager = Annotated[BackgroundTasks, Depends()]
```

**Benefits:**
- **Type safety** - Compile-time validation
- **Documentation** - Self-documenting code
- **IDE support** - Better autocomplete and error detection
- **Consistency** - Standardized parameter definitions

### Response Type Hierarchy
```python
class RouteResponse(BaseModel):
    """Base response wrapper for consistent API responses."""
    success: bool = Field(description="Operation success status")
    data: Optional[Dict[str, Any]] = Field(description="Response data")
    message: Optional[str] = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(description="Request ID for tracking")

class AnalysisDetailResponse(RouteResponse):
    """Response type for analysis detail endpoints."""
    data: Optional[AnalysisResponse] = Field(description="Analysis details")

class AnalysisListResponse(RouteResponse):
    """Response type for analysis listing endpoints."""
    data: Optional[PaginatedResponse[AnalysisResponse]] = Field(description="Paginated analysis results")

class BatchDetailResponse(RouteResponse):
    """Response type for batch analysis endpoints."""
    data: Optional[BatchAnalysisResponse] = Field(description="Batch analysis details")

class HealthCheckResponse(RouteResponse):
    """Response type for health check endpoints."""
    data: Optional[HealthResponse] = Field(description="Health check results")
```

## 🔧 Route Decorators and Utilities

### Response Wrapper Decorator
```python
def with_response_wrapper(response_model: type):
    """Decorator to wrap responses in consistent format."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # If result is already a RouteResponse, return as is
                if isinstance(result, RouteResponse):
                    return result
                
                # Wrap in appropriate response type
                return response_model(
                    success=True,
                    data=result,
                    message="Operation completed successfully"
                )
                
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                # Wrap other exceptions in error response
                return ErrorDetailResponse(
                    success=False,
                    data=ErrorResponse(
                        error=str(e),
                        error_code="INTERNAL_ERROR",
                        detail="An unexpected error occurred",
                        timestamp=datetime.now()
                    ),
                    message="Operation failed"
                )
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_response_wrapper(AnalysisDetailResponse)
async def create_analysis_handler(request: TextAnalysisRequest) -> AnalysisDetailResponse:
    # Function implementation
    pass
```

**Benefits:**
- **Consistent responses** - All endpoints return same format
- **Error handling** - Automatic error wrapping
- **Type safety** - Guaranteed return types
- **Clean code** - No repetitive response wrapping

### Request Logging Decorator
```python
def with_request_logging():
    """Decorator to log request details."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = structlog.get_logger("route_handler")
            
            # Extract request info
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if request:
                logger.info(
                    "Route request started",
                    method=request.method,
                    url=str(request.url),
                    client_ip=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent")
                )
            
            try:
                result = await func(*args, **kwargs)
                logger.info("Route request completed successfully")
                return result
            except Exception as e:
                logger.error("Route request failed", error=str(e))
                raise
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_request_logging()
async def get_analysis_handler(analysis_id: int) -> AnalysisDetailResponse:
    # Function implementation with automatic logging
    pass
```

**Benefits:**
- **Automatic logging** - No manual log statements needed
- **Request tracking** - Complete request lifecycle logging
- **Error logging** - Automatic error capture
- **Performance monitoring** - Request timing information

### Performance Monitoring Decorator
```python
def with_performance_monitoring():
    """Decorator to monitor route performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = await func(*args, **kwargs)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Log performance metrics
                logger = structlog.get_logger("performance")
                logger.info(
                    "Route performance",
                    function=func.__name__,
                    processing_time_seconds=processing_time,
                    success=True
                )
                
                return result
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logger = structlog.get_logger("performance")
                logger.error(
                    "Route performance",
                    function=func.__name__,
                    processing_time_seconds=processing_time,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator
```

**Usage:**
```python
@with_performance_monitoring()
async def update_analysis_handler(analysis_id: int, update_data: dict) -> AnalysisDetailResponse:
    # Function implementation with performance monitoring
    pass
```

**Benefits:**
- **Performance tracking** - Automatic timing measurement
- **Success/failure metrics** - Track operation success rates
- **Function-level monitoring** - Identify slow operations
- **Error correlation** - Link errors to performance issues

## 🏗️ Dependency Functions

### Database Manager Dependency
```python
async def get_db_manager() -> Any:
    """Dependency to get database manager."""
    # In real implementation, return actual database manager
    from .sqlalchemy_2_implementation import SQLAlchemy2Manager
    return SQLAlchemy2Manager()
```

### Authentication Dependency
```python
async def get_current_user(auth_token: AuthToken) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    # In real implementation, validate token and return user
    return {
        "id": 1,
        "email": "user@example.com",
        "role": "user"
    }
```

### Request ID Dependency
```python
async def get_request_id(request: Request) -> str:
    """Dependency to get or generate request ID."""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())
    return request_id
```

## 🎯 Declarative Route Definitions

### Analysis Routes
```python
class AnalysisRoutes:
    """Declarative route definitions for analysis endpoints."""
    
    def __init__(self, router: APIRouter):
        self.router = router
        self._register_routes()
    
    def _register_routes(self):
        """Register all analysis routes."""
        
        # POST /analyses - Create new analysis
        @self.router.post(
            "/analyses",
            response_model=AnalysisDetailResponse,
            status_code=status.HTTP_201_CREATED,
            summary="Create Text Analysis",
            description="Create a new text analysis with the specified parameters",
            response_description="Analysis created successfully",
            tags=["Analysis"]
        )
        @with_response_wrapper(AnalysisDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def create_analysis(
            request: TextAnalysisRequest = Body(
                ...,
                description="Analysis request parameters",
                example={
                    "text_content": "This is a positive text for sentiment analysis.",
                    "analysis_type": "sentiment",
                    "optimization_tier": "standard",
                    "metadata": {"source": "user_input", "priority": "high"}
                }
            ),
            background_tasks: BackgroundTaskManager = Depends(),
            db_manager: DBManager = Depends(get_db_manager),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ) -> AnalysisDetailResponse:
            """
            Create a new text analysis.
            
            This endpoint creates a new text analysis with the specified parameters.
            The analysis will be processed asynchronously and results will be available
            once processing is complete.
            
            Args:
                request: Analysis request parameters
                background_tasks: Background task manager
                db_manager: Database manager
                current_user: Current authenticated user
                request_id: Request ID for tracking
                
            Returns:
                AnalysisDetailResponse: Created analysis details
                
            Raises:
                HTTPException: If validation fails or database error occurs
            """
            
            # Validate request
            validation_result = validate_text_content(request.text_content)
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Validation failed",
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings
                    }
                )
            
            # Create analysis
            result = await create_analysis_service(request, db_manager)
            
            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.error
                )
            
            # Add background task for processing
            background_tasks.add_task(
                self._process_analysis_background,
                result.data.id,
                request.text_content,
                request.analysis_type,
                db_manager
            )
            
            return AnalysisDetailResponse(
                success=True,
                data=result.data,
                message="Analysis created successfully",
                request_id=request_id
            )
```

**Features:**
- **Clear return type** - `-> AnalysisDetailResponse`
- **Comprehensive documentation** - Detailed docstring with Args/Returns/Raises
- **Input validation** - Pydantic model validation
- **Error handling** - Proper HTTP exception handling
- **Background processing** - Async task management
- **Request tracking** - Request ID for tracing

### GET Analysis by ID
```python
@self.router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Analysis by ID",
    description="Retrieve analysis details by ID",
    response_description="Analysis details",
    tags=["Analysis"]
)
@with_response_wrapper(AnalysisDetailResponse)
@with_request_logging()
@with_performance_monitoring()
async def get_analysis(
    analysis_id: AnalysisID,
    db_manager: DBManager = Depends(get_db_manager),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_id: str = Depends(get_request_id)
) -> AnalysisDetailResponse:
    """
    Get analysis by ID.
    
    Retrieve detailed information about a specific analysis by its ID.
    
    Args:
        analysis_id: Analysis ID
        db_manager: Database manager
        current_user: Current authenticated user
        request_id: Request ID for tracking
        
    Returns:
        AnalysisDetailResponse: Analysis details
        
    Raises:
        HTTPException: If analysis not found
    """
    
    result = await get_analysis_service(analysis_id, db_manager)
    
    if not result.success:
        if "not found" in result.error.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )
    
    return AnalysisDetailResponse(
        success=True,
        data=result.data,
        message="Analysis retrieved successfully",
        request_id=request_id
    )
```

### List Analyses with Filtering
```python
@self.router.get(
    "/analyses",
    response_model=AnalysisListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Analyses",
    description="Retrieve paginated list of analyses with optional filtering",
    response_description="Paginated analysis results",
    tags=["Analysis"]
)
@with_response_wrapper(AnalysisListResponse)
@with_request_logging()
@with_performance_monitoring()
async def list_analyses(
    page: PageNumber,
    size: PageSize,
    order_by: OrderBy,
    order_desc: OrderDesc,
    analysis_type: Optional[AnalysisTypeEnum] = Query(
        None, description="Filter by analysis type"
    ),
    status: Optional[AnalysisStatusEnum] = Query(
        None, description="Filter by status"
    ),
    optimization_tier: Optional[OptimizationTierEnum] = Query(
        None, description="Filter by optimization tier"
    ),
    date_from: Optional[datetime] = Query(
        None, description="Filter from date (ISO format)"
    ),
    date_to: Optional[datetime] = Query(
        None, description="Filter to date (ISO format)"
    ),
    min_sentiment_score: Optional[float] = Query(
        None, ge=-1.0, le=1.0, description="Minimum sentiment score"
    ),
    max_sentiment_score: Optional[float] = Query(
        None, ge=-1.0, le=1.0, description="Maximum sentiment score"
    ),
    db_manager: DBManager = Depends(get_db_manager),
    current_user: Dict[str, Any] = Depends(get_current_user),
    request_id: str = Depends(get_request_id)
) -> AnalysisListResponse:
    """
    List analyses with pagination and filtering.
    
    Retrieve a paginated list of analyses with optional filtering by various criteria.
    
    Args:
        page: Page number (1-based)
        size: Page size (1-100)
        order_by: Field to order by
        order_desc: Descending order flag
        analysis_type: Filter by analysis type
        status: Filter by status
        optimization_tier: Filter by optimization tier
        date_from: Filter from date
        date_to: Filter to date
        min_sentiment_score: Minimum sentiment score
        max_sentiment_score: Maximum sentiment score
        db_manager: Database manager
        current_user: Current authenticated user
        request_id: Request ID for tracking
        
    Returns:
        AnalysisListResponse: Paginated analysis results
    """
    
    # Build pagination request
    pagination = PaginationRequest(
        page=page,
        size=size,
        order_by=order_by,
        order_desc=order_desc
    )
    
    # Build filter request
    filters = AnalysisFilterRequest(
        analysis_type=analysis_type,
        status=status,
        optimization_tier=optimization_tier,
        date_from=date_from,
        date_to=date_to,
        min_sentiment_score=min_sentiment_score,
        max_sentiment_score=max_sentiment_score
    )
    
    result = await list_analyses_service(pagination, filters, db_manager)
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error
        )
    
    return AnalysisListResponse(
        success=True,
        data=result.data,
        message=f"Retrieved {len(result.data.items)} analyses",
        request_id=request_id
    )
```

## 🔄 Background Task Processing

### Analysis Background Processing
```python
async def _process_analysis_background(
    self,
    analysis_id: int,
    text_content: str,
    analysis_type: AnalysisTypeEnum,
    db_manager: Any
):
    """Background task to process analysis."""
    logger = structlog.get_logger("background_processor")
    
    try:
        logger.info(f"Starting background processing for analysis {analysis_id}")
        
        # Simulate processing
        await asyncio.sleep(2)
        
        # Update analysis with results
        update_request = AnalysisUpdateRequest(
            status=AnalysisStatusEnum.COMPLETED,
            sentiment_score=0.5,
            processing_time_ms=2000.0,
            model_used="background-processor"
        )
        
        await update_analysis_service(analysis_id, update_request, db_manager)
        
        logger.info(f"Completed background processing for analysis {analysis_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for analysis {analysis_id}: {e}")
        
        # Update analysis with error
        error_request = AnalysisUpdateRequest(
            status=AnalysisStatusEnum.ERROR,
            error_message=str(e)
        )
        
        await update_analysis_service(analysis_id, error_request, db_manager)
```

## 🏥 Health Check Routes

### Basic Health Check
```python
@self.router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check system health and status",
    response_description="System health status",
    tags=["Health"]
)
@with_response_wrapper(HealthCheckResponse)
@with_request_logging()
@with_performance_monitoring()
async def health_check(
    db_manager: DBManager = Depends(get_db_manager),
    request_id: str = Depends(get_request_id)
) -> HealthCheckResponse:
    """
    Health check endpoint.
    
    Check the health status of the system and its dependencies.
    
    Args:
        db_manager: Database manager
        request_id: Request ID for tracking
        
    Returns:
        HealthCheckResponse: System health status
    """
    
    start_time = datetime.now()
    
    # Check database health
    db_healthy = True  # In real implementation, check actual database
    try:
        # await db_manager.ping()
        pass
    except Exception:
        db_healthy = False
    
    # Calculate uptime
    uptime = (datetime.now() - start_time).total_seconds()
    
    health_data = HealthResponse(
        status="healthy" if db_healthy else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        database={"status": "healthy" if db_healthy else "unhealthy"},
        performance={"response_time_ms": uptime * 1000}
    )
    
    return HealthCheckResponse(
        success=db_healthy,
        data=health_data,
        message="Health check completed",
        request_id=request_id
    )
```

### Detailed Health Check
```python
@self.router.get(
    "/health/detailed",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed Health Check",
    description="Detailed system health check with performance metrics",
    response_description="Detailed health status",
    tags=["Health"]
)
@with_response_wrapper(HealthCheckResponse)
@with_request_logging()
@with_performance_monitoring()
async def detailed_health_check(
    db_manager: DBManager = Depends(get_db_manager),
    request_id: str = Depends(get_request_id)
) -> HealthCheckResponse:
    """
    Detailed health check endpoint.
    
    Perform a detailed health check with performance metrics and dependency status.
    
    Args:
        db_manager: Database manager
        request_id: Request ID for tracking
        
    Returns:
        HealthCheckResponse: Detailed health status
    """
    
    start_time = datetime.now()
    
    # Perform detailed checks
    checks = {
        "database": await self._check_database(db_manager),
        "cache": await self._check_cache(),
        "memory": await self._check_memory(),
        "disk": await self._check_disk()
    }
    
    # Determine overall health
    all_healthy = all(check["status"] == "healthy" for check in checks.values())
    
    # Calculate performance metrics
    processing_time = (datetime.now() - start_time).total_seconds()
    
    health_data = HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=processing_time,
        database=checks["database"],
        performance={
            "response_time_ms": processing_time * 1000,
            "checks_performed": len(checks)
        },
        errors=[check["error"] for check in checks.values() if check["error"]]
    )
    
    return HealthCheckResponse(
        success=all_healthy,
        data=health_data,
        message="Detailed health check completed",
        request_id=request_id
    )
```

## 🏗️ Application Factory

### Router Factory
```python
def create_analysis_router() -> APIRouter:
    """Create analysis router with all routes."""
    router = APIRouter(prefix="/api/v1", tags=["Analysis"])
    
    # Register route classes
    AnalysisRoutes(router)
    BatchRoutes(router)
    HealthRoutes(router)
    
    return router
```

### Application Factory
```python
def create_app() -> FastAPI:
    """Create FastAPI application with declarative routes."""
    app = FastAPI(
        title="Text Analysis API",
        description="Functional FastAPI application with declarative routes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add routers
    analysis_router = create_analysis_router()
    app.include_router(analysis_router)
    
    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger = structlog.get_logger("exception_handler")
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if app.debug else "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return app
```

## 🧪 Testing Strategy

### Route Handler Testing
```python
@pytest.mark.asyncio
async def test_create_analysis_success(router, sample_text_analysis_request, 
                                     mock_db_manager, sample_analysis_response):
    """Test successful analysis creation."""
    # Mock service response
    with patch('declarative_routes.create_analysis_service') as mock_service:
        mock_service.return_value.success = True
        mock_service.return_value.data = sample_analysis_response
        
        # Mock dependencies
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                        mock_bg_tasks.return_value.add_task = MagicMock()
                        
                        # Call the route handler
                        result = await router._register_routes.create_analysis(
                            request=sample_text_analysis_request,
                            background_tasks=mock_bg_tasks.return_value,
                            db_manager=mock_db_manager,
                            current_user={"id": 1},
                            request_id="test-id"
                        )
                        
                        assert isinstance(result, AnalysisDetailResponse)
                        assert result.success is True
                        assert result.data == sample_analysis_response
                        assert result.message == "Analysis created successfully"
```

### Decorator Testing
```python
@pytest.mark.asyncio
async def test_response_wrapper_success():
    """Test successful response wrapping."""
    @with_response_wrapper(AnalysisDetailResponse)
    async def test_func():
        return {"id": 1, "text": "test"}
    
    result = await test_func()
    
    assert isinstance(result, AnalysisDetailResponse)
    assert result.success is True
    assert result.data == {"id": 1, "text": "test"}
    assert result.message == "Operation completed successfully"
```

### Integration Testing
```python
def test_openapi_schema_generation(fastapi_app):
    """Test OpenAPI schema generation."""
    client = TestClient(fastapi_app)
    
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    
    # Check basic schema structure
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    
    # Check API info
    assert schema["info"]["title"] == "Text Analysis API"
    assert schema["info"]["version"] == "1.0.0"
    
    # Check paths exist
    paths = schema["paths"]
    assert "/api/v1/analyses" in paths
    assert "/api/v1/batches" in paths
    assert "/api/v1/health" in paths
```

## 🚀 Best Practices

### 1. Clear Return Type Annotations
- **Always specify return types** for route handlers
- **Use specific response models** instead of generic types
- **Document complex return types** with detailed docstrings
- **Validate return types** in tests

### 2. Declarative Route Organization
- **Group related routes** in classes
- **Use consistent naming** for route handlers
- **Separate concerns** with different route classes
- **Register routes systematically** in factory functions

### 3. Consistent Response Formatting
- **Use response wrapper decorators** for consistency
- **Include success/error indicators** in all responses
- **Add request tracking** with request IDs
- **Provide meaningful messages** for users

### 4. Comprehensive Error Handling
- **Handle specific exceptions** with appropriate HTTP status codes
- **Provide detailed error messages** for debugging
- **Log errors appropriately** for monitoring
- **Use global exception handlers** for unhandled errors

### 5. Performance Monitoring
- **Monitor route performance** with decorators
- **Track request timing** for optimization
- **Log performance metrics** for analysis
- **Set up alerts** for slow operations

### 6. Type Safety Throughout
- **Use type aliases** for common parameter types
- **Validate input parameters** with Pydantic models
- **Ensure return type consistency** across handlers
- **Test type annotations** in CI/CD

### 7. Documentation and Testing
- **Generate comprehensive OpenAPI docs** automatically
- **Write detailed docstrings** for all handlers
- **Test all route scenarios** including error cases
- **Validate API documentation** with integration tests

## 📊 Performance Considerations

### 1. Decorator Performance
```python
# Use decorators efficiently
@with_response_wrapper(AnalysisDetailResponse)
@with_request_logging()
@with_performance_monitoring()
async def route_handler():
    # Handler implementation
    pass
```

### 2. Dependency Injection
```python
# Cache dependencies when possible
@lru_cache()
def get_cached_dependency():
    return expensive_operation()

async def route_handler(
    cached_dep: Any = Depends(get_cached_dependency)
):
    # Use cached dependency
    pass
```

### 3. Background Task Management
```python
# Use background tasks for long-running operations
async def route_handler(
    background_tasks: BackgroundTasks = Depends()
):
    # Add background task
    background_tasks.add_task(long_running_operation)
    
    # Return immediately
    return immediate_response
```

## 🔧 Configuration

### Environment-based Configuration
```python
class AppConfig(BaseModel):
    title: str = Field(default="Text Analysis API", env="API_TITLE")
    version: str = Field(default="1.0.0", env="API_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    
    class Config:
        env_file = ".env"
```

### Route Configuration
```python
class RouteConfig(BaseModel):
    prefix: str = Field(default="/api/v1", env="API_PREFIX")
    tags: List[str] = Field(default=["Analysis"], env="API_TAGS")
    response_model: str = Field(default="RouteResponse", env="RESPONSE_MODEL")
    enable_logging: bool = Field(default=True, env="ENABLE_LOGGING")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
```

## 📈 Monitoring and Observability

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "Route request completed",
    method="POST",
    path="/api/v1/analyses",
    status_code=201,
    processing_time_ms=150.5,
    user_id=123
)
```

### Performance Metrics
```python
from prometheus_client import Counter, Histogram

route_requests = Counter('route_requests_total', 'Total route requests', ['method', 'path', 'status'])
route_duration = Histogram('route_duration_seconds', 'Route processing time', ['method', 'path'])

# In route handler
@route_duration.time()
async def route_handler():
    # Handler implementation
    route_requests.labels(method="POST", path="/analyses", status="201").inc()
```

### Health Checks
```python
async def health_check() -> HealthCheckResponse:
    """Comprehensive health check."""
    checks = {
        "database": await check_database_health(),
        "cache": await check_cache_health(),
        "memory": await check_memory_usage(),
        "disk": await check_disk_usage()
    }
    
    all_healthy = all(check["status"] == "healthy" for check in checks.values())
    
    return HealthCheckResponse(
        success=all_healthy,
        data=HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database=checks["database"],
            performance=checks["performance"]
        )
    )
```

## 🎯 Benefits Summary

### 1. **Type Safety**
- Clear return type annotations
- Compile-time error detection
- IDE support and autocomplete
- Runtime validation with Pydantic

### 2. **Maintainability**
- Declarative route organization
- Consistent patterns across endpoints
- Clear separation of concerns
- Easy to understand and modify

### 3. **Testability**
- Easy to unit test route handlers
- Mockable dependencies
- Clear input/output contracts
- Comprehensive test coverage

### 4. **Documentation**
- Automatic OpenAPI generation
- Self-documenting code
- Detailed docstrings
- Interactive API documentation

### 5. **Performance**
- Efficient decorator usage
- Background task processing
- Performance monitoring
- Optimized dependency injection

### 6. **Observability**
- Comprehensive logging
- Performance metrics
- Health checks
- Error tracking

### 7. **Scalability**
- Modular route organization
- Reusable components
- Background processing
- Stateless operations

## 🎯 Use Cases

This declarative approach is particularly well-suited for:

### 1. **RESTful APIs**
- CRUD operations
- Resource management
- Pagination and filtering
- Bulk operations

### 2. **Microservices**
- Service-to-service communication
- Health checks and monitoring
- Error handling and resilience
- Performance optimization

### 3. **Data Processing APIs**
- Batch operations
- Background task management
- Progress tracking
- Result aggregation

### 4. **Monitoring and Observability**
- Health checks
- Metrics collection
- Performance monitoring
- Error tracking

## 🚀 Getting Started

1. **Install Dependencies**
```bash
pip install fastapi pydantic sqlalchemy structlog pytest httpx
```

2. **Import Components**
```python
from declarative_routes import (
    create_app, AnalysisRoutes, BatchRoutes, HealthRoutes,
    AnalysisDetailResponse, BatchDetailResponse, HealthCheckResponse
)
```

3. **Create Application**
```python
# Create application
app = create_app()

# Start server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

4. **Test Endpoints**
```python
# Test with client
from fastapi.testclient import TestClient

client = TestClient(app)

# Create analysis
response = client.post("/api/v1/analyses", json={
    "text_content": "Test text",
    "analysis_type": "sentiment"
})

assert response.status_code == 201
assert response.json()["success"] is True
```

## 📚 Additional Resources

- **Functional Components Guide** - `FUNCTIONAL_FASTAPI_GUIDE.md`
- **Test Suite** - `test_declarative_routes.py`
- **SQLAlchemy 2.0 Integration** - `sqlalchemy_2_implementation.py`
- **FastAPI Best Practices** - `FASTAPI_BEST_PRACTICES.md`

---

This declarative approach provides a robust, maintainable, and scalable foundation for building FastAPI applications with comprehensive type safety, clear return type annotations, and excellent developer experience. 