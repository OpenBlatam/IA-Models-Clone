# Structured Routes and Dependencies Guide for Video-OpusClip

## Overview

This guide provides comprehensive documentation for the structured routing system in Video-OpusClip, which organizes routes and dependencies clearly to optimize readability and maintainability. The system follows FastAPI best practices and provides a scalable architecture for AI video processing applications.

## 🏗️ Architecture Overview

### Key Principles

1. **Modular Organization**: Routes organized by business domain
2. **Clear Dependencies**: Centralized dependency injection with proper lifecycle management
3. **Consistent Patterns**: Standardized error handling and response formats
4. **Performance Monitoring**: Built-in metrics and health checks
5. **Type Safety**: Full type annotations and Pydantic validation
6. **Documentation**: Automatic OpenAPI documentation generation

### System Components

```
structured_routes.py
├── Route Organization (RouteCategory, RoutePriority, RouteConfig)
├── Base Router (BaseRouter)
├── Common Dependencies (CommonDependencies)
├── Route Handlers (VideoRouteHandlers, BatchRouteHandlers, AnalyticsRouteHandlers)
├── Router Factory (RouterFactory)
├── Route Registry (RouteRegistry)
└── Application Factory (create_structured_app)
```

## 📁 Route Organization

### Route Categories

Routes are organized into logical categories for better maintainability:

```python
class RouteCategory(str, Enum):
    AUTH = "authentication"           # User authentication and authorization
    VIDEO = "video"                   # Video processing operations
    PROCESSING = "processing"         # AI model processing
    BATCH = "batch"                   # Batch processing operations
    ANALYTICS = "analytics"           # Analytics and monitoring
    ADMIN = "admin"                   # Administrative operations
    SYSTEM = "system"                 # System-level operations
    HEALTH = "health"                 # Health checks and diagnostics
    MONITORING = "monitoring"         # Performance monitoring
    API = "api"                       # General API operations
    WEBHOOK = "webhook"               # Webhook endpoints
    FILE = "file"                     # File upload/download
    SEARCH = "search"                 # Search operations
    NOTIFICATION = "notification"     # Notification systems
    INTEGRATION = "integration"       # Third-party integrations
```

### Route Priorities

Routes are prioritized for better resource management:

```python
class RoutePriority(int, Enum):
    CRITICAL = 1      # Health checks, authentication
    HIGH = 2          # Core video processing
    NORMAL = 3        # Standard operations
    LOW = 4           # Analytics, reporting
    BACKGROUND = 5    # Batch processing, cleanup
```

### Route Configuration

Each route can be configured with detailed metadata:

```python
@dataclass
class RouteConfig:
    path: str                         # Route path
    method: str                       # HTTP method
    tags: List[str]                   # OpenAPI tags
    summary: str                      # Route summary
    description: str                  # Detailed description
    response_model: Optional[Type[BaseModel]] = None
    status_code: int = 200
    dependencies: List[Callable] = field(default_factory=list)
    deprecated: bool = False
    include_in_schema: bool = True
    category: RouteCategory = RouteCategory.API
    priority: RoutePriority = RoutePriority.NORMAL
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None
```

## 🔧 Dependency Injection System

### Common Dependencies

The `CommonDependencies` class provides centralized access to all shared resources:

```python
class CommonDependencies:
    def __init__(self, container: DependencyContainer):
        self.container = container
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials) -> Optional[Dict[str, Any]]:
        """Get current authenticated user."""
    
    async def get_db_session(self):
        """Get database session."""
    
    async def get_cache_client(self):
        """Get cache client."""
    
    async def get_video_database(self) -> AsyncVideoDatabase:
        """Get video database service."""
    
    async def get_batch_database(self) -> AsyncBatchDatabaseOperations:
        """Get batch database operations."""
    
    async def get_youtube_api(self) -> AsyncYouTubeAPI:
        """Get YouTube API service."""
    
    async def get_openai_api(self) -> AsyncOpenAIAPI:
        """Get OpenAI API service."""
    
    async def get_stability_api(self) -> AsyncStabilityAIAPI:
        """Get Stability AI API service."""
    
    async def get_elevenlabs_api(self) -> AsyncElevenLabsAPI:
        """Get ElevenLabs API service."""
```

### Dependency Usage Patterns

#### 1. Basic Dependency Injection

```python
@router.post("/videos")
async def create_video(
    video_data: VideoRequest,
    current_user: Dict[str, Any] = Depends(deps.get_current_user),
    video_db: AsyncVideoDatabase = Depends(deps.get_video_database)
) -> VideoResponse:
    # Route implementation
    pass
```

#### 2. Optional Dependencies

```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict[str, Any]]:
    if not credentials:
        return None
    # Validate and return user
```

#### 3. Service Dependencies

```python
async def get_video_database(self) -> AsyncVideoDatabase:
    """Get video database service with connection pooling."""
    db_ops = await self.container.get_db_session_dependency()()
    return AsyncVideoDatabase(db_ops)
```

## 🎯 Route Handlers

### Video Route Handlers

Handles all video processing operations:

```python
class VideoRouteHandlers:
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def create_video(self, video_data: VideoRequest, ...) -> VideoResponse:
        """Create a new video processing request."""
    
    async def get_video(self, video_id: int, ...) -> VideoResponse:
        """Get video by ID."""
    
    async def update_video_status(self, video_id: int, status: ProcessingStatus, ...) -> VideoResponse:
        """Update video processing status."""
```

### Batch Route Handlers

Handles batch processing operations:

```python
class BatchRouteHandlers:
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def create_batch_videos(self, batch_data: BatchVideoRequest, ...) -> BatchVideoResponse:
        """Create multiple video processing requests."""
```

### Analytics Route Handlers

Handles analytics and monitoring:

```python
class AnalyticsRouteHandlers:
    def __init__(self, dependencies: CommonDependencies):
        self.deps = dependencies
    
    async def get_processing_stats(self, ...) -> Dict[str, Any]:
        """Get video processing statistics."""
```

## 🏭 Router Factory

The `RouterFactory` creates organized routers with proper dependency injection:

### Video Router

```python
def create_video_router(self) -> APIRouter:
    router = APIRouter(prefix="/api/v1/videos", tags=["video"])
    
    @router.post("/", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
    async def create_video(...):
        return await self.handlers["video"].create_video(...)
    
    @router.get("/{video_id}", response_model=VideoResponse)
    async def get_video(...):
        return await self.handlers["video"].get_video(...)
    
    @router.patch("/{video_id}/status", response_model=VideoResponse)
    async def update_video_status(...):
        return await self.handlers["video"].update_video_status(...)
    
    @router.get("/", response_model=List[VideoResponse])
    async def list_videos(...):
        # Implementation with pagination and filtering
        pass
    
    return router
```

### Batch Router

```python
def create_batch_router(self) -> APIRouter:
    router = APIRouter(prefix="/api/v1/batch", tags=["batch"])
    
    @router.post("/videos", response_model=BatchVideoResponse, status_code=status.HTTP_201_CREATED)
    async def create_batch_videos(...):
        return await self.handlers["batch"].create_batch_videos(...)
    
    return router
```

### Analytics Router

```python
def create_analytics_router(self) -> APIRouter:
    router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
    
    @router.get("/stats")
    async def get_processing_stats(...):
        return await self.handlers["analytics"].get_processing_stats(...)
    
    return router
```

### Health Router

```python
def create_health_router(self) -> APIRouter:
    router = APIRouter(prefix="/api/v1/health", tags=["health"])
    
    @router.get("/", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(status="healthy", timestamp=datetime.now(), version="1.0.0")
    
    @router.get("/detailed")
    async def detailed_health_check():
        # Detailed health check with all services
        pass
    
    return router
```

## 📋 Route Registry

The `RouteRegistry` manages all routers centrally:

```python
class RouteRegistry:
    def __init__(self, app: FastAPI, container: DependencyContainer):
        self.app = app
        self.container = container
        self.dependencies = CommonDependencies(container)
        self.factory = RouterFactory(self.dependencies)
        self.routers: Dict[str, APIRouter] = {}
        self.route_configs: Dict[str, RouteConfig] = {}
    
    def register_router(self, name: str, router: APIRouter, config: Optional[RouteConfig] = None):
        """Register a router with optional configuration."""
        self.routers[name] = router
        if config:
            self.route_configs[name] = config
        self.app.include_router(router)
    
    def register_all_routers(self):
        """Register all standard routers."""
        # Video processing router
        video_router = self.factory.create_video_router()
        self.register_router("video", video_router, RouteConfig(...))
        
        # Batch processing router
        batch_router = self.factory.create_batch_router()
        self.register_router("batch", batch_router, RouteConfig(...))
        
        # Analytics router
        analytics_router = self.factory.create_analytics_router()
        self.register_router("analytics", analytics_router, RouteConfig(...))
        
        # Health check router
        health_router = self.factory.create_health_router()
        self.register_router("health", health_router, RouteConfig(...))
```

## 🚀 Application Factory

The `create_structured_app()` function creates a complete FastAPI application:

```python
def create_structured_app() -> FastAPI:
    # Create FastAPI app
    app = FastAPI(
        title="Video-OpusClip API",
        description="AI-powered video processing system with structured routes and dependencies",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add middleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Get dependency container
    container = get_dependency_container()
    
    # Create route registry
    registry = RouteRegistry(app, container)
    
    # Register all routers
    registry.register_all_routers()
    
    # Add global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(...)
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(...)
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Log request and response with timing
        pass
    
    return app
```

## 📊 API Endpoints

### Video Processing Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| POST | `/api/v1/videos/` | Create video processing request | `VideoResponse` |
| GET | `/api/v1/videos/{video_id}` | Get video by ID | `VideoResponse` |
| PATCH | `/api/v1/videos/{video_id}/status` | Update video status | `VideoResponse` |
| GET | `/api/v1/videos/` | List videos with pagination | `List[VideoResponse]` |

### Batch Processing Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| POST | `/api/v1/batch/videos` | Create batch video processing | `BatchVideoResponse` |

### Analytics Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/api/v1/analytics/stats` | Get processing statistics | `Dict[str, Any]` |

### Health Check Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/api/v1/health/` | Basic health check | `HealthResponse` |
| GET | `/api/v1/health/detailed` | Detailed health check | `Dict[str, Any]` |

## 🔍 Error Handling

### Global Exception Handlers

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )
```

### Route-Level Error Handling

```python
async def create_video(...):
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create video: {str(e)}"
        )
```

## 📈 Performance Monitoring

### Request Logging Middleware

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    # Add process time to response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
```

### Health Monitoring

```python
@router.get("/detailed")
async def detailed_health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": {
                "status": "healthy",
                "response_time": 0.001,
                "last_check": datetime.now().isoformat()
            },
            "cache": {
                "status": "healthy",
                "response_time": 0.0005,
                "last_check": datetime.now().isoformat()
            },
            "models": {
                "status": "healthy",
                "loaded_models": ["video_processor", "caption_generator"],
                "last_check": datetime.now().isoformat()
            }
        }
    }
```

## 🧪 Testing Support

### Dependency Overrides

```python
def create_test_app() -> FastAPI:
    app = create_structured_app()
    
    # Override dependencies for testing
    def override_get_current_user():
        return {"id": "test_user", "email": "test@example.com", "role": "user"}
    
    app.dependency_overrides[get_current_user] = override_get_current_user
    
    return app
```

### Mock Services

```python
class MockVideoDatabase:
    async def create_video_record(self, video_data: Dict[str, Any]) -> int:
        return 1
    
    async def get_video_by_id(self, video_id: int) -> Optional[Dict[str, Any]]:
        return {"id": video_id, "title": "Test Video", "status": "pending"}

# Use in tests
app.dependency_overrides[get_video_database] = lambda: MockVideoDatabase()
```

## 🚀 Best Practices

### 1. Route Organization

- **Group by Domain**: Organize routes by business domain (video, batch, analytics)
- **Consistent Naming**: Use consistent naming conventions for routes and handlers
- **Clear Documentation**: Provide comprehensive documentation for each route
- **Proper Status Codes**: Use appropriate HTTP status codes

### 2. Dependency Injection

- **Centralized Dependencies**: Use the `CommonDependencies` class for shared resources
- **Lazy Loading**: Load dependencies only when needed
- **Error Handling**: Handle dependency failures gracefully
- **Caching**: Cache expensive dependencies appropriately

### 3. Error Handling

- **Global Handlers**: Use global exception handlers for common errors
- **Route-Level Handling**: Handle domain-specific errors in route handlers
- **Logging**: Log all errors with appropriate context
- **User-Friendly Messages**: Provide clear error messages to users

### 4. Performance

- **Middleware**: Use middleware for cross-cutting concerns (logging, CORS, compression)
- **Caching**: Implement caching for expensive operations
- **Monitoring**: Monitor request/response times and error rates
- **Health Checks**: Implement comprehensive health checks

### 5. Security

- **Authentication**: Implement proper authentication for protected routes
- **Authorization**: Check user permissions for sensitive operations
- **Input Validation**: Validate all input data with Pydantic models
- **Rate Limiting**: Implement rate limiting for API endpoints

## 📚 Usage Examples

### Basic Usage

```python
# Create the application
app = create_structured_app()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Custom Router

```python
# Create custom router
class CustomRouter(BaseRouter):
    def __init__(self):
        super().__init__(prefix="/api/v1/custom", tags=["custom"])
    
    def setup_routes(self):
        @self.router.get("/")
        async def custom_endpoint():
            return {"message": "Custom endpoint"}

# Register custom router
custom_router = CustomRouter()
custom_router.setup_routes()
registry.register_router("custom", custom_router.get_router())
```

### Custom Dependencies

```python
# Add custom dependency
async def get_custom_service():
    return CustomService()

# Use in route
@router.get("/custom")
async def custom_route(
    custom_service: CustomService = Depends(get_custom_service)
):
    return await custom_service.process()
```

## 🔮 Future Enhancements

### Planned Features

1. **API Versioning**: Support for multiple API versions
2. **Rate Limiting**: Built-in rate limiting middleware
3. **Caching**: Response caching with Redis
4. **Metrics**: Prometheus metrics integration
5. **Documentation**: Enhanced OpenAPI documentation
6. **Testing**: Comprehensive testing utilities
7. **Deployment**: Docker and Kubernetes support
8. **Monitoring**: Advanced monitoring and alerting

### Performance Improvements

1. **Connection Pooling**: Optimized database connection pooling
2. **Async Processing**: Background task processing
3. **Load Balancing**: Request load balancing
4. **Caching Strategy**: Multi-level caching
5. **Compression**: Response compression optimization

This structured routing system provides a solid foundation for building scalable, maintainable, and performant AI video processing applications with clear separation of concerns and comprehensive error handling. 