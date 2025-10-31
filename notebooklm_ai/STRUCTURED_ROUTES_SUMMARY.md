# Structured Routes and Dependencies - FastAPI Organization Guide

## Overview

This document outlines the improved structure for organizing FastAPI routes and dependencies to optimize readability and maintainability. The new structure follows modular architecture principles with clear separation of concerns.

## Key Improvements

### 1. Modular Architecture
- **Domain-Specific Routers**: Routes organized by business domain
- **Service Layer**: Business logic separated from route handlers
- **Dependency Injection**: Centralized dependency management
- **Clear Interfaces**: Well-defined service contracts

### 2. File Organization

```
notebooklm_ai/
├── structured_routes_app.py      # Main application with core structure
├── route_organization.py         # Route handlers and router factory
├── dependency_management.py      # Dependency injection container
└── STRUCTURED_ROUTES_SUMMARY.md  # This documentation
```

## Architecture Components

### 1. Dependency Container (`dependency_management.py`)

**Purpose**: Centralized dependency injection with lifecycle management

**Key Features**:
- Service factory pattern
- Health monitoring for all services
- Graceful shutdown handling
- Service status tracking

**Service Types**:
```python
class ServiceType(str, Enum):
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    DIFFUSION = "diffusion"
    AUTH = "auth"
    MONITORING = "monitoring"
```

**Usage Example**:
```python
# Get service from container
diffusion_service = await container.get_service(ServiceType.DIFFUSION)

# Health check all services
health_status = await container.health_check()
```

### 2. Router Factory (`route_organization.py`)

**Purpose**: Create domain-specific routers with proper organization

**Router Types**:
- **Diffusion Router**: Image generation endpoints
- **Health Router**: System monitoring endpoints
- **Admin Router**: Administrative operations

**Benefits**:
- Clear separation by domain
- Consistent error handling
- Centralized route registration
- Easy to extend and maintain

**Example Router Creation**:
```python
# Create diffusion router
diffusion_router = RouterFactory.create_diffusion_router()

# Include in main app
app.include_router(diffusion_router)
```

### 3. Route Handlers (`route_organization.py`)

**Purpose**: Organized route handlers by domain

**Handler Classes**:
- `DiffusionRouteHandlers`: Image generation logic
- `HealthRouteHandlers`: System health monitoring
- `AdminRouteHandlers`: Administrative operations

**Benefits**:
- Clear responsibility separation
- Easy to test individual handlers
- Consistent error handling patterns
- Reusable business logic

**Example Handler**:
```python
class DiffusionRouteHandlers:
    @staticmethod
    async def generate_single_image(
        request: DiffusionRequest,
        diffusion_service: AsyncDiffusionService = Depends(get_diffusion_service),
        current_user: str = Depends(get_current_user)
    ) -> DiffusionResponse:
        # Business logic here
        pass
```

## Dependency Injection Patterns

### 1. Service Dependencies

**Pattern**: Use dependency injection for all external services

```python
async def get_diffusion_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> DiffusionService:
    return await container.get_service(ServiceType.DIFFUSION)
```

**Benefits**:
- Easy to mock for testing
- Consistent service access
- Centralized configuration
- Lifecycle management

### 2. Authentication Dependencies

**Pattern**: Centralized authentication with optional user access

```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth_service: AuthService = Depends(get_auth_service)
) -> str:
    payload = await auth_service.validate_token(credentials.credentials)
    return payload.get("user_id", "anonymous")

async def get_current_user_optional(request: Request) -> Optional[str]:
    # For public endpoints
    pass
```

### 3. Rate Limiting Dependencies

**Pattern**: Automatic rate limiting with cache-based tracking

```python
async def get_rate_limit_info(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    # Rate limiting logic
    pass
```

## Route Organization Principles

### 1. Domain Separation

**Principle**: Group routes by business domain

```
/api/v1/diffusion/     # Image generation
/api/v1/health/        # System monitoring
/api/v1/admin/         # Administrative operations
```

### 2. Consistent Patterns

**Principle**: Use consistent patterns across all routes

- **Request/Response Models**: Pydantic models for validation
- **Error Handling**: Consistent error responses
- **Authentication**: JWT-based authentication
- **Rate Limiting**: Automatic rate limiting
- **Monitoring**: Performance metrics collection

### 3. Clear Documentation

**Principle**: Self-documenting routes with OpenAPI

```python
@app.add_api_route(
    "/api/v1/diffusion/generate",
    DiffusionRouteHandlers.generate_single_image,
    methods=["POST"],
    response_model=DiffusionResponse,
    summary="Generate single image from text prompt",
    description="Generate an image from a text prompt using diffusion models",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
```

## Service Layer Architecture

### 1. Service Interfaces

**Principle**: All services implement a common interface

```python
class ServiceInterface:
    async def health_check(self) -> ServiceStatus:
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        raise NotImplementedError
```

### 2. Service Implementation

**Example**: Database Service

```python
class DatabaseService(ServiceInterface):
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._database: Optional[Database] = None
    
    async def get_database(self) -> Database:
        if self._database is None:
            self._database = Database(self.database_url)
            await self._database.connect()
        return self._database
    
    async def health_check(self) -> ServiceStatus:
        try:
            await self.execute_query("SELECT 1")
            return ServiceStatus.HEALTHY
        except Exception:
            return ServiceStatus.UNHEALTHY
```

## Error Handling Strategy

### 1. Global Exception Handlers

**Pattern**: Centralized error handling

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### 2. Service-Level Error Handling

**Pattern**: Handle errors at service level

```python
async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
    try:
        result = await db.fetch_all(text(query), params or {})
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise
```

## Performance Optimization

### 1. Connection Pooling

**Pattern**: Reuse connections for better performance

```python
async def get_session(self) -> aiohttp.ClientSession:
    if self._session is None or self._session.closed:
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        self._session = aiohttp.ClientSession(connector=connector)
    return self._session
```

### 2. Health Check Caching

**Pattern**: Cache health check results to reduce overhead

```python
async def health_check(self) -> ServiceStatus:
    current_time = time.time()
    
    # Cache health check results
    if current_time - self._last_health_check < self._health_cache_duration:
        return ServiceStatus.HEALTHY
    
    # Perform actual health check
    # ...
```

### 3. Async Operations

**Pattern**: Use async/await for all I/O operations

```python
async def generate_batch_images(self, requests: List[DiffusionRequest]) -> BatchDiffusionResponse:
    # Process requests in parallel
    tasks = [
        self.generate_single_image(req)
        for req in requests
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ...
```

## Testing Strategy

### 1. Dependency Mocking

**Pattern**: Easy to mock dependencies for testing

```python
# In tests
async def test_generate_image():
    mock_diffusion_service = Mock()
    mock_diffusion_service.generate_single_image.return_value = expected_response
    
    result = await DiffusionRouteHandlers.generate_single_image(
        request=test_request,
        diffusion_service=mock_diffusion_service,
        current_user="test_user"
    )
    
    assert result == expected_response
```

### 2. Service Testing

**Pattern**: Test services independently

```python
async def test_database_service():
    service = DatabaseService("test_url")
    
    # Test health check
    status = await service.health_check()
    assert status == ServiceStatus.HEALTHY
    
    # Test query execution
    result = await service.execute_query("SELECT 1")
    assert len(result) == 1
```

## Configuration Management

### 1. Environment-Based Configuration

**Pattern**: Load configuration from environment

```python
class AppConfig:
    def __init__(self):
        self.app_name: str = "notebooklm_ai"
        self.version: str = "1.0.0"
        self.database_url: str = "postgresql://user:pass@localhost/db"
        self.redis_url: str = "redis://localhost:6379"
        # ...
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        config = cls()
        # Load from environment variables
        return config
```

### 2. Service Configuration

**Pattern**: Pass configuration to services

```python
# Create container with config
config = AppConfig.from_env()
container = DependencyContainer(config)

# Services get configuration automatically
database_service = await container.get_service(ServiceType.DATABASE)
```

## Monitoring and Observability

### 1. Performance Metrics

**Pattern**: Collect metrics for all operations

```python
def monitor_performance(operation_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric
                monitoring_service.record_metric(
                    f"{operation_name}_duration",
                    duration,
                    {"status": "success"}
                )
                
                return result
            except Exception as e:
                # Record error metric
                monitoring_service.record_metric(
                    f"{operation_name}_duration",
                    duration,
                    {"status": "error", "error_type": type(e).__name__}
                )
                raise
        return wrapper
    return decorator
```

### 2. Health Monitoring

**Pattern**: Comprehensive health checks

```python
async def health_check(self) -> Dict[str, ServiceStatus]:
    health_status = {}
    
    for service_type in ServiceType:
        try:
            service = await self.get_service(service_type)
            health_status[service_type.value] = await service.health_check()
        except Exception as e:
            health_status[service_type.value] = ServiceStatus.UNHEALTHY
    
    return health_status
```

## Best Practices Summary

### 1. Code Organization
- ✅ Separate routes by domain
- ✅ Use dependency injection
- ✅ Implement service interfaces
- ✅ Centralize configuration
- ✅ Consistent error handling

### 2. Performance
- ✅ Use connection pooling
- ✅ Implement health check caching
- ✅ Async/await for I/O operations
- ✅ Parallel processing where possible
- ✅ Monitor performance metrics

### 3. Maintainability
- ✅ Clear separation of concerns
- ✅ Consistent naming conventions
- ✅ Comprehensive documentation
- ✅ Easy to test structure
- ✅ Modular architecture

### 4. Scalability
- ✅ Stateless services
- ✅ Horizontal scaling support
- ✅ Load balancing ready
- ✅ Monitoring and alerting
- ✅ Graceful degradation

## Migration Guide

### From Monolithic Structure

1. **Extract Services**: Move business logic to service classes
2. **Create Routers**: Organize routes by domain
3. **Implement DI**: Add dependency injection container
4. **Add Monitoring**: Implement health checks and metrics
5. **Update Tests**: Refactor tests for new structure

### Benefits After Migration

- **Improved Readability**: Clear structure and organization
- **Better Maintainability**: Modular components
- **Enhanced Testability**: Easy to mock dependencies
- **Increased Performance**: Optimized async operations
- **Better Monitoring**: Comprehensive health checks
- **Easier Scaling**: Horizontal scaling support

## Conclusion

The structured routes and dependencies approach provides a solid foundation for building maintainable, scalable FastAPI applications. By following these patterns, you can create applications that are easy to understand, test, and extend while maintaining high performance and reliability.

The key is to start with a clear architecture and stick to the established patterns throughout the development process. This ensures consistency and makes the codebase easier to work with as it grows in complexity. 