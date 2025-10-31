# FastAPI Dependency Injection Implementation v15

## Overview

The Ultra-Optimized SEO Service v15 implements comprehensive FastAPI dependency injection for managing state and shared resources. This approach provides better resource management, improved testability, and cleaner architecture.

## Key Features

### 1. Dependency Container
- Centralized resource management
- Lazy initialization of services
- Automatic cleanup and lifecycle management
- Thread-safe singleton pattern

### 2. Service Layer Architecture
- SEO service with injected dependencies
- Clean separation of concerns
- Easy testing and mocking
- Consistent error handling

### 3. Resource Management
- Connection pooling for databases
- HTTP client management
- Cache management
- Rate limiting with dependency injection

## Architecture Components

### DependencyContainer Class

```python
class DependencyContainer:
    """FastAPI dependency injection container for managing shared resources."""
    
    def __init__(self):
        self._cache_manager: Optional[CacheManager] = None
        self._static_cache: Optional[StaticDataCache] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._lazy_loader: Optional[LazyDataLoader] = None
        self._bulk_processor: Optional[BulkSEOProcessor] = None
        self._redis_client: Optional[redis.Redis] = None
        self._mongo_client: Optional[AsyncIOMotorClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._config: Optional[Config] = None
        self._logger: Optional[structlog.BoundLogger] = None
        self._startup_time: float = time.time()
        self._request_count: int = 0
        self._error_count: int = 0
```

### SEO Service with Dependency Injection

```python
class SEOService:
    """SEO analysis service with dependency injection."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        static_cache: StaticDataCache,
        http_client: httpx.AsyncClient,
        logger: structlog.BoundLogger
    ):
        self.cache_manager = cache_manager
        self.static_cache = static_cache
        self.http_client = http_client
        self.logger = logger
```

## Dependency Injection Functions

### Core Dependencies

```python
async def get_dependency_container() -> DependencyContainer:
    """Get dependency container instance."""
    return container

async def get_config() -> Config:
    """Get application configuration."""
    return container.config

async def get_logger() -> structlog.BoundLogger:
    """Get structured logger instance."""
    return container.logger
```

### Service Dependencies

```python
async def get_cache_manager() -> CacheManager:
    """Get cache manager instance."""
    return container.cache_manager

async def get_static_cache() -> StaticDataCache:
    """Get static cache instance."""
    return container.static_cache

async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return container.rate_limiter
```

### Database Dependencies

```python
async def get_redis() -> Optional[redis.Redis]:
    """Get Redis client with async connection pooling."""
    if container._redis_client is None:
        try:
            container._redis_client = redis.from_url(
                container.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                health_check_interval=30
            )
            await container._redis_client.ping()
            container.logger.info("Redis connection established")
        except Exception as e:
            container.logger.warning("Redis connection failed", error=str(e))
            container._redis_client = None
    return container._redis_client

async def get_mongo() -> Optional[AsyncIOMotorClient]:
    """Get MongoDB client with async connection pooling."""
    if container._mongo_client is None:
        try:
            container._mongo_client = AsyncIOMotorClient(
                container.config.mongo_url,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            await container._mongo_client.admin.command('ping')
            container.logger.info("MongoDB connection established")
        except Exception as e:
            container.logger.warning("MongoDB connection failed", error=str(e))
            container._mongo_client = None
    return container._mongo_client
```

### HTTP Client Dependency

```python
async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client with async connection pooling."""
    if container._http_client is None:
        container._http_client = httpx.AsyncClient(
            timeout=container.config.timeout,
            limits=httpx.Limits(
                max_connections=container.config.max_connections,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            http2=True,
            follow_redirects=True
        )
        container.logger.info("HTTP client initialized")
    return container._http_client
```

### Service Factory

```python
async def get_seo_service(
    cache_manager: CacheManager = Depends(get_cache_manager),
    static_cache: StaticDataCache = Depends(get_static_cache),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    logger: structlog.BoundLogger = Depends(get_logger)
) -> 'SEOService':
    """Get SEO service instance with all dependencies."""
    return SEOService(
        cache_manager=cache_manager,
        static_cache=static_cache,
        http_client=http_client,
        logger=logger
    )
```

## API Endpoints with Dependency Injection

### SEO Analysis Endpoint

```python
@app.post("/analyze", response_model=SEOResponse)
async def analyze_seo_endpoint(
    request: SEORequest,
    background_tasks: BackgroundTasks,
    seo_service: SEOService = Depends(get_seo_service),
    rate_limit: None = Depends(check_rate_limit),
    container: DependencyContainer = Depends(get_dependency_container)
):
    """Analyze SEO for given URL with dependency injection."""
    try:
        # Increment request counter
        container.increment_request_count()
        
        # Use model_dump for optimized serialization
        params = SEOParamsModel(**request.model_dump(mode='json'))
        result = await seo_service.analyze_seo(params)
        
        # Add background task for metrics
        background_tasks.add_task(
            log_metrics, 
            params.url, 
            result.score,
            container.logger
        )
        
        return SEOResponse(**result.model_dump(mode='json'))
        
    except Exception as e:
        container.increment_error_count()
        container.logger.error("Unexpected error in SEO analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Health Check Endpoint

```python
@app.get("/health")
async def health_check(
    container: DependencyContainer = Depends(get_dependency_container),
    redis_client: Optional[redis.Redis] = Depends(get_redis),
    mongo_client: Optional[AsyncIOMotorClient] = Depends(get_mongo)
):
    """Health check endpoint with dependency injection."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": container.uptime,
        "version": "15.0.0",
        "checks": {}
    }
    
    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            health_status["checks"]["redis"] = "healthy"
        except Exception as e:
            health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["redis"] = "unavailable"
        health_status["status"] = "degraded"
    
    # Check MongoDB
    if mongo_client:
        try:
            await mongo_client.admin.command('ping')
            health_status["checks"]["mongodb"] = "healthy"
        except Exception as e:
            health_status["checks"]["mongodb"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["mongodb"] = "unavailable"
        health_status["status"] = "degraded"
    
    # Check memory usage
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    health_status["checks"]["memory_usage_mb"] = memory_usage
    
    if memory_usage > 1000:  # More than 1GB
        health_status["status"] = "degraded"
    
    return health_status
```

### Cache Management Endpoints

```python
@app.post("/cache/clear")
async def clear_cache_endpoint(
    pattern: str = None,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Clear cache entries with dependency injection."""
    try:
        await cache_manager.clear(pattern)
        return {"message": f"Cache cleared successfully", "pattern": pattern}
    except Exception as e:
        logger.error("Cache clear failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cache clear failed")

@app.get("/cache/stats")
async def cache_stats_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Get cache statistics with dependency injection."""
    try:
        return cache_manager.get_stats()
    except Exception as e:
        logger.error("Cache stats failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cache stats failed")
```

## Rate Limiting with Dependency Injection

```python
async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    config: Config = Depends(get_config),
    logger: structlog.BoundLogger = Depends(get_logger)
) -> None:
    """Check rate limit for request with dependency injection."""
    client_id = request.client.host if request.client else 'unknown'
    
    try:
        rate_limit_result = await rate_limiter.check_rate_limit(
            client_id, config.rate_limit, 60
        )
        
        if not rate_limit_result['allowed']:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Try again in {int(rate_limit_result['reset_time'] - time.time())} seconds"
            )
    except Exception as e:
        logger.warning("Rate limiting check failed", error=str(e))
        # Allow request to proceed if rate limiting fails
```

## Lifecycle Management

### Startup Event

```python
@app.on_event("startup")
async def startup_event():
    """Application startup with dependency injection."""
    container.logger.info("Starting Ultra-Optimized SEO Service v15 with dependency injection")
    
    # Initialize Sentry
    if container.config.sentry_dsn:
        sentry_sdk.init(
            dsn=container.config.sentry_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.1
        )
        container.logger.info("Sentry initialized")
    
    # Initialize Redis with async connection pooling
    try:
        redis_client = await get_redis()
        if redis_client:
            container.logger.info("Redis connected successfully with async pooling")
        else:
            container.logger.warning("Redis connection failed, using memory cache only")
    except Exception as e:
        container.logger.warning("Redis connection failed", error=str(e))
    
    # Initialize MongoDB with async connection pooling
    try:
        mongo_client = await get_mongo()
        if mongo_client:
            container.logger.info("MongoDB connected successfully with async pooling")
        else:
            container.logger.warning("MongoDB connection failed")
    except Exception as e:
        container.logger.warning("MongoDB connection failed", error=str(e))
    
    # Initialize HTTP client with async connection pooling
    try:
        http_client = await get_http_client()
        container.logger.info("HTTP client initialized with async pooling")
    except Exception as e:
        container.logger.warning("HTTP client initialization failed", error=str(e))
    
    # Preload static data into cache
    await container.static_cache.preload_static_data()
    container.logger.info("Static data preloaded into cache")
    
    container.logger.info("Application startup completed with dependency injection")
```

### Shutdown Event

```python
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown with dependency injection."""
    container.logger.info("Shutting down Ultra-Optimized SEO Service v15")
    
    # Close Redis connection asynchronously
    if container._redis_client:
        try:
            await container._redis_client.close()
            container.logger.info("Redis connection closed")
        except Exception as e:
            container.logger.warning("Error closing Redis connection", error=str(e))
    
    # Close MongoDB connection asynchronously
    if container._mongo_client:
        try:
            container._mongo_client.close()
            container.logger.info("MongoDB connection closed")
        except Exception as e:
            container.logger.warning("Error closing MongoDB connection", error=str(e))
    
    # Close HTTP client asynchronously
    if container._http_client:
        try:
            await container._http_client.aclose()
            container.logger.info("HTTP client closed")
        except Exception as e:
            container.logger.warning("Error closing HTTP client", error=str(e))
    
    container.logger.info("Application shutdown completed with dependency injection")
```

## Benefits of Dependency Injection

### 1. Testability
- Easy to mock dependencies for unit tests
- Isolated testing of individual components
- Clear dependency boundaries

### 2. Maintainability
- Centralized resource management
- Consistent error handling
- Easy to modify dependencies

### 3. Performance
- Lazy initialization of resources
- Connection pooling
- Efficient resource sharing

### 4. Scalability
- Easy to add new dependencies
- Modular architecture
- Clear separation of concerns

## Testing with Dependency Injection

### Example Test

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture
def mock_cache_manager():
    return AsyncMock()

@pytest.fixture
def mock_static_cache():
    return MagicMock()

@pytest.fixture
def mock_http_client():
    return AsyncMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def seo_service(mock_cache_manager, mock_static_cache, mock_http_client, mock_logger):
    return SEOService(
        cache_manager=mock_cache_manager,
        static_cache=mock_static_cache,
        http_client=mock_http_client,
        logger=mock_logger
    )

async def test_analyze_seo_with_dependencies(seo_service):
    # Arrange
    params = SEOParamsModel(url="https://example.com")
    seo_service.cache_manager.get.return_value = None
    
    # Act
    result = await seo_service.analyze_seo(params)
    
    # Assert
    assert result.url == "https://example.com"
    seo_service.cache_manager.get.assert_called_once()
```

## Migration Guide

### From Global State to Dependency Injection

1. **Replace global variables with dependencies:**
   ```python
   # Before
   result = await analyze_seo(params)
   
   # After
   result = await seo_service.analyze_seo(params)
   ```

2. **Update endpoint signatures:**
   ```python
   # Before
   async def analyze_seo_endpoint(request: SEORequest):
   
   # After
   async def analyze_seo_endpoint(
       request: SEORequest,
       seo_service: SEOService = Depends(get_seo_service)
   ):
   ```

3. **Use dependency injection for logging:**
   ```python
   # Before
   logger.info("Processing request")
   
   # After
   container.logger.info("Processing request")
   ```

## Best Practices

### 1. Dependency Organization
- Group related dependencies together
- Use clear naming conventions
- Document dependency purposes

### 2. Error Handling
- Handle dependency failures gracefully
- Provide fallback mechanisms
- Log dependency errors appropriately

### 3. Resource Management
- Initialize resources lazily
- Clean up resources properly
- Monitor resource usage

### 4. Testing
- Mock dependencies for unit tests
- Test dependency injection scenarios
- Verify resource cleanup

## Performance Considerations

### 1. Lazy Initialization
- Resources are created only when needed
- Reduces startup time
- Saves memory for unused features

### 2. Connection Pooling
- Reuse database connections
- HTTP connection pooling
- Efficient resource utilization

### 3. Caching
- Cache frequently used dependencies
- Avoid repeated initialization
- Optimize dependency resolution

## Monitoring and Observability

### 1. Health Checks
- Monitor dependency health
- Track resource availability
- Alert on dependency failures

### 2. Metrics
- Track dependency usage
- Monitor performance metrics
- Log dependency interactions

### 3. Logging
- Structured logging with dependencies
- Trace dependency calls
- Monitor error rates

## Future Enhancements

### 1. Advanced Dependency Injection
- Circular dependency resolution
- Conditional dependencies
- Dependency scoping

### 2. Configuration Management
- Environment-based configuration
- Dynamic dependency configuration
- Configuration validation

### 3. Service Discovery
- Dynamic service registration
- Load balancing
- Service health monitoring

## Conclusion

The FastAPI dependency injection implementation in the Ultra-Optimized SEO Service v15 provides a robust, scalable, and maintainable architecture. It improves testability, resource management, and overall code quality while maintaining high performance and reliability.

The implementation follows FastAPI best practices and provides a solid foundation for future enhancements and scaling. 