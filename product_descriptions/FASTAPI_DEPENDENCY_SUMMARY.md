# FastAPI Dependency Injection System for Lazy Loading

## Overview

This document provides a comprehensive overview of the FastAPI dependency injection system implemented for managing lazy loading state and shared resources. The system provides proper lifecycle management, configuration, testing support, and performance monitoring.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Dependency Injection Patterns](#dependency-injection-patterns)
4. [Resource Management](#resource-management)
5. [Configuration Management](#configuration-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Error Handling](#error-handling)
8. [Testing Support](#testing-support)
9. [Integration Guide](#integration-guide)
10. [Best Practices](#best-practices)

## Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │ Dependency      │    │ Lazy Loading    │
│                 │    │ Manager         │    │ System          │
│  - Routes       │───▶│  - State Mgmt   │───▶│  - Loaders      │
│  - Dependencies │    │  - Lifecycle    │    │  - Data Sources │
│  - Middleware   │    │  - Monitoring   │    │  - Cache        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Relationships

- **DependencyManager**: Central manager for all shared resources
- **ResourceState**: State management for lazy loading components
- **DependencyConfig**: Configuration management with validation
- **LazyLoadingService**: Service layer using dependency injection
- **PerformanceMonitor**: Request performance tracking
- **TestDependencyManager**: Testing utilities

## Core Components

### 1. DependencyManager

Central manager for dependency injection and resource lifecycle:

```python
class DependencyManager:
    """
    Central dependency manager for FastAPI application.
    
    Manages the lifecycle of shared resources and provides
    dependency injection functions for FastAPI routes.
    """
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.state = ResourceState()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialization_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize all shared resources."""
        async with self._initialization_lock:
            if self.state.is_initialized:
                return
            
            # Initialize lazy loading manager
            self.state.lazy_manager = get_lazy_loading_manager()
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Initialize loaders
            await self._initialize_loaders()
            
            # Start cleanup task
            if self.config.enable_cleanup:
                self._start_cleanup_task()
            
            self.state.is_initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown dependency manager and cleanup resources."""
        self.state.is_shutting_down = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            await self._cleanup_task
        
        # Close lazy loading manager
        if self.state.lazy_manager:
            await self.state.lazy_manager.close_all()
        
        # Clear state
        self.state.data_sources.clear()
        self.state.loaders.clear()
        self.state.stats.clear()
```

### 2. ResourceState

State management for shared resources:

```python
class ResourceState(BaseModel):
    """State management for shared resources."""
    
    lazy_manager: Optional[LazyLoadingManager] = None
    data_sources: Dict[str, Any] = Field(default_factory=dict)
    loaders: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)
    is_initialized: bool = False
    is_shutting_down: bool = False
```

### 3. DependencyConfig

Configuration management with validation:

```python
class DependencyConfig(BaseModel):
    """Configuration for dependency injection system."""
    
    # Lazy loading configurations
    default_strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND
    default_batch_size: int = 100
    default_cache_ttl: int = 300
    default_max_memory: int = 1024 * 1024 * 100  # 100MB
    
    # Resource management
    enable_cleanup: bool = True
    cleanup_interval: int = 60
    max_connections: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    enable_metrics: bool = True
    
    # Performance
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
```

## Dependency Injection Patterns

### 1. Service Layer Dependencies

```python
class LazyLoadingService:
    """Service layer using dependency injection."""
    
    def __init__(
        self,
        lazy_manager: LazyLoadingManager = Depends(get_lazy_manager_dependency),
        config: DependencyConfig = Depends(get_config)
    ):
        self.lazy_manager = lazy_manager
        self.config = config
    
    async def get_product(
        self,
        product_id: str,
        loader = Depends(get_products_on_demand_loader)
    ) -> Dict[str, Any]:
        """Get product using on-demand loading."""
        try:
            data = await loader.get_item(product_id)
            if data is None:
                raise HTTPException(status_code=404, detail="Product not found")
            return data
        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to load product")
```

### 2. Route Dependencies

```python
@app.get("/products/{product_id}")
async def get_product(
    product_id: str,
    request: Request,
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get product using on-demand loading with dependency injection."""
    try:
        product = await service.get_product(product_id)
        
        return {
            "product": product,
            "request_id": get_request_id(request),
            "user_context": user_context,
            "cached": service.lazy_manager.get_loader("products_on_demand").stats.cache_hits > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```

### 3. Request-Scoped Dependencies

```python
def get_request_id(request: Request) -> str:
    """Get unique request ID."""
    return getattr(request.state, "request_id", "unknown")

def get_user_context(request: Request) -> Dict[str, Any]:
    """Get user context from request."""
    return {
        "user_id": getattr(request.state, "user_id", None),
        "session_id": getattr(request.state, "session_id", None),
        "request_id": get_request_id(request)
    }
```

### 4. Configuration Dependencies

```python
@lru_cache()
def get_default_config() -> DependencyConfig:
    """Get default configuration (cached)."""
    return DependencyConfig()

def get_custom_config(
    strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND,
    batch_size: int = 100,
    cache_ttl: int = 300
) -> DependencyConfig:
    """Get custom configuration."""
    return DependencyConfig(
        default_strategy=strategy,
        default_batch_size=batch_size,
        default_cache_ttl=cache_ttl
    )
```

## Resource Management

### 1. Lifecycle Management

```python
def create_app(config: Optional[DependencyConfig] = None) -> FastAPI:
    """Create FastAPI application with dependency injection."""
    
    # Use provided config or default
    if config is None:
        config = get_default_config()
    
    # Create dependency manager
    dependency_manager = DependencyManager(config)
    set_dependency_manager(dependency_manager)
    
    # Create FastAPI app with lifespan management
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await dependency_manager.initialize()
        yield
        # Shutdown
        await dependency_manager.shutdown()
    
    app = FastAPI(
        title="Lazy Loading API with Dependency Injection",
        description="Advanced lazy loading system with FastAPI dependency injection",
        version="1.0.0",
        lifespan=lifespan
    )
    
    return app
```

### 2. Background Tasks

```python
@app.post("/products/batch")
async def get_products_batch(
    request: BatchProductRequest,
    background_tasks: BackgroundTasks,
    service: LazyLoadingService = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get multiple products using background loading."""
    try:
        products = await service.get_products_batch(request.product_ids)
        
        # Add background task for cleanup
        background_tasks.add_task(cleanup_old_cache, service.lazy_manager)
        
        return {
            "products": products,
            "requested_count": len(request.product_ids),
            "loaded_count": len(products),
            "user_context": user_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def cleanup_old_cache(lazy_manager):
    """Background task to cleanup old cache entries."""
    try:
        # Simulate cleanup
        await asyncio.sleep(1)
        logger.info("Cache cleanup completed")
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
```

### 3. Resource Cleanup

```python
async def _cleanup_loop(self) -> None:
    """Background cleanup loop."""
    while not self.state.is_shutting_down:
        try:
            await asyncio.sleep(self.config.cleanup_interval)
            await self._cleanup_resources()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def _cleanup_resources(self) -> None:
    """Cleanup expired resources."""
    if self.state.lazy_manager:
        # Update stats
        self.state.stats = self.state.lazy_manager.get_stats()
        
        # Log cleanup info
        logger.debug(f"Cleanup completed. Stats: {self.state.stats}")
```

## Configuration Management

### 1. Environment-Based Configuration

```python
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    @staticmethod
    def get_lazy_loading_config() -> LazyLoadingConfig:
        """Get configuration from environment variables."""
        return LazyLoadingConfig(
            strategy=LoadingStrategy(os.getenv("LAZY_LOADING_STRATEGY", "on_demand")),
            batch_size=int(os.getenv("LAZY_LOADING_BATCH_SIZE", "100")),
            max_memory=int(os.getenv("LAZY_LOADING_MAX_MEMORY", str(1024 * 1024 * 100))),
            cache_ttl=int(os.getenv("LAZY_LOADING_CACHE_TTL", "300")),
            enable_monitoring=os.getenv("LAZY_LOADING_MONITORING", "true").lower() == "true",
            enable_cleanup=os.getenv("LAZY_LOADING_CLEANUP", "true").lower() == "true"
        )
```

### 2. Configuration Validation

```python
class DependencyConfig(BaseModel):
    """Configuration for dependency injection system."""
    
    default_strategy: LoadingStrategy = Field(
        default=LoadingStrategy.ON_DEMAND,
        description="Default loading strategy"
    )
    
    default_batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Default batch size for operations"
    )
    
    default_cache_ttl: int = Field(
        default=300,
        ge=1,
        description="Default cache TTL in seconds"
    )
    
    @validator('default_batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size is reasonable."""
        if v > 10000:
            raise ValueError("Batch size too large")
        return v
```

## Performance Monitoring

### 1. Request Performance Tracking

```python
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Track request performance."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=True)
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=False)
        raise
```

### 2. Performance Monitor

```python
def get_performance_monitor():
    """Get performance monitor dependency."""
    class PerformanceMonitor:
        def __init__(self):
            self.request_times: List[float] = []
            self.error_count = 0
            self.success_count = 0
        
        def record_request(self, duration: float, success: bool = True):
            """Record request performance."""
            self.request_times.append(duration)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
        
        def get_stats(self) -> Dict[str, Any]:
            """Get performance statistics."""
            if not self.request_times:
                return {"error": "No requests recorded"}
            
            return {
                "total_requests": len(self.request_times),
                "success_count": self.success_count,
                "error_count": self.error_count,
                "avg_response_time": sum(self.request_times) / len(self.request_times),
                "min_response_time": min(self.request_times),
                "max_response_time": max(self.request_times)
            }
    
    return PerformanceMonitor()
```

### 3. System Statistics

```python
@app.get("/stats/system")
async def get_system_stats(
    service: LazyLoadingService = Depends(),
    performance_stats: Dict[str, Any] = Depends(lambda: performance_monitor.get_stats())
):
    """Get comprehensive system statistics."""
    try:
        lazy_stats = await service.get_system_stats()
        
        return SystemStatsResponse(
            loaders=lazy_stats,
            memory_usage=lazy_stats.get("memory", {}),
            performance=performance_stats,
            uptime=time.time() - lazy_stats.get("start_time", time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
```

## Error Handling

### 1. Exception Handlers with Dependencies

```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError with dependency injection."""
    performance_monitor.record_request(0, success=False)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": str(exc),
            "request_id": get_request_id(request),
            "timestamp": time.time()
        }
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle RuntimeError with dependency injection."""
    performance_monitor.record_request(0, success=False)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "request_id": get_request_id(request),
            "timestamp": time.time()
        }
    )
```

### 2. Service Layer Error Handling

```python
async def get_product(
    self,
    product_id: str,
    loader = Depends(get_products_on_demand_loader)
) -> Dict[str, Any]:
    """Get product using on-demand loading."""
    try:
        data = await loader.get_item(product_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return data
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load product")
```

## Testing Support

### 1. Test Dependency Manager

```python
class TestDependencyManager:
    """Test dependency manager for unit testing."""
    
    def __init__(self, config: Optional[DependencyConfig] = None):
        self.config = config or DependencyConfig()
        self.manager = DependencyManager(self.config)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.manager.initialize()
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.manager.shutdown()

def get_test_dependencies():
    """Get test dependencies for unit testing."""
    config = DependencyConfig(
        default_strategy=LoadingStrategy.ON_DEMAND,
        default_batch_size=10,
        default_cache_ttl=60,
        enable_cleanup=False
    )
    
    return TestDependencyManager(config)
```

### 2. Testing Endpoints

```python
@app.get("/test/dependencies")
async def test_dependencies(
    config: DependencyConfig = Depends(get_config),
    lazy_manager = Depends(get_lazy_manager_dependency),
    stats: Dict[str, Any] = Depends(get_stats_dependency),
    request_id: str = Depends(get_request_id)
):
    """Test dependency injection system."""
    return {
        "config": config.dict(),
        "lazy_manager_initialized": lazy_manager is not None,
        "stats": stats,
        "request_id": request_id,
        "timestamp": time.time()
    }

@app.get("/test/loaders")
async def test_loaders(
    products_loader = Depends(get_loader_dependency("products_on_demand")),
    users_loader = Depends(get_loader_dependency("users_paginated")),
    items_loader = Depends(get_loader_dependency("items_streaming"))
):
    """Test loader dependencies."""
    return {
        "products_loader": type(products_loader).__name__,
        "users_loader": type(users_loader).__name__,
        "items_loader": type(items_loader).__name__,
        "all_initialized": all([
            products_loader is not None,
            users_loader is not None,
            items_loader is not None
        ])
    }
```

## Integration Guide

### 1. Basic Integration

```python
from fastapi import FastAPI, Depends
from fastapi_dependency_injection import create_app, get_config

# Create app with dependency injection
app = create_app()

@app.get("/health")
async def health_check(
    config = Depends(get_config)
):
    return {"status": "healthy", "config": config.dict()}
```

### 2. Service Layer Integration

```python
from fastapi_dependency_injection import LazyLoadingService

@app.get("/products/{product_id}")
async def get_product(
    product_id: str,
    service: LazyLoadingService = Depends()
):
    return await service.get_product(product_id)
```

### 3. Custom Configuration

```python
from fastapi_dependency_injection import DependencyConfig, create_app

# Custom configuration
config = DependencyConfig(
    default_strategy=LoadingStrategy.PAGINATED,
    default_batch_size=200,
    enable_monitoring=True
)

# Create app with custom config
app = create_app(config)
```

### 4. Testing Integration

```python
import pytest
from fastapi.testclient import TestClient
from fastapi_dependency_injection import TestDependencyManager

@pytest.fixture
async def test_app():
    """Create test app with dependency injection."""
    async with TestDependencyManager() as manager:
        app = create_app()
        yield app

@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

## Best Practices

### 1. Dependency Organization

```python
# ✅ Good: Organized dependencies
def get_lazy_manager_dependency() -> LazyLoadingManager:
    """Get lazy loading manager dependency."""
    return get_dependency_manager().get_lazy_manager()

def get_loader_dependency(loader_name: str):
    """Get specific loader dependency."""
    return get_dependency_manager().get_loader(loader_name)

def get_config() -> DependencyConfig:
    """Get dependency configuration."""
    return get_dependency_manager().config
```

### 2. Resource Lifecycle

```python
# ✅ Good: Proper lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await dependency_manager.initialize()
    yield
    # Shutdown
    await dependency_manager.shutdown()
```

### 3. Error Handling

```python
# ✅ Good: Comprehensive error handling
async def get_product(
    self,
    product_id: str,
    loader = Depends(get_products_on_demand_loader)
) -> Dict[str, Any]:
    try:
        data = await loader.get_item(product_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return data
    except Exception as e:
        logger.error(f"Failed to get product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load product")
```

### 4. Performance Monitoring

```python
# ✅ Good: Performance tracking
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=True)
        return response
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_request(duration, success=False)
        raise
```

### 5. Testing

```python
# ✅ Good: Test utilities
class TestDependencyManager:
    async def __aenter__(self):
        await self.manager.initialize()
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.shutdown()
```

## Benefits

### 1. **Resource Management**
- Proper lifecycle management for shared resources
- Automatic cleanup and memory management
- Background task support

### 2. **Testing Support**
- Easy mocking and testing of dependencies
- Isolated test environments
- Comprehensive test utilities

### 3. **Performance Monitoring**
- Request performance tracking
- System statistics collection
- Performance optimization insights

### 4. **Configuration Management**
- Environment-based configuration
- Type-safe configuration validation
- Dynamic configuration updates

### 5. **Error Handling**
- Centralized error handling
- Proper error responses
- Error tracking and logging

### 6. **Scalability**
- Efficient resource sharing
- Connection pooling
- Load balancing support

## Conclusion

The FastAPI dependency injection system provides:

1. **Comprehensive Resource Management**: Proper lifecycle management for all shared resources
2. **Type-Safe Configuration**: Pydantic-based configuration with validation
3. **Performance Monitoring**: Built-in performance tracking and statistics
4. **Testing Support**: Complete testing utilities and mocking support
5. **Error Handling**: Centralized error handling with proper responses
6. **Scalability**: Efficient resource sharing and management

This system ensures that the lazy loading functionality is properly integrated with FastAPI's dependency injection system, providing excellent maintainability, testability, and performance monitoring capabilities. 