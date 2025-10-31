# Lifespan Migration Guide - Instagram Captions API v14.0

## Overview
This guide documents the migration from deprecated `app.on_event("startup")` and `app.on_event("shutdown")` to modern `lifespan` context managers in FastAPI applications.

## Why Migrate to Lifespan Context Managers?

### Benefits of Lifespan Context Managers
1. **Better Resource Management**: Proper cleanup guaranteed even on exceptions
2. **Cleaner Code**: Single function handles both startup and shutdown
3. **Async Context**: Native async/await support for initialization
4. **Error Handling**: Automatic cleanup on startup failures
5. **Future-Proof**: Recommended approach in FastAPI documentation
6. **Testing**: Easier to test and mock lifecycle events

### Problems with `app.on_event`
1. **Deprecated**: Marked for removal in future FastAPI versions
2. **No Guaranteed Cleanup**: Shutdown events may not run on startup failures
3. **Separate Functions**: Startup and shutdown logic scattered
4. **Limited Error Handling**: No automatic cleanup on exceptions

## Migration Examples

### Before: Deprecated `app.on_event` Pattern

```python
# ❌ DEPRECATED PATTERN
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting up...")
    # Initialize resources
    await initialize_database()
    await load_models()
    print("✅ Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down...")
    # Cleanup resources
    await close_database()
    await unload_models()
    print("✅ Shutdown complete")
```

### After: Modern Lifespan Context Manager

```python
# ✅ MODERN PATTERN
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper startup and shutdown management."""
    # Startup
    print("🚀 Starting up...")
    try:
        await initialize_database()
        await load_models()
        print("✅ Startup complete")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown (guaranteed to run)
    print("🛑 Shutting down...")
    try:
        await close_database()
        await unload_models()
        print("✅ Shutdown complete")
    except Exception as e:
        print(f"⚠️ Shutdown warning: {e}")

app = FastAPI(lifespan=lifespan)
```

## Real-World Implementation in v14.0

### Complete Lifespan Implementation

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.optimized_engine import optimized_engine, performance_monitor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with proper startup and shutdown management."""
    # Startup
    print("🚀 Instagram Captions API v14.0 - Optimized starting up...")
    print("📁 Modular structure loaded:")
    print("   • Types: Pydantic models and schemas")
    print("   • Routes: Organized endpoint routers")
    print("   • Utils: Validation and helper functions")
    print("   • Core: Optimized AI engine")
    print("   • Static: OpenAPI documentation")
    print("⚡ Performance optimizations enabled:")
    print("   • JIT compilation")
    print("   • Multi-level caching")
    print("   • Batch processing")
    print("   • Mixed precision")
    print("   • GPU acceleration")
    
    try:
        # Initialize engine if needed
        if not optimized_engine.model:
            await optimized_engine._initialize_models()
        
        # Initialize performance monitoring
        performance_monitor.start()
        
        print("✅ All services initialized successfully")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # Cleanup any partially initialized resources
        await _cleanup_on_startup_failure()
        raise
    
    yield
    
    # Shutdown (guaranteed to run)
    print("🛑 Shutting down Instagram Captions API v14.0...")
    try:
        # Shutdown thread pool executor
        if optimized_engine.executor:
            optimized_engine.executor.shutdown(wait=True)
        
        # Stop performance monitoring
        performance_monitor.stop()
        
        # Clear caches
        optimized_engine.cache.clear()
        
        print("✅ Shutdown completed successfully")
        
    except Exception as e:
        print(f"⚠️ Shutdown warning: {e}")

async def _cleanup_on_startup_failure():
    """Cleanup resources if startup fails."""
    try:
        if optimized_engine.executor:
            optimized_engine.executor.shutdown(wait=False)
        optimized_engine.cache.clear()
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

# FastAPI app with lifespan
app = FastAPI(
    title="Instagram Captions API v14.0 - Optimized",
    description="Ultra-fast Instagram captions generation with advanced AI",
    version="14.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
```

## Advanced Lifespan Patterns

### 1. Resource Management with Context Managers

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Advanced lifespan with multiple resource managers."""
    
    # Startup with multiple resources
    async with (
        DatabaseConnection() as db,
        ModelManager() as models,
        CacheManager() as cache,
        MonitoringManager() as monitoring
    ):
        # All resources initialized
        yield
        # All resources automatically cleaned up
```

### 2. Conditional Initialization

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan with conditional initialization."""
    
    # Startup
    resources = {}
    
    try:
        # Initialize database only if configured
        if config.DATABASE_ENABLED:
            resources['db'] = await initialize_database()
        
        # Initialize AI models only if available
        if config.AI_MODELS_ENABLED:
            resources['models'] = await initialize_ai_models()
        
        # Initialize cache only if Redis available
        if config.REDIS_ENABLED:
            resources['cache'] = await initialize_redis_cache()
        
        yield
        
    finally:
        # Cleanup all initialized resources
        for resource_name, resource in resources.items():
            try:
                await resource.close()
            except Exception as e:
                print(f"⚠️ Failed to close {resource_name}: {e}")
```

### 3. Health Check Integration

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan with health check integration."""
    
    # Startup
    app.state.health_status = "starting"
    
    try:
        await initialize_services()
        app.state.health_status = "healthy"
        yield
    except Exception as e:
        app.state.health_status = "unhealthy"
        raise
    finally:
        app.state.health_status = "shutting_down"
        await cleanup_services()
```

## Migration Checklist

### ✅ Steps to Migrate

1. **Import Required Modules**
   ```python
   from contextlib import asynccontextmanager
   ```

2. **Create Lifespan Function**
   ```python
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Startup logic
       yield
       # Shutdown logic
   ```

3. **Update FastAPI App**
   ```python
   app = FastAPI(lifespan=lifespan)
   ```

4. **Remove Old Event Handlers**
   ```python
   # Remove these:
   # @app.on_event("startup")
   # @app.on_event("shutdown")
   ```

5. **Test Startup and Shutdown**
   - Verify startup logs
   - Test graceful shutdown
   - Check resource cleanup

### 🔍 Testing Lifespan Events

```python
import pytest
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

@pytest.fixture
def test_app():
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        # Test startup
        app.state.test_startup = True
        yield
        # Test shutdown
        app.state.test_shutdown = True
    
    app = FastAPI(lifespan=test_lifespan)
    return app

def test_lifespan_events(test_app):
    with TestClient(test_app) as client:
        # Startup should have run
        assert test_app.state.test_startup is True
        assert test_app.state.test_shutdown is None
    
    # Shutdown should have run
    assert test_app.state.test_shutdown is True
```

## Performance Considerations

### Startup Optimization
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized lifespan with parallel initialization."""
    
    # Parallel initialization for better startup time
    startup_tasks = [
        initialize_database(),
        initialize_ai_models(),
        initialize_cache(),
        initialize_monitoring()
    ]
    
    try:
        await asyncio.gather(*startup_tasks)
        yield
    finally:
        # Sequential cleanup for safety
        await cleanup_services()
```

### Memory Management
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan with memory management."""
    
    # Startup
    import gc
    gc.collect()  # Clean memory before startup
    
    try:
        await initialize_services()
        yield
    finally:
        # Cleanup and garbage collection
        await cleanup_services()
        gc.collect()  # Clean memory after shutdown
```

## Best Practices

### 1. Error Handling
- Always use try/finally in lifespan
- Log startup and shutdown events
- Handle partial initialization failures
- Provide meaningful error messages

### 2. Resource Management
- Use context managers for resources
- Implement proper cleanup procedures
- Handle cleanup failures gracefully
- Monitor resource usage

### 3. Logging
- Log all startup and shutdown events
- Include timing information
- Log resource initialization status
- Provide clear status messages

### 4. Testing
- Test startup with missing dependencies
- Test shutdown under various conditions
- Verify resource cleanup
- Test error scenarios

## Summary

The migration to lifespan context managers provides:

- **Better Resource Management**: Guaranteed cleanup
- **Cleaner Code**: Single function for lifecycle
- **Future-Proof**: Modern FastAPI approach
- **Error Handling**: Automatic cleanup on failures
- **Testing**: Easier to test and mock

This approach ensures robust application lifecycle management and follows FastAPI best practices. 