# 🔄 Lifespan Context Managers

## Overview

This guide covers migrating from deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` decorators to modern lifespan context managers in FastAPI.

## Why Migrate?

### Problems with `@app.on_event()`
- **Deprecated**: No longer recommended in FastAPI
- **Poor error handling**: Startup failures don't prevent app from running
- **Resource leaks**: Shutdown events might not execute if startup fails
- **Testing difficulties**: Hard to test startup/shutdown logic
- **No cleanup guarantees**: Resources might not be properly cleaned up

### Benefits of Lifespan Context Managers
- **Modern approach**: Recommended by FastAPI
- **Better error handling**: Startup failures prevent app from starting
- **Guaranteed cleanup**: `finally` block ensures resources are cleaned up
- **Easier testing**: Can test startup/shutdown logic independently
- **Cleaner code**: Single context manager for both startup and shutdown

## Migration Patterns

### Basic Migration

#### ❌ Old Way (Don't use)
```python
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.redis = redis.Redis()
    app.state.db = create_engine("postgresql://...")
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.redis.close()
    app.state.db.dispose()
    logger.info("Application shutdown")
```

#### ✅ New Way (Use this)
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    app.state.redis = redis.Redis()
    app.state.db = create_engine("postgresql://...")
    logger.info("Application started")
    
    yield
    
    # Shutdown
    await app.state.redis.close()
    app.state.db.dispose()
    logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)
```

### Advanced Migration with Error Handling

```python
@asynccontextmanager
async def lifespan_with_error_handling(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with comprehensive error handling."""
    
    logger.info("🚀 Starting application...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        app.state.db_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost/ai_video",
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        
        # Test database connection
        async with app.state.db_engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database initialized")
        
        # Initialize Redis
        logger.info("🔴 Initializing Redis...")
        app.state.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("✅ Redis initialized")
        
        # Load AI models
        logger.info("🤖 Loading AI models...")
        app.state.video_model = await load_video_model()
        app.state.text_model = await load_text_model()
        logger.info("✅ AI models loaded")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        logger.info("🛑 Shutting down application...")
        
        try:
            # Cleanup in reverse order
            if hasattr(app.state, 'video_model'):
                await unload_model(app.state.video_model)
            
            if hasattr(app.state, 'text_model'):
                await unload_model(app.state.text_model)
            
            if hasattr(app.state, 'redis'):
                await app.state.redis.close()
            
            if hasattr(app.state, 'db_engine'):
                await app.state.db_engine.dispose()
                
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
```

## Best Practices

### 1. Use Try-Finally for Guaranteed Cleanup

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Startup logic
        await initialize_services(app)
        yield
    finally:
        # Shutdown logic (always executes)
        await cleanup_services(app)
```

### 2. Initialize Services in Order, Cleanup in Reverse

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Initialize in dependency order
        await initialize_database(app)
        await initialize_cache(app)
        await initialize_models(app)
        yield
    finally:
        # Cleanup in reverse order
        await cleanup_models(app)
        await cleanup_cache(app)
        await cleanup_database(app)
```

### 3. Add Health Checks

```python
@asynccontextmanager
async def lifespan_with_health_checks(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Initialize services
        await initialize_services(app)
        
        # Run health checks
        health_status = await run_health_checks(app)
        if not health_status["healthy"]:
            raise RuntimeError(f"Health checks failed: {health_status['errors']}")
        
        yield
        
    finally:
        await cleanup_services(app)
```

### 4. Handle Background Tasks

```python
@asynccontextmanager
async def lifespan_with_background_tasks(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Start background tasks
        app.state.monitoring_task = asyncio.create_task(
            monitor_system_resources(app)
        )
        app.state.cleanup_task = asyncio.create_task(
            cleanup_old_files(app)
        )
        yield
        
    finally:
        # Stop background tasks
        if hasattr(app.state, 'monitoring_task'):
            app.state.monitoring_task.cancel()
            try:
                await app.state.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(app.state, 'cleanup_task'):
            app.state.cleanup_task.cancel()
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass
```

### 5. Add Retry Logic for Critical Services

```python
@asynccontextmanager
async def lifespan_with_retry(app: FastAPI) -> AsyncGenerator[None, None]:
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Startup attempt {attempt + 1}/{max_retries}")
            await initialize_services_with_retry(app, attempt)
            yield
            break
            
        except Exception as e:
            logger.error(f"Startup attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All startup attempts failed")
                raise
    
    # Shutdown
    await cleanup_services(app)
```

## Common Patterns

### Database Connection Management

```python
@asynccontextmanager
async def database_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Create database engine
        app.state.db_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost/ai_video",
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        
        # Create session maker
        app.state.db_sessionmaker = async_sessionmaker(
            app.state.db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with app.state.db_engine.begin() as conn:
            await conn.execute("SELECT 1")
        
        yield
        
    finally:
        if hasattr(app.state, 'db_engine'):
            await app.state.db_engine.dispose()
```

### Cache Management

```python
@asynccontextmanager
async def cache_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Initialize Redis
        app.state.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Test connection
        await app.state.redis.ping()
        
        yield
        
    finally:
        if hasattr(app.state, 'redis'):
            await app.state.redis.close()
```

### AI Model Management

```python
@asynccontextmanager
async def model_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        # Load models
        app.state.video_model = await load_video_model()
        app.state.text_model = await load_text_model()
        app.state.diffusion_pipeline = await load_diffusion_pipeline()
        
        yield
        
    finally:
        # Unload models
        if hasattr(app.state, 'video_model'):
            await unload_model(app.state.video_model)
        
        if hasattr(app.state, 'text_model'):
            await unload_model(app.state.text_model)
        
        if hasattr(app.state, 'diffusion_pipeline'):
            await unload_model(app.state.diffusion_pipeline)
```

## Testing Lifespan

### Unit Testing

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_lifespan():
    """Test lifespan context manager."""
    
    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.test_value = "initialized"
        yield
        app.state.test_value = "cleaned_up"
    
    app = FastAPI(lifespan=test_lifespan)
    
    with TestClient(app) as client:
        # Test that startup worked
        assert app.state.test_value == "initialized"
        
        # Make a request
        response = client.get("/")
        assert response.status_code == 200
    
    # Test that shutdown worked
    assert app.state.test_value == "cleaned_up"
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_database_lifespan():
    """Test database lifespan integration."""
    
    app = FastAPI(lifespan=database_lifespan)
    
    with TestClient(app) as client:
        # Test database connection
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["database"] == "healthy"
```

## Migration Checklist

### Before Migration
- [ ] Identify all `@app.on_event("startup")` decorators
- [ ] Identify all `@app.on_event("shutdown")` decorators
- [ ] Document current startup/shutdown logic
- [ ] Identify dependencies between services

### During Migration
- [ ] Create lifespan context manager
- [ ] Move startup logic to try block
- [ ] Move shutdown logic to finally block
- [ ] Add error handling
- [ ] Test startup failure scenarios
- [ ] Test shutdown cleanup

### After Migration
- [ ] Remove old `@app.on_event()` decorators
- [ ] Update FastAPI app to use lifespan
- [ ] Test application startup/shutdown
- [ ] Monitor for resource leaks
- [ ] Update documentation

## Summary

- **Replace** `@app.on_event("startup")` and `@app.on_event("shutdown")` with lifespan context managers
- **Use** `try/finally` for guaranteed cleanup
- **Initialize** services in dependency order
- **Cleanup** services in reverse order
- **Add** health checks and error handling
- **Test** startup failure and shutdown scenarios
- **Monitor** for resource leaks after migration

This approach provides better resource management, improved error handling, and more maintainable code. 