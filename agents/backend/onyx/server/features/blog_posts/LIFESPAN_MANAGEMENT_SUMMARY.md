# 🔄 FastAPI Lifespan Management Summary

## Quick Reference

### ❌ Don't Use (Deprecated)
```python
# app.on_event approach - DEPRECATED
app = FastAPI()

@app.on_event("startup")
async def startup():
    app.state.db = await create_db_connection()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()
```

### ✅ Use This (Modern Approach)
```python
# lifespan context manager approach
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    # Startup
    app_state = AppState()
    app_state.db = await create_db_connection()
    app.state = app_state
    
    yield app_state
    
    # Shutdown (automatic cleanup)
    await app_state.db.close()

app = FastAPI(lifespan=lifespan)
```

## Key Benefits

### 1. Automatic Resource Management
- **Context managers** ensure cleanup even if startup fails
- **Exception safety** with proper error propagation
- **Resource tracking** for better debugging

### 2. Better Error Handling
- **Proper exception propagation** through the context manager
- **Graceful degradation** when resources fail to initialize
- **Comprehensive logging** of startup/shutdown events

### 3. Improved Testing
- **Easier mocking** of individual components
- **Isolated testing** of startup/shutdown logic
- **Better test coverage** with context manager patterns

### 4. Type Safety
- **Better type hints** with AsyncGenerator
- **IDE support** for autocomplete and error detection
- **Compile-time checks** for resource management

## Migration Guide

### Step 1: Replace app.on_event with lifespan

```python
# Before
app = FastAPI()

@app.on_event("startup")
async def startup():
    app.state.db = await create_db_pool()
    app.state.redis = await create_redis_client()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()
    await app.state.redis.close()

# After
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    app_state = AppState()
    
    try:
        app_state.db = await create_db_pool()
        app_state.redis = await create_redis_client()
        app.state = app_state
        yield app_state
    finally:
        await app_state.db.close()
        await app_state.redis.close()

app = FastAPI(lifespan=lifespan)
```

### Step 2: Add Application State Management

```python
@dataclass
class AppState:
    """Application state for managing resources."""
    startup_time: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    database_pool: Optional[Any] = None
    redis_client: Optional[Any] = None
    background_tasks: List[asyncio.Task] = field(default_factory=list)
    shutdown_event: Optional[asyncio.Event] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
```

### Step 3: Implement Resource Management

```python
async def initialize_resources(app_state: AppState):
    """Initialize all application resources."""
    # Database
    app_state.database_pool = await initialize_database_pool()
    
    # Redis
    app_state.redis_client = await initialize_redis_client()
    
    # Cache
    app_state.cache_manager = await initialize_cache_manager(app_state.redis_client)

async def cleanup_resources(app_state: AppState):
    """Cleanup all application resources."""
    # Stop background tasks
    await stop_background_tasks(app_state)
    
    # Close cache
    if app_state.cache_manager:
        await close_cache_manager(app_state.cache_manager)
    
    # Close Redis
    if app_state.redis_client:
        await close_redis_client(app_state.redis_client)
    
    # Close database
    if app_state.database_pool:
        await close_database_pool(app_state.database_pool)
```

## Complete Implementation

### Lifespan Context Manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    """Application lifespan manager."""
    # Load configuration
    config = load_config()
    
    # Initialize application state
    app_state = AppState(
        config=config.__dict__,
        shutdown_event=asyncio.Event()
    )
    
    # Setup logging
    setup_logging(config.log_level)
    logger = structlog.get_logger("lifespan")
    
    try:
        logger.info("🚀 Starting application", app_name=config.app_name)
        
        # Initialize resources
        await initialize_resources(app_state)
        
        # Setup signal handlers
        setup_signal_handlers(app_state.shutdown_event)
        
        # Start background tasks
        await start_background_tasks(app_state)
        
        # Health check
        await perform_health_check(app_state)
        
        # Store app state
        app.state = app_state
        
        logger.info("✅ Application startup completed")
        yield app_state
        
    except Exception as e:
        logger.error("❌ Application startup failed", error=str(e))
        app_state.is_healthy = False
        raise
        
    finally:
        logger.info("🛑 Starting application shutdown")
        
        try:
            await cleanup_resources(app_state)
            logger.info("✅ Application shutdown completed")
        except Exception as e:
            logger.error("❌ Application shutdown failed", error=str(e))
            raise
```

### Application Factory

```python
def create_app() -> FastAPI:
    """Create FastAPI application with lifespan."""
    # Load configuration
    config = load_config()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=config.app_name,
        version=config.version,
        debug=config.debug,
        lifespan=lifespan  # ✅ Use lifespan instead of on_event
    )
    
    # Add middleware
    setup_middleware(app, config)
    
    # Add routes
    setup_routes(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app
```

## Resource Management

### Database Connection Pool

```python
async def initialize_database_pool(config: DatabaseConfig) -> async_sessionmaker:
    """Initialize database connection pool."""
    engine = create_async_engine(
        config.url,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_pre_ping=config.pool_pre_ping
    )
    
    session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Test connection
    async with session_maker() as session:
        await session.execute("SELECT 1")
    
    return session_maker

async def close_database_pool(session_maker: async_sessionmaker):
    """Close database connection pool."""
    await session_maker.close_all()
    await session_maker.kw["bind"].dispose()
```

### Redis Client

```python
async def initialize_redis_client(config: RedisConfig) -> redis.Redis:
    """Initialize Redis client."""
    redis_client = redis.from_url(
        config.url,
        max_connections=config.max_connections,
        decode_responses=config.decode_responses
    )
    
    # Test connection
    await redis_client.ping()
    
    return redis_client

async def close_redis_client(redis_client: redis.Redis):
    """Close Redis client."""
    await redis_client.close()
```

## Background Tasks

### Health Monitor

```python
async def health_monitor_task(app_state: AppState):
    """Monitor application health."""
    while not app_state.shutdown_event.is_set():
        try:
            # Check all components
            db_healthy = await check_database_health(app_state.database_pool)
            redis_healthy = await check_redis_health(app_state.redis_client)
            cache_healthy = await check_cache_health(app_state.cache_manager)
            
            # Update health status
            app_state.is_healthy = all([db_healthy, redis_healthy, cache_healthy])
            
            # Update metrics
            app_state.metrics.update({
                "last_health_check": datetime.now().isoformat(),
                "overall_healthy": app_state.is_healthy
            })
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            app_state.is_healthy = False
            await asyncio.sleep(60)
```

### Task Management

```python
async def start_background_tasks(app_state: AppState):
    """Start background tasks."""
    # Health monitor
    health_task = asyncio.create_task(health_monitor_task(app_state))
    app_state.background_tasks.append(health_task)
    
    # Metrics collector
    metrics_task = asyncio.create_task(metrics_collector_task(app_state))
    app_state.background_tasks.append(metrics_task)

async def stop_background_tasks(app_state: AppState):
    """Stop background tasks gracefully."""
    # Signal shutdown
    app_state.shutdown_event.set()
    
    # Cancel tasks
    for task in app_state.background_tasks:
        if not task.done():
            task.cancel()
    
    # Wait for completion
    if app_state.background_tasks:
        await asyncio.gather(*app_state.background_tasks, return_exceptions=True)
```

## Health Monitoring

### Health Check Functions

```python
async def check_database_health(session_maker: async_sessionmaker) -> bool:
    """Check database health."""
    try:
        async with session_maker() as session:
            await session.execute("SELECT 1")
        return True
    except Exception:
        return False

async def check_redis_health(redis_client: redis.Redis) -> bool:
    """Check Redis health."""
    try:
        await redis_client.ping()
        return True
    except Exception:
        return False

async def perform_health_check(app_state: AppState):
    """Perform initial health check."""
    db_healthy = await check_database_health(app_state.database_pool)
    redis_healthy = await check_redis_health(app_state.redis_client)
    
    app_state.is_healthy = all([db_healthy, redis_healthy])
```

## Graceful Shutdown

### Signal Handling

```python
def setup_signal_handlers(shutdown_event: asyncio.Event):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
```

### Shutdown Process

```python
async def graceful_shutdown(app_state: AppState):
    """Perform graceful shutdown."""
    try:
        # Stop background tasks
        await stop_background_tasks(app_state)
        
        # Close resources
        await cleanup_resources(app_state)
        
    except Exception as e:
        logger.error("Shutdown failed", error=str(e))
        raise
```

## Dependency Injection

### App State Access

```python
def get_app_state(request) -> AppState:
    """Get application state from request."""
    return request.app.state

def get_database_session(app_state: AppState = Depends(get_app_state)) -> AsyncSession:
    """Get database session."""
    if not app_state.database_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    return app_state.database_pool()

def get_cache_manager(app_state: AppState = Depends(get_app_state)) -> CacheManager:
    """Get cache manager."""
    if not app_state.cache_manager:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    return app_state.cache_manager
```

### Route Handlers

```python
@app.get("/health")
async def health_check(app_state: AppState = Depends(get_app_state)):
    """Health check endpoint."""
    return {
        "status": "healthy" if app_state.is_healthy else "unhealthy",
        "uptime": (datetime.now() - app_state.startup_time).total_seconds(),
        "metrics": app_state.metrics
    }

@app.post("/analyze")
async def analyze_text(
    request: TextAnalysisRequest,
    db_session: AsyncSession = Depends(get_database_session),
    cache_manager: CacheManager = Depends(get_cache_manager),
    app_state: AppState = Depends(get_app_state)
) -> AnalysisResponse:
    """Analyze text with proper dependency injection."""
    
    if not app_state.is_healthy:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Use injected dependencies
    analysis = await perform_analysis(request, db_session, cache_manager)
    
    return analysis
```

## Testing

### Lifespan Testing

```python
@pytest.mark.asyncio
async def test_lifespan_context_manager():
    """Test lifespan context manager."""
    app = FastAPI()
    
    with patch('lifespan_management.initialize_resources'):
        with patch('lifespan_management.start_background_tasks'):
            with patch('lifespan_management.cleanup_resources'):
                async with lifespan(app) as app_state:
                    assert isinstance(app_state, AppState)
                    assert app_state.is_healthy is True
```

### Resource Testing

```python
@pytest.mark.asyncio
async def test_database_pool_initialization():
    """Test database pool initialization."""
    config = DatabaseConfig(url="postgresql://test")
    
    with patch('lifespan_management.create_async_engine'):
        with patch('lifespan_management.async_sessionmaker'):
            result = await initialize_database_pool(config)
            assert result is not None
```

### Integration Testing

```python
def test_fastapi_app_with_lifespan():
    """Test FastAPI app with lifespan integration."""
    app = create_app()
    
    # Verify lifespan was set
    assert app.router.lifespan_context is not None
    
    # Test with TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
```

## Best Practices

### ✅ Do This

```python
# Use lifespan context managers
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    try:
        await initialize_resources(app_state)
        yield app_state
    finally:
        await cleanup_resources(app_state)

# Proper error handling
try:
    await initialize_database_pool(config)
except Exception as e:
    logger.error("Database initialization failed", error=str(e))
    raise

# Health monitoring
async def health_monitor_task(app_state: AppState):
    while not app_state.shutdown_event.is_set():
        app_state.is_healthy = await check_all_components()
        await asyncio.sleep(30)

# Graceful shutdown
def setup_signal_handlers(shutdown_event: asyncio.Event):
    def signal_handler(signum, frame):
        shutdown_event.set()
    signal.signal(signal.SIGTERM, signal_handler)
```

### ❌ Don't Do This

```python
# Don't use app.on_event (deprecated)
@app.on_event("startup")
async def startup():
    app.state.db = await create_db_connection()

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

# Don't ignore errors
async def initialize_resources():
    app_state.db = await create_db_connection()  # No error handling

# Don't block the event loop
async def health_check():
    time.sleep(30)  # Use asyncio.sleep instead

# Don't forget cleanup
async def lifespan(app: FastAPI):
    app_state.db = await create_db_connection()
    yield app_state
    # Missing cleanup!
```

## Performance Considerations

### Resource Pooling
- **Database connections**: Use connection pools with appropriate sizes
- **Redis connections**: Configure max connections based on load
- **Background tasks**: Limit concurrent tasks to prevent resource exhaustion

### Health Check Frequency
- **Too frequent**: Can impact performance (every 5 seconds)
- **Too infrequent**: May miss issues (every 5 minutes)
- **Optimal**: Every 30-60 seconds for most applications

### Graceful Shutdown Timeout
- **Too short**: May not complete cleanup (5 seconds)
- **Too long**: May delay container restart (5 minutes)
- **Optimal**: 30-60 seconds for most applications

## Monitoring and Observability

### Metrics to Track
- **Startup time**: How long application takes to start
- **Shutdown time**: How long cleanup takes
- **Health check frequency**: How often health checks run
- **Resource usage**: Database connections, Redis connections
- **Error rates**: Startup/shutdown failures

### Logging Best Practices
```python
# Structured logging
logger.info("Application startup completed",
           startup_time=startup_duration,
           resources_initialized=resource_count)

logger.error("Resource initialization failed",
            resource="database",
            error=str(e),
            retry_count=retry_count)
```

## Checklist for Implementation

### Before Implementation
- [ ] Identify all resources that need initialization/cleanup
- [ ] Plan health monitoring strategy
- [ ] Design background task architecture
- [ ] Plan graceful shutdown process
- [ ] Design error handling strategy

### During Implementation
- [ ] Replace `app.on_event` with lifespan context manager
- [ ] Implement resource initialization and cleanup
- [ ] Add health monitoring for all components
- [ ] Implement background tasks for monitoring
- [ ] Add signal handlers for graceful shutdown
- [ ] Implement proper error handling and logging

### After Implementation
- [ ] Test startup and shutdown processes
- [ ] Verify health monitoring works correctly
- [ ] Test graceful shutdown with signals
- [ ] Monitor performance and resource usage
- [ ] Update documentation and runbooks

## Key Takeaways

1. **Use lifespan context managers** instead of `app.on_event("startup")` and `app.on_event("shutdown")`
2. **Implement proper resource management** with automatic cleanup
3. **Add health monitoring** for all critical components
4. **Use background tasks** for monitoring and maintenance
5. **Implement graceful shutdown** with signal handling
6. **Manage application state** centrally
7. **Use dependency injection** for accessing resources
8. **Test thoroughly** with proper mocking
9. **Monitor performance** and health metrics
10. **Document all lifecycle events**

## Related Files

- `lifespan_management.py` - Main implementation
- `test_lifespan_management.py` - Comprehensive test suite
- `LIFESPAN_MANAGEMENT_GUIDE.md` - Detailed guide
- `functional_fastapi_components.py` - Supporting components
- `sync_async_operations.py` - Resource management patterns

This summary provides a quick reference for implementing lifespan context managers effectively in FastAPI applications, ensuring robust resource management, proper error handling, and maintainable code. 