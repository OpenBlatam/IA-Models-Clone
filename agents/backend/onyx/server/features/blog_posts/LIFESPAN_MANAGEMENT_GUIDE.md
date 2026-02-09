# 🔄 FastAPI Lifespan Management Guide

## Table of Contents

1. [Overview](#overview)
2. [Why Use Lifespan Context Managers?](#why-use-lifespan-context-managers)
3. [Migration from app.on_event](#migration-from-apponevent)
4. [Lifespan Context Manager Implementation](#lifespan-context-manager-implementation)
5. [Resource Management](#resource-management)
6. [Background Tasks](#background-tasks)
7. [Health Monitoring](#health-monitoring)
8. [Graceful Shutdown](#graceful-shutdown)
9. [Configuration Management](#configuration-management)
10. [Testing Strategies](#testing-strategies)
11. [Best Practices](#best-practices)
12. [Common Patterns](#common-patterns)
13. [Real-World Examples](#real-world-examples)

## Overview

FastAPI's lifespan context managers provide a modern, robust approach to managing application lifecycle events, replacing the deprecated `app.on_event("startup")` and `app.on_event("shutdown")` methods.

### Key Benefits

- **Better resource management**: Automatic cleanup with context managers
- **Improved error handling**: Proper exception propagation
- **Cleaner code**: More readable and maintainable
- **Better testing**: Easier to test and mock
- **Type safety**: Better type hints and IDE support

## Why Use Lifespan Context Managers?

### Problems with `app.on_event`

```python
# ❌ Deprecated approach - Don't use this
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize resources
    app.state.db = await create_database_connection()
    app.state.cache = await create_cache_connection()

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources
    await app.state.db.close()
    await app.state.cache.close()
```

**Issues:**
- No automatic cleanup if startup fails
- Hard to test individual components
- No proper error handling
- Difficult to manage dependencies
- No type safety

### Benefits of Lifespan Context Managers

```python
# ✅ Modern approach - Use this
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.db = await create_database_connection()
    app.state.cache = await create_cache_connection()
    
    yield app.state
    
    # Shutdown (automatic cleanup)
    await app.state.db.close()
    await app.state.cache.close()

app = FastAPI(lifespan=lifespan)
```

**Benefits:**
- Automatic cleanup with context managers
- Better error handling and propagation
- Easier testing and mocking
- Type-safe resource management
- Cleaner dependency injection

## Migration from app.on_event

### Step 1: Identify Current Event Handlers

```python
# Before: app.on_event approach
app = FastAPI()

@app.on_event("startup")
async def startup():
    # Database initialization
    app.state.db = await create_db_pool()
    
    # Cache initialization
    app.state.redis = await create_redis_client()
    
    # Background tasks
    app.state.background_tasks = []
    for task in get_background_tasks():
        app.state.background_tasks.append(asyncio.create_task(task()))

@app.on_event("shutdown")
async def shutdown():
    # Stop background tasks
    for task in app.state.background_tasks:
        task.cancel()
    
    # Close connections
    await app.state.db.close()
    await app.state.redis.close()
```

### Step 2: Convert to Lifespan Context Manager

```python
# After: lifespan context manager approach
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    # Initialize application state
    app_state = AppState()
    
    try:
        # Database initialization
        app_state.db = await create_db_pool()
        
        # Cache initialization
        app_state.redis = await create_redis_client()
        
        # Background tasks
        app_state.background_tasks = []
        for task in get_background_tasks():
            app_state.background_tasks.append(asyncio.create_task(task()))
        
        # Store state in app
        app.state = app_state
        
        yield app_state
        
    finally:
        # Automatic cleanup
        for task in app_state.background_tasks:
            task.cancel()
        
        await app_state.db.close()
        await app_state.redis.close()

app = FastAPI(lifespan=lifespan)
```

## Lifespan Context Manager Implementation

### Basic Structure

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
import structlog

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    """
    Lifespan context manager for application lifecycle management.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        Application state object
    """
    logger = structlog.get_logger("lifespan")
    
    # Initialize application state
    app_state = AppState()
    
    try:
        logger.info("🚀 Starting application")
        
        # Initialize resources
        await initialize_resources(app_state)
        
        # Start background tasks
        await start_background_tasks(app_state)
        
        # Health check
        await perform_health_check(app_state)
        
        # Store state in app
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
            # Stop background tasks
            await stop_background_tasks(app_state)
            
            # Cleanup resources
            await cleanup_resources(app_state)
            
            logger.info("✅ Application shutdown completed")
            
        except Exception as e:
            logger.error("❌ Application shutdown failed", error=str(e))
            raise
```

### Application State Management

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

@dataclass
class AppState:
    """Application state for managing resources and configuration."""
    startup_time: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    database_pool: Optional[Any] = None
    redis_client: Optional[Any] = None
    cache_manager: Optional[Any] = None
    background_tasks: List[asyncio.Task] = field(default_factory=list)
    shutdown_event: Optional[asyncio.Event] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
```

## Resource Management

### Database Connection Management

```python
async def initialize_database_pool(config: DatabaseConfig) -> async_sessionmaker:
    """Initialize database connection pool."""
    logger = structlog.get_logger("database")
    
    try:
        # Create async engine
        engine = create_async_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_pre_ping=config.pool_pre_ping,
            echo=config.echo,
            pool_recycle=config.pool_recycle
        )
        
        # Create session maker
        session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with session_maker() as session:
            await session.execute("SELECT 1")
        
        logger.info("✅ Database connection pool initialized")
        return session_maker
        
    except Exception as e:
        logger.error("❌ Database connection pool initialization failed", error=str(e))
        raise

async def close_database_pool(session_maker: async_sessionmaker):
    """Close database connection pool."""
    logger = structlog.get_logger("database")
    
    try:
        # Close all sessions
        await session_maker.close_all()
        
        # Close engine
        await session_maker.kw["bind"].dispose()
        
        logger.info("✅ Database connection pool closed")
        
    except Exception as e:
        logger.error("❌ Database connection pool closure failed", error=str(e))
        raise
```

### Redis Client Management

```python
async def initialize_redis_client(config: RedisConfig) -> redis.Redis:
    """Initialize Redis client."""
    logger = structlog.get_logger("redis")
    
    try:
        # Create Redis client
        redis_client = redis.from_url(
            config.url,
            max_connections=config.max_connections,
            decode_responses=config.decode_responses,
            health_check_interval=config.health_check_interval
        )
        
        # Test connection
        await redis_client.ping()
        
        logger.info("✅ Redis client initialized")
        return redis_client
        
    except Exception as e:
        logger.error("❌ Redis client initialization failed", error=str(e))
        raise

async def close_redis_client(redis_client: redis.Redis):
    """Close Redis client."""
    logger = structlog.get_logger("redis")
    
    try:
        await redis_client.close()
        logger.info("✅ Redis client closed")
        
    except Exception as e:
        logger.error("❌ Redis client closure failed", error=str(e))
        raise
```

### Cache Manager

```python
class CacheManager:
    """Cache manager for application caching."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = structlog.get_logger("cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            return value
        except Exception as e:
            self.logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache."""
        try:
            await self.redis.set(key, value, ex=ttl)
            return True
        except Exception as e:
            self.logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def health_check(self) -> bool:
        """Perform cache health check."""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            self.logger.error("Cache health check failed", error=str(e))
            return False

async def initialize_cache_manager(redis_client: redis.Redis) -> CacheManager:
    """Initialize cache manager."""
    logger = structlog.get_logger("cache")
    
    try:
        cache_manager = CacheManager(redis_client)
        
        # Test cache functionality
        test_key = "health_check"
        await cache_manager.set(test_key, "ok", ttl=60)
        test_value = await cache_manager.get(test_key)
        await cache_manager.delete(test_key)
        
        if test_value != "ok":
            raise Exception("Cache functionality test failed")
        
        logger.info("✅ Cache manager initialized")
        return cache_manager
        
    except Exception as e:
        logger.error("❌ Cache manager initialization failed", error=str(e))
        raise
```

## Background Tasks

### Health Monitor Task

```python
async def health_monitor_task(app_state: AppState):
    """Background task for monitoring application health."""
    logger = structlog.get_logger("health_monitor")
    
    while not app_state.shutdown_event.is_set():
        try:
            # Check database health
            db_healthy = await check_database_health(app_state.database_pool)
            
            # Check Redis health
            redis_healthy = await check_redis_health(app_state.redis_client)
            
            # Check cache health
            cache_healthy = await check_cache_health(app_state.cache_manager)
            
            # Update application health status
            app_state.is_healthy = all([db_healthy, redis_healthy, cache_healthy])
            
            # Update metrics
            app_state.metrics.update({
                "last_health_check": datetime.now().isoformat(),
                "database_healthy": db_healthy,
                "redis_healthy": redis_healthy,
                "cache_healthy": cache_healthy,
                "overall_healthy": app_state.is_healthy
            })
            
            if not app_state.is_healthy:
                logger.warning("Application health check failed", 
                              db_healthy=db_healthy,
                              redis_healthy=redis_healthy,
                              cache_healthy=cache_healthy)
            
            # Wait before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Health monitor task failed", error=str(e))
            app_state.is_healthy = False
            await asyncio.sleep(60)  # Wait longer on error
```

### Metrics Collector Task

```python
async def metrics_collector_task(app_state: AppState):
    """Background task for collecting application metrics."""
    logger = structlog.get_logger("metrics_collector")
    
    while not app_state.shutdown_event.is_set():
        try:
            # Collect system metrics
            metrics = await collect_system_metrics()
            
            # Store metrics in cache
            if app_state.cache_manager:
                await app_state.cache_manager.set(
                    "system_metrics",
                    metrics,
                    ttl=300
                )
            
            # Update app state metrics
            app_state.metrics.update(metrics)
            
            # Wait before next collection
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error("Metrics collector task failed", error=str(e))
            await asyncio.sleep(120)  # Wait longer on error
```

### Background Task Management

```python
async def start_background_tasks(app_state: AppState):
    """Start background tasks."""
    logger = structlog.get_logger("background_tasks")
    
    try:
        # Start health monitor task
        health_task = asyncio.create_task(health_monitor_task(app_state))
        app_state.background_tasks.append(health_task)
        
        # Start metrics collector task
        metrics_task = asyncio.create_task(metrics_collector_task(app_state))
        app_state.background_tasks.append(metrics_task)
        
        logger.info("✅ Background tasks started", 
                   task_count=len(app_state.background_tasks))
        
    except Exception as e:
        logger.error("❌ Background tasks startup failed", error=str(e))
        raise

async def stop_background_tasks(app_state: AppState):
    """Stop background tasks gracefully."""
    logger = structlog.get_logger("background_tasks")
    
    try:
        # Set shutdown event
        if app_state.shutdown_event:
            app_state.shutdown_event.set()
        
        # Cancel all background tasks
        for task in app_state.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if app_state.background_tasks:
            await asyncio.gather(*app_state.background_tasks, return_exceptions=True)
        
        logger.info("✅ Background tasks stopped")
        
    except Exception as e:
        logger.error("❌ Background tasks shutdown failed", error=str(e))
        raise
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

async def check_cache_health(cache_manager: CacheManager) -> bool:
    """Check cache health."""
    try:
        return await cache_manager.health_check()
    except Exception:
        return False

async def perform_health_check(app_state: AppState):
    """Perform initial health check."""
    logger = structlog.get_logger("health_check")
    
    try:
        # Check database
        db_healthy = await check_database_health(app_state.database_pool)
        
        # Check Redis
        redis_healthy = await check_redis_health(app_state.redis_client)
        
        # Check cache
        cache_healthy = await check_cache_health(app_state.cache_manager)
        
        # Update health status
        app_state.is_healthy = all([db_healthy, redis_healthy, cache_healthy])
        
        logger.info("Initial health check completed",
                   db_healthy=db_healthy,
                   redis_healthy=redis_healthy,
                   cache_healthy=cache_healthy,
                   overall_healthy=app_state.is_healthy)
        
    except Exception as e:
        logger.error("Initial health check failed", error=str(e))
        app_state.is_healthy = False
        raise
```

### System Metrics Collection

```python
async def collect_system_metrics() -> Dict[str, Any]:
    """Collect system metrics."""
    import psutil
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "uptime": (datetime.now() - datetime.now()).total_seconds()
    }
```

## Graceful Shutdown

### Signal Handling

```python
def setup_signal_handlers(shutdown_event: asyncio.Event):
    """Setup signal handlers for graceful shutdown."""
    logger = structlog.get_logger("signal_handler")
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("✅ Signal handlers setup completed")
```

### Shutdown Process

```python
async def graceful_shutdown(app_state: AppState):
    """Perform graceful shutdown."""
    logger = structlog.get_logger("shutdown")
    
    try:
        logger.info("🛑 Starting graceful shutdown")
        
        # Stop background tasks
        await stop_background_tasks(app_state)
        
        # Close cache manager
        if app_state.cache_manager:
            await close_cache_manager(app_state.cache_manager)
        
        # Close Redis client
        if app_state.redis_client:
            await close_redis_client(app_state.redis_client)
        
        # Close database pool
        if app_state.database_pool:
            await close_database_pool(app_state.database_pool)
        
        logger.info("✅ Graceful shutdown completed")
        
    except Exception as e:
        logger.error("❌ Graceful shutdown failed", error=str(e))
        raise
```

## Configuration Management

### Configuration Classes

```python
@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    echo: bool = False
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str
    max_connections: int = 20
    decode_responses: bool = True
    health_check_interval: int = 30

@dataclass
class AppConfig:
    """Application configuration."""
    app_name: str = "Blatam Academy NLP API"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig(
        url="postgresql+asyncpg://user:pass@localhost/db"
    ))
    redis: RedisConfig = field(default_factory=lambda: RedisConfig(
        url="redis://localhost:6379"
    ))
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    trusted_hosts: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
```

### Configuration Loading

```python
def load_config() -> AppConfig:
    """Load application configuration."""
    # In production, load from environment variables or config files
    return AppConfig(
        app_name="Blatam Academy NLP API",
        version="1.0.0",
        debug=False,
        database=DatabaseConfig(
            url="postgresql+asyncpg://user:pass@localhost/blatam_academy",
            pool_size=20,
            max_overflow=30
        ),
        redis=RedisConfig(
            url="redis://localhost:6379",
            max_connections=20
        ),
        cors_origins=["http://localhost:3000", "https://blatam.academy"],
        trusted_hosts=["localhost", "blatam.academy"],
        log_level="INFO"
    )
```

## Testing Strategies

### Testing Lifespan Context Manager

```python
@pytest.mark.asyncio
async def test_lifespan_context_manager():
    """Test lifespan context manager functionality."""
    app = FastAPI()
    
    # Mock all dependencies
    with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
        with patch('lifespan_management.setup_logging'):
            with patch('lifespan_management.initialize_database_pool') as mock_db:
                with patch('lifespan_management.initialize_redis_client') as mock_redis:
                    with patch('lifespan_management.initialize_cache_manager') as mock_cache:
                        with patch('lifespan_management.setup_signal_handlers'):
                            with patch('lifespan_management.start_background_tasks'):
                                with patch('lifespan_management.perform_health_check'):
                                    with patch('lifespan_management.stop_background_tasks'):
                                        with patch('lifespan_management.close_cache_manager'):
                                            with patch('lifespan_management.close_redis_client'):
                                                with patch('lifespan_management.close_database_pool'):
                                                    async with lifespan(app) as app_state:
                                                        # Verify app state was created
                                                        assert isinstance(app_state, AppState)
                                                        assert app_state.is_healthy is True
                                                        
                                                        # Verify dependencies were initialized
                                                        mock_db.assert_called_once()
                                                        mock_redis.assert_called_once()
                                                        mock_cache.assert_called_once()
```

### Testing Resource Management

```python
@pytest.mark.asyncio
async def test_database_pool_initialization():
    """Test database pool initialization."""
    config = DatabaseConfig(
        url="postgresql+asyncpg://test:test@localhost/testdb",
        pool_size=5,
        max_overflow=10
    )
    
    with patch('lifespan_management.create_async_engine') as mock_create_engine:
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        with patch('lifespan_management.async_sessionmaker') as mock_session_maker:
            mock_session_maker_instance = Mock()
            mock_session_maker.return_value = mock_session_maker_instance
            
            # Mock session context manager
            mock_session = AsyncMock()
            mock_session_maker_instance.return_value.__aenter__.return_value = mock_session
            mock_session_maker_instance.return_value.__aexit__.return_value = None
            
            result = await initialize_database_pool(config)
            
            assert result == mock_session_maker_instance
            mock_create_engine.assert_called_once()
            mock_session_maker.assert_called_once()
```

### Testing Background Tasks

```python
@pytest.mark.asyncio
async def test_background_task_management():
    """Test background task management."""
    app_state = AppState(shutdown_event=asyncio.Event())
    
    # Test starting tasks
    with patch('lifespan_management.health_monitor_task') as mock_health_task:
        with patch('lifespan_management.metrics_collector_task') as mock_metrics_task:
            await start_background_tasks(app_state)
            
            assert len(app_state.background_tasks) == 2
            assert all(isinstance(task, asyncio.Task) for task in app_state.background_tasks)
    
    # Test stopping tasks
    mock_task1 = AsyncMock()
    mock_task1.done.return_value = False
    mock_task2 = AsyncMock()
    mock_task2.done.return_value = False
    app_state.background_tasks = [mock_task1, mock_task2]
    
    await stop_background_tasks(app_state)
    
    # Verify shutdown event was set
    assert app_state.shutdown_event.is_set()
    
    # Verify tasks were cancelled
    mock_task1.cancel.assert_called_once()
    mock_task2.cancel.assert_called_once()
```

## Best Practices

### 1. Use Context Managers for Resource Management

```python
# ✅ Good: Use context managers
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    try:
        # Initialize resources
        app_state = await initialize_resources()
        yield app_state
    finally:
        # Automatic cleanup
        await cleanup_resources(app_state)

# ❌ Bad: Manual resource management
@app.on_event("startup")
async def startup():
    app.state.resources = await create_resources()

@app.on_event("shutdown")
async def shutdown():
    await cleanup_resources(app.state.resources)
```

### 2. Proper Error Handling

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    app_state = AppState()
    
    try:
        # Initialize resources
        await initialize_resources(app_state)
        yield app_state
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        app_state.is_healthy = False
        raise
    finally:
        # Always cleanup, even if startup failed
        await cleanup_resources(app_state)
```

### 3. Health Monitoring

```python
async def health_monitor_task(app_state: AppState):
    """Monitor application health."""
    while not app_state.shutdown_event.is_set():
        try:
            # Check all components
            db_healthy = await check_database_health(app_state.database_pool)
            redis_healthy = await check_redis_health(app_state.redis_client)
            
            # Update overall health
            app_state.is_healthy = all([db_healthy, redis_healthy])
            
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

### 4. Graceful Shutdown

```python
def setup_signal_handlers(shutdown_event: asyncio.Event):
    """Setup signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

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

### 5. Configuration Management

```python
@dataclass
class AppConfig:
    """Type-safe configuration."""
    app_name: str
    version: str
    database: DatabaseConfig
    redis: RedisConfig
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            app_name=os.getenv("APP_NAME", "Default API"),
            version=os.getenv("APP_VERSION", "1.0.0"),
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL", "postgresql://localhost/db")
            ),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        )
```

## Common Patterns

### 1. Resource Initialization Pattern

```python
async def initialize_resources(app_state: AppState):
    """Initialize all application resources."""
    logger = structlog.get_logger("resources")
    
    # Initialize database
    logger.info("Initializing database")
    app_state.database_pool = await initialize_database_pool(app_state.config.database)
    
    # Initialize Redis
    logger.info("Initializing Redis")
    app_state.redis_client = await initialize_redis_client(app_state.config.redis)
    
    # Initialize cache
    logger.info("Initializing cache")
    app_state.cache_manager = await initialize_cache_manager(app_state.redis_client)
    
    logger.info("All resources initialized successfully")
```

### 2. Health Check Pattern

```python
async def perform_health_check(app_state: AppState):
    """Perform comprehensive health check."""
    logger = structlog.get_logger("health_check")
    
    health_checks = {
        "database": check_database_health(app_state.database_pool),
        "redis": check_redis_health(app_state.redis_client),
        "cache": check_cache_health(app_state.cache_manager)
    }
    
    results = await asyncio.gather(*health_checks.values(), return_exceptions=True)
    
    # Process results
    health_status = {}
    for name, result in zip(health_checks.keys(), results):
        if isinstance(result, Exception):
            health_status[name] = False
            logger.error(f"{name} health check failed", error=str(result))
        else:
            health_status[name] = result
    
    # Update overall health
    app_state.is_healthy = all(health_status.values())
    app_state.metrics.update(health_status)
    
    logger.info("Health check completed", health_status=health_status)
```

### 3. Background Task Pattern

```python
async def start_background_tasks(app_state: AppState):
    """Start all background tasks."""
    logger = structlog.get_logger("background_tasks")
    
    tasks = [
        ("health_monitor", health_monitor_task(app_state)),
        ("metrics_collector", metrics_collector_task(app_state)),
        ("cleanup", cleanup_task(app_state))
    ]
    
    for name, task_coro in tasks:
        try:
            task = asyncio.create_task(task_coro, name=name)
            app_state.background_tasks.append(task)
            logger.info(f"Started background task: {name}")
        except Exception as e:
            logger.error(f"Failed to start background task: {name}", error=str(e))
            raise
    
    logger.info(f"Started {len(app_state.background_tasks)} background tasks")
```

### 4. Graceful Shutdown Pattern

```python
async def graceful_shutdown(app_state: AppState):
    """Perform graceful shutdown."""
    logger = structlog.get_logger("shutdown")
    
    shutdown_steps = [
        ("background_tasks", stop_background_tasks(app_state)),
        ("cache_manager", close_cache_manager(app_state.cache_manager)),
        ("redis_client", close_redis_client(app_state.redis_client)),
        ("database_pool", close_database_pool(app_state.database_pool))
    ]
    
    for name, step_coro in shutdown_steps:
        try:
            await step_coro
            logger.info(f"Completed shutdown step: {name}")
        except Exception as e:
            logger.error(f"Failed shutdown step: {name}", error=str(e))
            # Continue with other steps even if one fails
    
    logger.info("Graceful shutdown completed")
```

## Real-World Examples

### Example 1: Complete Application with Lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from typing import AsyncGenerator

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
        app_state.database_pool = await initialize_database_pool(config.database)
        app_state.redis_client = await initialize_redis_client(config.redis)
        app_state.cache_manager = await initialize_cache_manager(app_state.redis_client)
        
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
            await graceful_shutdown(app_state)
            logger.info("✅ Application shutdown completed")
        except Exception as e:
            logger.error("❌ Application shutdown failed", error=str(e))
            raise

# Create FastAPI app with lifespan
app = FastAPI(
    title="Blatam Academy NLP API",
    version="1.0.0",
    lifespan=lifespan
)

# Add routes
@app.get("/health")
async def health_check(app_state: AppState = Depends(get_app_state)):
    """Health check endpoint."""
    return {
        "status": "healthy" if app_state.is_healthy else "unhealthy",
        "uptime": (datetime.now() - app_state.startup_time).total_seconds(),
        "metrics": app_state.metrics
    }

@app.get("/metrics")
async def get_metrics(app_state: AppState = Depends(get_app_state)):
    """Get application metrics."""
    return {
        "metrics": app_state.metrics,
        "timestamp": datetime.now().isoformat()
    }
```

### Example 2: Dependency Injection with App State

```python
def get_app_state(request) -> AppState:
    """Get application state from request."""
    return request.app.state

def get_database_session(app_state: AppState = Depends(get_app_state)) -> AsyncSession:
    """Get database session."""
    if not app_state.database_pool:
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    
    return app_state.database_pool()

def get_cache_manager(app_state: AppState = Depends(get_app_state)) -> CacheManager:
    """Get cache manager."""
    if not app_state.cache_manager:
        raise HTTPException(
            status_code=503,
            detail="Cache not available"
        )
    
    return app_state.cache_manager

@app.post("/analyze")
async def analyze_text(
    request: TextAnalysisRequest,
    db_session: AsyncSession = Depends(get_database_session),
    cache_manager: CacheManager = Depends(get_cache_manager),
    app_state: AppState = Depends(get_app_state)
) -> AnalysisResponse:
    """Analyze text with proper dependency injection."""
    
    # Check application health
    if not app_state.is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable"
        )
    
    # Use injected dependencies
    analysis = await perform_analysis(request, db_session, cache_manager)
    
    return analysis
```

### Example 3: Background Task with Health Monitoring

```python
async def health_monitor_task(app_state: AppState):
    """Monitor application health in background."""
    logger = structlog.get_logger("health_monitor")
    
    while not app_state.shutdown_event.is_set():
        try:
            # Perform health checks
            health_results = await asyncio.gather(
                check_database_health(app_state.database_pool),
                check_redis_health(app_state.redis_client),
                check_cache_health(app_state.cache_manager),
                return_exceptions=True
            )
            
            # Process results
            db_healthy, redis_healthy, cache_healthy = health_results
            
            # Handle exceptions
            if isinstance(db_healthy, Exception):
                logger.error("Database health check failed", error=str(db_healthy))
                db_healthy = False
            
            if isinstance(redis_healthy, Exception):
                logger.error("Redis health check failed", error=str(redis_healthy))
                redis_healthy = False
            
            if isinstance(cache_healthy, Exception):
                logger.error("Cache health check failed", error=str(cache_healthy))
                cache_healthy = False
            
            # Update health status
            app_state.is_healthy = all([db_healthy, redis_healthy, cache_healthy])
            
            # Update metrics
            app_state.metrics.update({
                "last_health_check": datetime.now().isoformat(),
                "database_healthy": bool(db_healthy),
                "redis_healthy": bool(redis_healthy),
                "cache_healthy": bool(cache_healthy),
                "overall_healthy": app_state.is_healthy
            })
            
            # Log health status
            if not app_state.is_healthy:
                logger.warning("Application health degraded",
                              db_healthy=db_healthy,
                              redis_healthy=redis_healthy,
                              cache_healthy=cache_healthy)
            
            # Wait before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Health monitor task failed", error=str(e))
            app_state.is_healthy = False
            await asyncio.sleep(60)
```

## Summary

### Key Takeaways

1. **Use lifespan context managers** instead of `app.on_event("startup")` and `app.on_event("shutdown")`
2. **Implement proper resource management** with automatic cleanup
3. **Add health monitoring** for all critical components
4. **Use background tasks** for monitoring and maintenance
5. **Implement graceful shutdown** with signal handling
6. **Manage application state** centrally
7. **Use dependency injection** for accessing resources
8. **Test thoroughly** with proper mocking

### Migration Checklist

- [ ] Replace `app.on_event("startup")` with lifespan context manager
- [ ] Replace `app.on_event("shutdown")` with lifespan cleanup
- [ ] Implement proper resource initialization and cleanup
- [ ] Add health monitoring for all components
- [ ] Implement graceful shutdown handling
- [ ] Add background tasks for monitoring
- [ ] Update dependency injection to use app state
- [ ] Add comprehensive error handling
- [ ] Write tests for lifespan functionality
- [ ] Update documentation

### Best Practices Checklist

- [ ] Use context managers for resource management
- [ ] Implement proper error handling and logging
- [ ] Add health monitoring for all critical components
- [ ] Use background tasks for long-running operations
- [ ] Implement graceful shutdown with signal handling
- [ ] Centralize application state management
- [ ] Use dependency injection for resource access
- [ ] Add comprehensive testing
- [ ] Monitor performance and health metrics
- [ ] Document all lifecycle events

This guide provides a comprehensive approach to using lifespan context managers effectively in FastAPI applications, ensuring robust resource management, proper error handling, and maintainable code. 