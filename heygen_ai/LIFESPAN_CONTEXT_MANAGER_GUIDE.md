# Lifespan Context Manager Guide

A comprehensive guide for implementing lifespan context managers in FastAPI applications, replacing the deprecated `app.on_event("startup")` and `app.on_event("shutdown")` approach.

## 🚀 Overview

This guide covers:
- **Why Use Lifespan Context Managers**: Benefits over deprecated event handlers
- **Modern FastAPI Patterns**: Current best practices for startup/shutdown
- **Resource Management**: Proper initialization and cleanup of resources
- **Error Handling**: Robust error handling during startup and shutdown
- **Monitoring and Metrics**: Health checks and application monitoring
- **Best Practices**: Patterns for production-ready applications

## 📋 Table of Contents

1. [Why Use Lifespan Context Managers](#why-use-lifespan-context-managers)
2. [Basic Lifespan Implementation](#basic-lifespan-implementation)
3. [Advanced Resource Management](#advanced-resource-management)
4. [Error Handling and Recovery](#error-handling-and-recovery)
5. [Monitoring and Health Checks](#monitoring-and-health-checks)
6. [Production Patterns](#production-patterns)
7. [Testing Lifespan Managers](#testing-lifespan-managers)
8. [Migration from app.on_event](#migration-from-apponevent)

## 🎯 Why Use Lifespan Context Managers

### Problems with `app.on_event`

```python
# ❌ Deprecated approach (don't use)
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize resources
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup resources
    pass
```

**Issues:**
- **Deprecated**: FastAPI no longer recommends this approach
- **No Error Handling**: Startup failures don't prevent shutdown
- **Resource Leaks**: Shutdown might not run if startup fails
- **No Context**: Limited access to application state
- **Testing Difficulties**: Hard to test startup/shutdown logic

### Benefits of Lifespan Context Managers

```python
# ✅ Modern approach (recommended)
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()

app = FastAPI(lifespan=lifespan)
```

**Benefits:**
- **Modern**: Current FastAPI best practice
- **Error Handling**: Proper exception handling and cleanup
- **Resource Management**: Guaranteed cleanup even on startup failure
- **Context Access**: Full access to application state
- **Testable**: Easy to test startup/shutdown logic
- **Type Safe**: Better type checking and IDE support

## 🔧 Basic Lifespan Implementation

### Simple Lifespan Manager

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
import structlog

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Basic lifespan context manager.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None: Application runs during yield
    """
    # Startup phase
    logger.info("Starting application")
    
    try:
        # Initialize resources
        await initialize_resources()
        
        # Application runs here
        yield
        
    finally:
        # Shutdown phase (always runs)
        logger.info("Shutting down application")
        await cleanup_resources()

async def initialize_resources():
    """Initialize application resources."""
    logger.info("Initializing resources")
    # Add your initialization logic here
    pass

async def cleanup_resources():
    """Cleanup application resources."""
    logger.info("Cleaning up resources")
    # Add your cleanup logic here
    pass

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
```

### Lifespan with State Management

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any
from fastapi import FastAPI
from datetime import datetime, timezone

class AppState:
    """Application state management."""
    
    def __init__(self):
        self.startup_time: Optional[datetime] = None
        self.shutdown_time: Optional[datetime] = None
        self.resources: Dict[str, Any] = {}
        self.is_shutting_down: bool = False

@asynccontextmanager
async def lifespan_with_state(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager with state management.
    """
    # Initialize state
    app.state.app_state = AppState()
    app.state.app_state.startup_time = datetime.now(timezone.utc)
    
    try:
        # Startup phase
        logger.info("Starting application")
        await initialize_resources_with_state(app.state.app_state)
        
        # Application runs here
        yield
        
    finally:
        # Shutdown phase
        app.state.app_state.shutdown_time = datetime.now(timezone.utc)
        app.state.app_state.is_shutting_down = True
        
        logger.info("Shutting down application")
        await cleanup_resources_with_state(app.state.app_state)

async def initialize_resources_with_state(state: AppState):
    """Initialize resources with state tracking."""
    logger.info("Initializing resources")
    
    # Example: Initialize database connection
    state.resources['database'] = await create_database_connection()
    
    # Example: Initialize Redis connection
    state.resources['redis'] = await create_redis_connection()
    
    logger.info("Resources initialized successfully")

async def cleanup_resources_with_state(state: AppState):
    """Cleanup resources with state tracking."""
    logger.info("Cleaning up resources")
    
    # Cleanup database
    if 'database' in state.resources:
        await state.resources['database'].close()
    
    # Cleanup Redis
    if 'redis' in state.resources:
        await state.resources['redis'].close()
    
    # Calculate uptime
    if state.startup_time and state.shutdown_time:
        uptime = state.shutdown_time - state.startup_time
        logger.info(f"Application uptime: {uptime.total_seconds():.2f} seconds")
    
    logger.info("Resources cleaned up successfully")
```

## 🏗️ Advanced Resource Management

### Database Connection Management

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import Optional

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[Any] = None
        self.sessionmaker: Optional[Any] = None
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        try:
            logger.info("Initializing database connections")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                pool_size=20,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Create session maker
            self.sessionmaker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup database connections."""
        try:
            logger.info("Closing database connections")
            
            if self.engine:
                await self.engine.dispose()
            
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        if not self.sessionmaker:
            raise RuntimeError("Database not initialized")
        
        return self.sessionmaker()

# Usage in lifespan
@asynccontextmanager
async def lifespan_with_database(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with database management."""
    
    # Initialize database manager
    db_manager = DatabaseManager("postgresql+asyncpg://user:password@localhost/db")
    
    try:
        # Initialize database
        await db_manager.initialize()
        
        # Add to app state
        app.state.db_manager = db_manager
        
        yield
        
    finally:
        # Cleanup database
        await db_manager.cleanup()
```

### Redis Connection Management

```python
import redis.asyncio as redis
from typing import Optional

class RedisManager:
    """Redis connection management."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            logger.info("Initializing Redis connection")
            
            # Create Redis client
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Test connection
            await self.client.ping()
            
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Redis connection", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup Redis connection."""
        try:
            logger.info("Closing Redis connection")
            
            if self.client:
                await self.client.close()
            
            logger.info("Redis connection closed successfully")
            
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.client:
            raise RuntimeError("Redis not initialized")
        
        return self.client

# Usage in lifespan
@asynccontextmanager
async def lifespan_with_redis(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with Redis management."""
    
    # Initialize Redis manager
    redis_manager = RedisManager("redis://localhost:6379")
    
    try:
        # Initialize Redis
        await redis_manager.initialize()
        
        # Add to app state
        app.state.redis_manager = redis_manager
        
        yield
        
    finally:
        # Cleanup Redis
        await redis_manager.cleanup()
```

### Background Task Management

```python
import asyncio
from typing import List

class BackgroundTaskManager:
    """Background task management."""
    
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
        self.is_shutting_down: bool = False
    
    def add_task(self, task: asyncio.Task) -> None:
        """Add background task for tracking."""
        self.tasks.append(task)
    
    async def start_health_check(self) -> None:
        """Start health check background task."""
        task = asyncio.create_task(self._health_check_loop())
        self.add_task(task)
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self.is_shutting_down:
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retry
    
    async def _perform_health_check(self) -> None:
        """Perform health check."""
        try:
            # Add your health check logic here
            logger.debug("Health check completed")
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def cleanup(self) -> None:
        """Cleanup background tasks."""
        try:
            logger.info("Cancelling background tasks")
            
            # Mark as shutting down
            self.is_shutting_down = True
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            logger.info("Background tasks cancelled successfully")
            
        except Exception as e:
            logger.error("Error cancelling background tasks", error=str(e))

# Usage in lifespan
@asynccontextmanager
async def lifespan_with_background_tasks(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with background task management."""
    
    # Initialize background task manager
    task_manager = BackgroundTaskManager()
    
    try:
        # Start background tasks
        await task_manager.start_health_check()
        
        # Add to app state
        app.state.task_manager = task_manager
        
        yield
        
    finally:
        # Cleanup background tasks
        await task_manager.cleanup()
```

## 🛡️ Error Handling and Recovery

### Robust Error Handling

```python
@asynccontextmanager
async def lifespan_with_error_handling(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan with comprehensive error handling.
    """
    startup_successful = False
    
    try:
        # Startup phase with error handling
        logger.info("Starting application")
        
        # Initialize resources with error handling
        await initialize_resources_safely()
        startup_successful = True
        
        logger.info("Application started successfully")
        yield
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
        
    finally:
        # Shutdown phase (always runs)
        logger.info("Shutting down application")
        
        try:
            if startup_successful:
                # Normal shutdown
                await cleanup_resources_safely()
            else:
                # Emergency cleanup
                await emergency_cleanup()
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))

async def initialize_resources_safely():
    """Initialize resources with error handling."""
    try:
        # Initialize database
        await initialize_database()
        
        # Initialize Redis
        await initialize_redis()
        
        # Initialize other resources
        await initialize_other_resources()
        
    except Exception as e:
        logger.error("Resource initialization failed", error=str(e))
        # Cleanup any partially initialized resources
        await emergency_cleanup()
        raise

async def cleanup_resources_safely():
    """Cleanup resources with error handling."""
    cleanup_errors = []
    
    try:
        await cleanup_database()
    except Exception as e:
        cleanup_errors.append(f"Database cleanup failed: {e}")
    
    try:
        await cleanup_redis()
    except Exception as e:
        cleanup_errors.append(f"Redis cleanup failed: {e}")
    
    try:
        await cleanup_other_resources()
    except Exception as e:
        cleanup_errors.append(f"Other resources cleanup failed: {e}")
    
    if cleanup_errors:
        logger.error("Cleanup errors occurred", errors=cleanup_errors)

async def emergency_cleanup():
    """Emergency cleanup for failed startup."""
    logger.warning("Performing emergency cleanup")
    
    # Add emergency cleanup logic here
    # This should be minimal and safe
    pass
```

### Graceful Shutdown with Timeout

```python
import signal
from typing import Optional

class GracefulShutdownManager:
    """Graceful shutdown management."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.is_shutting_down = False
        self._original_handlers = {}
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.is_shutting_down = True
        
        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, signal_handler)
        self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal_handler)
    
    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal with timeout."""
        try:
            while not self.is_shutting_down:
                await asyncio.sleep(1)
            
            # Wait for graceful shutdown timeout
            await asyncio.sleep(self.timeout)
            
        except asyncio.CancelledError:
            pass

# Usage in lifespan
@asynccontextmanager
async def lifespan_with_graceful_shutdown(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with graceful shutdown."""
    
    # Initialize shutdown manager
    shutdown_manager = GracefulShutdownManager(timeout=30)
    
    try:
        # Setup signal handlers
        shutdown_manager.setup_signal_handlers()
        
        # Initialize resources
        await initialize_resources()
        
        # Add to app state
        app.state.shutdown_manager = shutdown_manager
        
        yield
        
    finally:
        # Restore signal handlers
        shutdown_manager.restore_signal_handlers()
        
        # Cleanup resources
        await cleanup_resources()
```

## 📊 Monitoring and Health Checks

### Health Check Implementation

```python
from datetime import datetime, timezone
from typing import Dict, Any

class HealthCheckManager:
    """Health check and monitoring management."""
    
    def __init__(self):
        self.startup_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": self._get_uptime_seconds(),
                "checks": {}
            }
            
            # Database health check
            health_status["checks"]["database"] = await self._check_database()
            
            # Redis health check
            health_status["checks"]["redis"] = await self._check_redis()
            
            # Overall status
            all_healthy = all(check["status"] == "healthy" for check in health_status["checks"].values())
            health_status["status"] = "healthy" if all_healthy else "unhealthy"
            
            self.last_health_check = datetime.now(timezone.utc)
            self.health_status = health_status
            
            return health_status
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
    
    def _get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        if not self.startup_time:
            return 0.0
        return (datetime.now(timezone.utc) - self.startup_time).total_seconds()
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Add your database health check logic here
            return {"status": "healthy", "message": "Database connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            # Add your Redis health check logic here
            return {"status": "healthy", "message": "Redis connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_manager: HealthCheckManager = get_health_manager()
    return await health_manager.perform_health_check()
```

### Metrics and Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class MetricsManager:
    """Metrics and monitoring management."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
        self.request_duration = Histogram('api_request_duration_seconds', 'Request duration in seconds')
        self.active_connections = Gauge('api_active_connections', 'Number of active connections')
    
    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def record_request(self, method: str, endpoint: str, duration: float):
        """Record request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint).inc()
        self.request_duration.observe(duration)
    
    def increment_connections(self):
        """Increment active connections."""
        self.active_connections.inc()
    
    def decrement_connections(self):
        """Decrement active connections."""
        self.active_connections.dec()

# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for collecting metrics."""
    start_time = datetime.now(timezone.utc)
    
    # Increment active connections
    metrics_manager: MetricsManager = get_metrics_manager()
    metrics_manager.increment_connections()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        metrics_manager.record_request(
            method=request.method,
            endpoint=request.url.path,
            duration=duration
        )
        
        return response
        
    finally:
        # Decrement active connections
        metrics_manager.decrement_connections()
```

## 🏭 Production Patterns

### Complete Production Lifespan

```python
@asynccontextmanager
async def production_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Complete production lifespan manager.
    """
    # Initialize managers
    db_manager = DatabaseManager("postgresql+asyncpg://user:password@localhost/db")
    redis_manager = RedisManager("redis://localhost:6379")
    task_manager = BackgroundTaskManager()
    shutdown_manager = GracefulShutdownManager(timeout=30)
    health_manager = HealthCheckManager()
    metrics_manager = MetricsManager(port=8001)
    
    startup_successful = False
    
    try:
        # Setup signal handlers
        shutdown_manager.setup_signal_handlers()
        
        # Start metrics server
        metrics_manager.start_metrics_server()
        
        # Initialize resources
        await db_manager.initialize()
        await redis_manager.initialize()
        await task_manager.start_health_check()
        
        # Set startup time
        health_manager.startup_time = datetime.now(timezone.utc)
        
        # Add to app state
        app.state.db_manager = db_manager
        app.state.redis_manager = redis_manager
        app.state.task_manager = task_manager
        app.state.shutdown_manager = shutdown_manager
        app.state.health_manager = health_manager
        app.state.metrics_manager = metrics_manager
        
        startup_successful = True
        logger.info("Application started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise
        
    finally:
        # Restore signal handlers
        shutdown_manager.restore_signal_handlers()
        
        # Cleanup resources
        if startup_successful:
            await task_manager.cleanup()
            await redis_manager.cleanup()
            await db_manager.cleanup()
        else:
            await emergency_cleanup()
        
        logger.info("Application shut down successfully")

# Create production app
app = FastAPI(
    title="HeyGen AI API",
    description="Production-ready AI video generation API",
    version="1.0.0",
    lifespan=production_lifespan
)
```

### Configuration Management

```python
from pydantic import BaseSettings
from typing import Optional

class AppSettings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/heygen_ai"
    database_pool_size: int = 20
    database_max_overflow: int = 0
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 20
    
    # Metrics
    metrics_port: int = 8001
    
    # Health check
    health_check_interval: int = 30
    
    # Shutdown
    graceful_shutdown_timeout: int = 30
    
    class Config:
        env_file = ".env"

# Usage in lifespan
@asynccontextmanager
async def configurable_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with configuration management."""
    
    # Load settings
    settings = AppSettings()
    
    # Initialize managers with settings
    db_manager = DatabaseManager(settings.database_url)
    redis_manager = RedisManager(settings.redis_url)
    metrics_manager = MetricsManager(settings.metrics_port)
    
    try:
        # Initialize with settings
        await db_manager.initialize()
        await redis_manager.initialize()
        metrics_manager.start_metrics_server()
        
        # Add to app state
        app.state.settings = settings
        app.state.db_manager = db_manager
        app.state.redis_manager = redis_manager
        app.state.metrics_manager = metrics_manager
        
        yield
        
    finally:
        # Cleanup
        await redis_manager.cleanup()
        await db_manager.cleanup()
```

## 🧪 Testing Lifespan Managers

### Unit Testing

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

class TestLifespanManager:
    """Test lifespan manager."""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self):
        """Test successful lifespan startup."""
        app = FastAPI()
        
        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.startup_called = True
            yield
            app.state.shutdown_called = True
        
        app = FastAPI(lifespan=test_lifespan)
        
        with TestClient(app) as client:
            # Startup should have been called
            assert hasattr(app.state, 'startup_called')
            assert app.state.startup_called is True
            
            # Make a request
            response = client.get("/docs")
            assert response.status_code == 200
        
        # Shutdown should have been called
        assert hasattr(app.state, 'shutdown_called')
        assert app.state.shutdown_called is True
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_failure(self):
        """Test lifespan startup failure."""
        app = FastAPI()
        
        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.startup_called = True
            raise RuntimeError("Startup failed")
            yield
            app.state.shutdown_called = True
        
        app = FastAPI(lifespan=test_lifespan)
        
        with pytest.raises(RuntimeError, match="Startup failed"):
            with TestClient(app) as client:
                pass
    
    @pytest.mark.asyncio
    async def test_lifespan_shutdown_always_runs(self):
        """Test that shutdown always runs."""
        app = FastAPI()
        
        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            app.state.startup_called = True
            try:
                yield
            finally:
                app.state.shutdown_called = True
        
        app = FastAPI(lifespan=test_lifespan)
        
        with TestClient(app) as client:
            # Simulate application error
            app.state.error_occurred = True
        
        # Shutdown should have been called even with error
        assert hasattr(app.state, 'shutdown_called')
        assert app.state.shutdown_called is True
```

### Integration Testing

```python
class TestLifespanIntegration:
    """Integration tests for lifespan manager."""
    
    @pytest.mark.asyncio
    async def test_database_lifespan(self):
        """Test database lifespan integration."""
        app = FastAPI()
        
        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            # Mock database initialization
            app.state.db_initialized = True
            yield
            app.state.db_closed = True
        
        app = FastAPI(lifespan=test_lifespan)
        
        with TestClient(app) as client:
            assert app.state.db_initialized is True
            
            # Test database-dependent endpoint
            response = client.get("/users")
            assert response.status_code in [200, 404]  # Depending on implementation
        
        assert app.state.db_closed is True
    
    @pytest.mark.asyncio
    async def test_redis_lifespan(self):
        """Test Redis lifespan integration."""
        app = FastAPI()
        
        @asynccontextmanager
        async def test_lifespan(app: FastAPI):
            # Mock Redis initialization
            app.state.redis_initialized = True
            yield
            app.state.redis_closed = True
        
        app = FastAPI(lifespan=test_lifespan)
        
        with TestClient(app) as client:
            assert app.state.redis_initialized is True
            
            # Test Redis-dependent endpoint
            response = client.get("/cache/status")
            assert response.status_code in [200, 404]  # Depending on implementation
        
        assert app.state.redis_closed is True
```

## 🔄 Migration from app.on_event

### Before (Deprecated)

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize database
    app.state.database = await create_database_connection()
    
    # Initialize Redis
    app.state.redis = await create_redis_connection()
    
    # Start background tasks
    app.state.background_task = asyncio.create_task(background_worker())

@app.on_event("shutdown")
async def shutdown_event():
    # Stop background tasks
    if hasattr(app.state, 'background_task'):
        app.state.background_task.cancel()
    
    # Close Redis
    if hasattr(app.state, 'redis'):
        await app.state.redis.close()
    
    # Close database
    if hasattr(app.state, 'database'):
        await app.state.database.close()
```

### After (Modern)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.database = await create_database_connection()
    app.state.redis = await create_redis_connection()
    app.state.background_task = asyncio.create_task(background_worker())
    
    yield
    
    # Shutdown (always runs)
    if hasattr(app.state, 'background_task'):
        app.state.background_task.cancel()
    
    if hasattr(app.state, 'redis'):
        await app.state.redis.close()
    
    if hasattr(app.state, 'database'):
        await app.state.database.close()

app = FastAPI(lifespan=lifespan)
```

### Migration Checklist

1. **Replace `@app.on_event("startup")`** with startup logic in lifespan context manager
2. **Replace `@app.on_event("shutdown")`** with shutdown logic in lifespan context manager
3. **Add error handling** for startup failures
4. **Ensure cleanup runs** even if startup fails
5. **Update tests** to use new lifespan approach
6. **Add monitoring** and health checks
7. **Test graceful shutdown** behavior

## 📚 Additional Resources

- [FastAPI Lifespan Documentation](https://fastapi.tiangolo.com/advanced/events/)
- [Context Managers in Python](https://docs.python.org/3/library/contextlib.html)
- [Async Context Managers](https://docs.python.org/3/library/contextlib.html#contextlib.asynccontextmanager)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

## 🚀 Next Steps

1. **Replace existing `app.on_event`** with lifespan context managers
2. **Implement proper resource management** for database, Redis, and other services
3. **Add comprehensive error handling** and recovery mechanisms
4. **Implement health checks** and monitoring
5. **Add graceful shutdown** with signal handling
6. **Write comprehensive tests** for lifespan managers
7. **Monitor application** startup and shutdown in production

This lifespan context manager guide provides a comprehensive framework for implementing modern, production-ready startup and shutdown management in your HeyGen AI FastAPI application, replacing the deprecated `app.on_event` approach with robust, testable, and maintainable patterns. 