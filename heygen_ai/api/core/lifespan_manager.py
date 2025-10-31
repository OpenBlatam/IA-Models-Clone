from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime, timezone
import structlog
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import redis.asyncio as redis
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import uvicorn
    from fastapi import Request
    from fastapi import Request
    from fastapi import Request
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Lifespan Context Manager for HeyGen AI API
Modern approach to managing startup and shutdown events in FastAPI.
"""


logger = structlog.get_logger()

# =============================================================================
# Metrics and Monitoring
# =============================================================================

# Prometheus metrics
REQUEST_COUNT = Counter('heygen_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('heygen_api_request_duration_seconds', 'Request duration in seconds')
ACTIVE_CONNECTIONS = Gauge('heygen_api_active_connections', 'Number of active connections')
DATABASE_CONNECTIONS = Gauge('heygen_api_database_connections', 'Number of database connections')
REDIS_CONNECTIONS = Gauge('heygen_api_redis_connections', 'Number of Redis connections')

# =============================================================================
# Configuration
# =============================================================================

class LifespanConfig:
    """Configuration for lifespan management."""
    
    def __init__(self) -> Any:
        self.database_url: str = "postgresql+asyncpg://user:password@localhost/heygen_ai"
        self.redis_url: str = "redis://localhost:6379"
        self.prometheus_port: int = 8001
        self.health_check_interval: int = 30
        self.graceful_shutdown_timeout: int = 30
        self.max_connections: int = 20
        self.min_connections: int = 5

# =============================================================================
# Lifespan State Management
# =============================================================================

class LifespanState:
    """State management for application lifespan."""
    
    def __init__(self) -> Any:
        self.startup_time: Optional[datetime] = None
        self.shutdown_time: Optional[datetime] = None
        self.database_engine: Optional[Any] = None
        self.database_sessionmaker: Optional[Any] = None
        self.redis_client: Optional[redis.Redis] = None
        self.background_tasks: List[asyncio.Task] = []
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_shutting_down: bool = False
        self.active_connections: int = 0
        self.config: LifespanConfig = LifespanConfig()

# =============================================================================
# Database Management
# =============================================================================

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, state: LifespanState):
        
    """__init__ function."""
self.state = state
    
    async def initialize_database(self) -> None:
        """Initialize database connections."""
        try:
            logger.info("Initializing database connections")
            
            # Create async engine
            self.state.database_engine = create_async_engine(
                self.state.config.database_url,
                pool_size=self.state.config.max_connections,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Create session maker
            self.state.database_sessionmaker = async_sessionmaker(
                self.state.database_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.state.database_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            DATABASE_CONNECTIONS.set(self.state.config.max_connections)
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    async def close_database(self) -> None:
        """Close database connections."""
        try:
            logger.info("Closing database connections")
            
            if self.state.database_engine:
                await self.state.database_engine.dispose()
                DATABASE_CONNECTIONS.set(0)
            
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
    
    async def get_database_session(self) -> AsyncSession:
        """Get database session."""
        if not self.state.database_sessionmaker:
            raise RuntimeError("Database not initialized")
        
        return self.state.database_sessionmaker()

# =============================================================================
# Redis Management
# =============================================================================

class RedisManager:
    """Redis connection management."""
    
    def __init__(self, state: LifespanState):
        
    """__init__ function."""
self.state = state
    
    async def initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            logger.info("Initializing Redis connection")
            
            # Create Redis client
            self.state.redis_client = redis.from_url(
                self.state.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.state.config.max_connections
            )
            
            # Test connection
            await self.state.redis_client.ping()
            
            REDIS_CONNECTIONS.set(self.state.config.max_connections)
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Redis connection", error=str(e))
            raise
    
    async def close_redis(self) -> None:
        """Close Redis connection."""
        try:
            logger.info("Closing Redis connection")
            
            if self.state.redis_client:
                await self.state.redis_client.close()
                REDIS_CONNECTIONS.set(0)
            
            logger.info("Redis connection closed successfully")
            
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.state.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return self.state.redis_client

# =============================================================================
# Health Check Management
# =============================================================================

class HealthCheckManager:
    """Health check and monitoring management."""
    
    def __init__(self, state: LifespanState):
        
    """__init__ function."""
self.state = state
    
    async def start_health_check(self) -> None:
        """Start periodic health checks."""
        try:
            logger.info("Starting health check monitoring")
            
            self.state.health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
            
            logger.info("Health check monitoring started successfully")
            
        except Exception as e:
            logger.error("Failed to start health check monitoring", error=str(e))
            raise
    
    async def stop_health_check(self) -> None:
        """Stop health check monitoring."""
        try:
            logger.info("Stopping health check monitoring")
            
            if self.state.health_check_task:
                self.state.health_check_task.cancel()
                try:
                    await self.state.health_check_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Health check monitoring stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping health check monitoring", error=str(e))
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self.state.is_shutting_down:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.state.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retry
    
    async def _perform_health_check(self) -> None:
        """Perform health check."""
        try:
            # Check database
            if self.state.database_engine:
                async with self.state.database_engine.begin() as conn:
                    await conn.execute("SELECT 1")
            
            # Check Redis
            if self.state.redis_client:
                await self.state.redis_client.ping()
            
            logger.debug("Health check completed successfully")
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))

# =============================================================================
# Signal Handling
# =============================================================================

class SignalHandler:
    """Signal handling for graceful shutdown."""
    
    def __init__(self, state: LifespanState):
        
    """__init__ function."""
self.state = state
        self._original_handlers: Dict[int, Any] = {}
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            logger.info("Setting up signal handlers")
            
            # Store original handlers
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Signal handlers setup successfully")
            
        except Exception as e:
            logger.error("Failed to setup signal handlers", error=str(e))
            raise
    
    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            logger.info("Restoring signal handlers")
            
            for sig, handler in self._original_handlers.items():
                signal.signal(sig, handler)
            
            logger.info("Signal handlers restored successfully")
            
        except Exception as e:
            logger.error("Error restoring signal handlers", error=str(e))
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.state.is_shutting_down = True

# =============================================================================
# Main Lifespan Manager
# =============================================================================

class LifespanManager:
    """Main lifespan manager for the application."""
    
    def __init__(self) -> Any:
        self.state = LifespanState()
        self.database_manager = DatabaseManager(self.state)
        self.redis_manager = RedisManager(self.state)
        self.health_check_manager = HealthCheckManager(self.state)
        self.signal_handler = SignalHandler(self.state)
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """
        Lifespan context manager for FastAPI application.
        
        This replaces the deprecated app.on_event("startup") and app.on_event("shutdown")
        with a modern context manager approach.
        """
        try:
            # Startup phase
            await self._startup(app)
            yield
        finally:
            # Shutdown phase
            await self._shutdown(app)
    
    async def _startup(self, app: FastAPI) -> None:
        """Application startup logic."""
        try:
            logger.info("Starting HeyGen AI API application")
            
            # Record startup time
            self.state.startup_time = datetime.now(timezone.utc)
            
            # Setup signal handlers
            self.signal_handler.setup_signal_handlers()
            
            # Start Prometheus metrics server
            await self._start_prometheus_server()
            
            # Initialize database
            await self.database_manager.initialize_database()
            
            # Initialize Redis
            await self.redis_manager.initialize_redis()
            
            # Start health check monitoring
            await self.health_check_manager.start_health_check()
            
            # Add state to app
            app.state.lifespan_state = self.state
            app.state.database_manager = self.database_manager
            app.state.redis_manager = self.redis_manager
            
            # Log startup completion
            startup_duration = datetime.now(timezone.utc) - self.state.startup_time
            logger.info(
                "HeyGen AI API application started successfully",
                startup_duration_ms=startup_duration.total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error("Failed to start application", error=str(e))
            await self._shutdown(None)  # Cleanup on startup failure
            raise
    
    async def _shutdown(self, app: Optional[FastAPI]) -> None:
        """Application shutdown logic."""
        try:
            logger.info("Shutting down HeyGen AI API application")
            
            # Mark as shutting down
            self.state.is_shutting_down = True
            
            # Record shutdown time
            self.state.shutdown_time = datetime.now(timezone.utc)
            
            # Stop health check monitoring
            await self.health_check_manager.stop_health_check()
            
            # Cancel background tasks
            await self._cancel_background_tasks()
            
            # Close Redis connection
            await self.redis_manager.close_redis()
            
            # Close database connections
            await self.database_manager.close_database()
            
            # Restore signal handlers
            self.signal_handler.restore_signal_handlers()
            
            # Log shutdown completion
            if self.state.startup_time:
                uptime = self.state.shutdown_time - self.state.startup_time
                logger.info(
                    "HeyGen AI API application shut down successfully",
                    uptime_seconds=uptime.total_seconds()
                )
            
        except Exception as e:
            logger.error("Error during application shutdown", error=str(e))
    
    async def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics server."""
        try:
            logger.info(f"Starting Prometheus metrics server on port {self.state.config.prometheus_port}")
            
            # Start HTTP server for metrics
            start_http_server(self.state.config.prometheus_port)
            
            logger.info("Prometheus metrics server started successfully")
            
        except Exception as e:
            logger.error("Failed to start Prometheus metrics server", error=str(e))
            raise
    
    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        try:
            logger.info("Cancelling background tasks")
            
            # Cancel all background tasks
            for task in self.state.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.state.background_tasks:
                await asyncio.gather(*self.state.background_tasks, return_exceptions=True)
            
            logger.info("Background tasks cancelled successfully")
            
        except Exception as e:
            logger.error("Error cancelling background tasks", error=str(e))
    
    def add_background_task(self, task: asyncio.Task) -> None:
        """Add background task for tracking."""
        self.state.background_tasks.append(task)
    
    async def wait_for_shutdown(self, timeout: Optional[int] = None) -> None:
        """Wait for shutdown signal."""
        try:
            while not self.state.is_shutting_down:
                await asyncio.sleep(1)
            
            # Wait for graceful shutdown timeout
            if timeout:
                await asyncio.sleep(timeout)
            
        except asyncio.CancelledError:
            pass

# =============================================================================
# Dependency Injection
# =============================================================================

async def get_database_session() -> AsyncSession:
    """Dependency to get database session."""
    # This will be set during startup
    request: Request = Request()
    database_manager: DatabaseManager = request.app.state.database_manager
    return await database_manager.get_database_session()

async def get_redis_client() -> redis.Redis:
    """Dependency to get Redis client."""
    # This will be set during startup
    request: Request = Request()
    redis_manager: RedisManager = request.app.state.redis_manager
    return await redis_manager.get_redis_client()

def get_lifespan_state() -> LifespanState:
    """Dependency to get lifespan state."""
    # This will be set during startup
    request: Request = Request()
    return request.app.state.lifespan_state

# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with lifespan management."""
    
    # Create lifespan manager
    lifespan_manager = LifespanManager()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="HeyGen AI API",
        description="Modern AI-powered video generation API",
        version="1.0.0",
        lifespan=lifespan_manager.lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add middleware for metrics
    @app.middleware("http")
    async def metrics_middleware(request, call_next) -> Any:
        """Middleware for collecting metrics."""
        start_time = datetime.now(timezone.utc)
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path
            ).inc()
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            REQUEST_DURATION.observe(duration)
            
            return response
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        lifespan_state: LifespanState = get_lifespan_state()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (
                datetime.now(timezone.utc) - lifespan_state.startup_time
            ).total_seconds() if lifespan_state.startup_time else 0,
            "active_connections": lifespan_state.active_connections,
            "is_shutting_down": lifespan_state.is_shutting_down
        }
        
        return health_status
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    return app

# =============================================================================
# Main Application Entry Point
# =============================================================================

def main():
    """Main application entry point."""
    try:
        # Create application
        app = create_app()
        
        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application failed to start", error=str(e))
        sys.exit(1)

match __name__:
    case "__main__":
    main() 