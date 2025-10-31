from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from .functional_fastapi_components import (
    import psutil
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🔄 FastAPI Lifespan Management with Context Managers
====================================================

Demonstrates proper use of lifespan context managers instead of:
- app.on_event("startup") - ❌ Deprecated approach
- app.on_event("shutdown") - ❌ Deprecated approach

Prefer lifespan context managers for:
- Resource initialization and cleanup
- Database connection management
- Cache initialization
- Background task management
- Graceful shutdown handling
- Dependency injection setup
"""



    TextAnalysisRequest, AnalysisResponse, AnalysisTypeEnum,
    OptimizationTierEnum, AnalysisStatusEnum
)

# ============================================================================
# Type Definitions
# ============================================================================

@dataclass
class AppState:
    """Application state for managing resources and configuration."""
    startup_time: datetime = field(default_factory=datetime.now)
    is_healthy: bool = True
    database_pool: Optional[async_sessionmaker] = None
    redis_client: Optional[redis.Redis] = None
    cache_manager: Optional[Any] = None
    background_tasks: List[asyncio.Task] = field(default_factory=list)
    shutdown_event: Optional[asyncio.Event] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

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

# ============================================================================
# Configuration Management
# ============================================================================

def load_config() -> AppConfig:
    """
    Load application configuration.
    
    Returns:
        Application configuration object
    """
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

# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[AppState, None]:
    """
    Lifespan context manager for managing application startup and shutdown.
    
    This replaces the deprecated app.on_event("startup") and app.on_event("shutdown")
    with a more robust and maintainable approach.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        Application state object
    """
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
    
    logger.info("🚀 Starting application", 
                app_name=config.app_name, 
                version=config.version)
    
    try:
        # Initialize database connection pool
        logger.info("📊 Initializing database connection pool")
        app_state.database_pool = await initialize_database_pool(config.database)
        
        # Initialize Redis client
        logger.info("🔴 Initializing Redis client")
        app_state.redis_client = await initialize_redis_client(config.redis)
        
        # Initialize cache manager
        logger.info("💾 Initializing cache manager")
        app_state.cache_manager = await initialize_cache_manager(app_state.redis_client)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(app_state.shutdown_event)
        
        # Start background tasks
        logger.info("🔄 Starting background tasks")
        await start_background_tasks(app_state)
        
        # Health check
        await perform_health_check(app_state)
        
        # Store app state
        app.state = app_state
        
        logger.info("✅ Application startup completed successfully")
        
        yield app_state
        
    except Exception as e:
        logger.error("❌ Application startup failed", error=str(e))
        app_state.is_healthy = False
        raise
    
    finally:
        # Graceful shutdown
        logger.info("🛑 Starting application shutdown")
        
        try:
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
            
            logger.info("✅ Application shutdown completed successfully")
            
        except Exception as e:
            logger.error("❌ Application shutdown failed", error=str(e))
            raise

# ============================================================================
# Database Management
# ============================================================================

async def initialize_database_pool(config: DatabaseConfig) -> async_sessionmaker:
    """
    Initialize database connection pool.
    
    Args:
        config: Database configuration
        
    Returns:
        Database session maker
    """
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
        
        logger.info("✅ Database connection pool initialized successfully")
        return session_maker
        
    except Exception as e:
        logger.error("❌ Database connection pool initialization failed", error=str(e))
        raise

async def close_database_pool(session_maker: async_sessionmaker):
    """
    Close database connection pool.
    
    Args:
        session_maker: Database session maker to close
    """
    logger = structlog.get_logger("database")
    
    try:
        # Close all sessions
        await session_maker.close_all()
        
        # Close engine
        await session_maker.kw["bind"].dispose()
        
        logger.info("✅ Database connection pool closed successfully")
        
    except Exception as e:
        logger.error("❌ Database connection pool closure failed", error=str(e))
        raise

# ============================================================================
# Redis Management
# ============================================================================

async def initialize_redis_client(config: RedisConfig) -> redis.Redis:
    """
    Initialize Redis client.
    
    Args:
        config: Redis configuration
        
    Returns:
        Redis client instance
    """
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
        
        logger.info("✅ Redis client initialized successfully")
        return redis_client
        
    except Exception as e:
        logger.error("❌ Redis client initialization failed", error=str(e))
        raise

async def close_redis_client(redis_client: redis.Redis):
    """
    Close Redis client.
    
    Args:
        redis_client: Redis client to close
    """
    logger = structlog.get_logger("redis")
    
    try:
        await redis_client.close()
        logger.info("✅ Redis client closed successfully")
        
    except Exception as e:
        logger.error("❌ Redis client closure failed", error=str(e))
        raise

# ============================================================================
# Cache Management
# ============================================================================

class CacheManager:
    """Cache manager for application caching."""
    
    def __init__(self, redis_client: redis.Redis):
        
    """__init__ function."""
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
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            self.logger.error("Cache delete failed", key=key, error=str(e))
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
    """
    Initialize cache manager.
    
    Args:
        redis_client: Redis client instance
        
    Returns:
        Cache manager instance
    """
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
        
        logger.info("✅ Cache manager initialized successfully")
        return cache_manager
        
    except Exception as e:
        logger.error("❌ Cache manager initialization failed", error=str(e))
        raise

async def close_cache_manager(cache_manager: CacheManager):
    """
    Close cache manager.
    
    Args:
        cache_manager: Cache manager to close
    """
    logger = structlog.get_logger("cache")
    
    try:
        # Cache manager cleanup (if needed)
        logger.info("✅ Cache manager closed successfully")
        
    except Exception as e:
        logger.error("❌ Cache manager closure failed", error=str(e))
        raise

# ============================================================================
# Background Tasks Management
# ============================================================================

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

async def start_background_tasks(app_state: AppState):
    """
    Start background tasks.
    
    Args:
        app_state: Application state
    """
    logger = structlog.get_logger("background_tasks")
    
    try:
        # Start health monitor task
        health_task = asyncio.create_task(health_monitor_task(app_state))
        app_state.background_tasks.append(health_task)
        
        # Start metrics collector task
        metrics_task = asyncio.create_task(metrics_collector_task(app_state))
        app_state.background_tasks.append(metrics_task)
        
        logger.info("✅ Background tasks started successfully", 
                   task_count=len(app_state.background_tasks))
        
    except Exception as e:
        logger.error("❌ Background tasks startup failed", error=str(e))
        raise

async def stop_background_tasks(app_state: AppState):
    """
    Stop background tasks gracefully.
    
    Args:
        app_state: Application state
    """
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
        
        logger.info("✅ Background tasks stopped successfully")
        
    except Exception as e:
        logger.error("❌ Background tasks shutdown failed", error=str(e))
        raise

# ============================================================================
# Health Checks
# ============================================================================

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

async def collect_system_metrics() -> Dict[str, Any]:
    """Collect system metrics."""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "uptime": (datetime.now() - datetime.now()).total_seconds()
    }

# ============================================================================
# Signal Handling
# ============================================================================

def setup_signal_handlers(shutdown_event: asyncio.Event):
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        shutdown_event: Event to signal shutdown
    """
    logger = structlog.get_logger("signal_handler")
    
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("✅ Signal handlers setup completed")

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_level: str):
    """
    Setup application logging.
    
    Args:
        log_level: Logging level
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

# ============================================================================
# FastAPI Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create FastAPI application with lifespan management.
    
    Returns:
        FastAPI application instance
    """
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

def setup_middleware(app: FastAPI, config: AppConfig):
    """Setup application middleware."""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.trusted_hosts
    )

def setup_routes(app: FastAPI):
    """Setup application routes."""
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "Blatam Academy NLP API", "status": "running"}
    
    @app.get("/health")
    async def health_check(app_state: AppState = Depends(get_app_state)):
        """Health check endpoint."""
        return {
            "status": "healthy" if app_state.is_healthy else "unhealthy",
            "uptime": (datetime.now() - app_state.startup_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "metrics": app_state.metrics
        }
    
    @app.get("/metrics")
    async def get_metrics(app_state: AppState = Depends(get_app_state)):
        """Get application metrics."""
        return {
            "metrics": app_state.metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/analyze")
    async def analyze_text(
        request: TextAnalysisRequest,
        app_state: AppState = Depends(get_app_state)
    ) -> AnalysisResponse:
        """Analyze text endpoint."""
        # Use app state resources
        if not app_state.is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable"
            )
        
        # Example analysis using app state resources
        analysis_data = {
            "id": 1,
            "text_content": request.text_content,
            "analysis_type": request.analysis_type,
            "status": "completed",
            "sentiment_score": 0.75,
            "processing_time_ms": 150.0,
            "created_at": datetime.now()
        }
        
        return AnalysisResponse(**analysis_data)

def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers."""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc) -> Any:
        """Global exception handler."""
        logger = structlog.get_logger("exception_handler")
        logger.error("Unhandled exception", 
                    error=str(exc), 
                    path=request.url.path)
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# ============================================================================
# Dependency Injection
# ============================================================================

def get_app_state(request) -> AppState:
    """Get application state from request."""
    return request.app.state

def get_database_session(app_state: AppState = Depends(get_app_state)) -> AsyncSession:
    """Get database session."""
    if not app_state.database_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    return app_state.database_pool()

def get_cache_manager(app_state: AppState = Depends(get_app_state)) -> CacheManager:
    """Get cache manager."""
    if not app_state.cache_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not available"
        )
    
    return app_state.cache_manager

# ============================================================================
# Application Instance
# ============================================================================

# Create application instance
app = create_app()

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    config = load_config()
    
    uvicorn.run(
        "lifespan_management:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        workers=config.workers
    ) 