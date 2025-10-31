from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import torch
import psutil
import signal
import sys
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🔄 LIFESPAN PATTERNS - MODERN FASTAPI STARTUP/SHUTDOWN
=====================================================

Replace deprecated @app.on_event() decorators with lifespan context managers
for better resource management, error handling, and maintainability.
"""


logger = logging.getLogger(__name__)

# ============================================================================
# ❌ DEPRECATED PATTERN - Don't use this
# ============================================================================

"""
# OLD WAY - Don't use this pattern
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    
    """startup_event function."""
# Initialize resources
    app.state.redis = redis.Redis()
    app.state.db = create_engine("postgresql://...")
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    
    """shutdown_event function."""
# Cleanup resources
    await app.state.redis.close()
    app.state.db.dispose()
    logger.info("Application shutdown")
"""

# ============================================================================
# ✅ MODERN PATTERN - Use lifespan context managers
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Modern lifespan context manager for FastAPI application lifecycle.
    
    This replaces @app.on_event("startup") and @app.on_event("shutdown")
    with a single, more maintainable context manager.
    """
    
    # ============================================================================
    # STARTUP PHASE
    # ============================================================================
    
    logger.info("🚀 Starting AI Video Application...")
    start_time = time.time()
    
    try:
        # Initialize application state
        app.state.startup_time = start_time
        app.state.is_healthy = True
        app.state.active_requests = 0
        
        # Initialize database connection
        logger.info("📊 Initializing database connection...")
        app.state.db_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost/ai_video",
            echo=False,
            pool_size=10,
            max_overflow=20
        )
        app.state.db_sessionmaker = async_sessionmaker(
            app.state.db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test database connection
        async with app.state.db_engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database connection established")
        
        # Initialize Redis connection
        logger.info("🔴 Initializing Redis connection...")
        app.state.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Test Redis connection
        await app.state.redis.ping()
        logger.info("✅ Redis connection established")
        
        # Initialize AI models
        logger.info("🤖 Loading AI models...")
        app.state.video_model = await load_video_model()
        app.state.text_model = await load_text_model()
        logger.info("✅ AI models loaded successfully")
        
        # Initialize performance monitoring
        logger.info("📈 Initializing performance monitoring...")
        app.state.performance_monitor = PerformanceMonitor()
        await app.state.performance_monitor.start()
        logger.info("✅ Performance monitoring started")
        
        # Initialize background tasks
        logger.info("🔄 Starting background task manager...")
        app.state.task_manager = BackgroundTaskManager()
        await app.state.task_manager.start()
        logger.info("✅ Background task manager started")
        
        # Set up signal handlers for graceful shutdown
        setup_signal_handlers(app)
        
        startup_time = time.time() - start_time
        logger.info(f"🎉 Application startup completed in {startup_time:.2f}s")
        
        # Yield control to FastAPI
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # ============================================================================
        # SHUTDOWN PHASE
        # ============================================================================
        
        logger.info("🛑 Shutting down AI Video Application...")
        shutdown_start = time.time()
        
        try:
            # Stop background tasks
            if hasattr(app.state, 'task_manager'):
                logger.info("🔄 Stopping background task manager...")
                await app.state.task_manager.stop()
                logger.info("✅ Background task manager stopped")
            
            # Stop performance monitoring
            if hasattr(app.state, 'performance_monitor'):
                logger.info("📈 Stopping performance monitoring...")
                await app.state.performance_monitor.stop()
                logger.info("✅ Performance monitoring stopped")
            
            # Close AI models
            if hasattr(app.state, 'video_model'):
                logger.info("🤖 Unloading AI models...")
                await unload_video_model(app.state.video_model)
                await unload_text_model(app.state.text_model)
                logger.info("✅ AI models unloaded")
            
            # Close Redis connection
            if hasattr(app.state, 'redis'):
                logger.info("🔴 Closing Redis connection...")
                await app.state.redis.close()
                logger.info("✅ Redis connection closed")
            
            # Close database connection
            if hasattr(app.state, 'db_engine'):
                logger.info("📊 Closing database connection...")
                await app.state.db_engine.dispose()
                logger.info("✅ Database connection closed")
            
            shutdown_time = time.time() - shutdown_start
            logger.info(f"✅ Application shutdown completed in {shutdown_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# FASTAPI APPLICATION WITH LIFESPAN
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with lifespan context manager."""
    
    app = FastAPI(
        title="AI Video Generation API",
        description="Modern FastAPI application with lifespan management",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan  # Use lifespan context manager
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return app

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self) -> Any:
        self.is_running = False
        self.metrics = {}
    
    async def start(self) -> Any:
        """Start performance monitoring."""
        self.is_running = True
        logger.info("Performance monitoring started")
    
    async def stop(self) -> Any:
        """Stop performance monitoring."""
        self.is_running = False
        logger.info("Performance monitoring stopped")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

class BackgroundTaskManager:
    """Background task management system."""
    
    def __init__(self) -> Any:
        self.tasks = set()
        self.is_running = False
    
    async def start(self) -> Any:
        """Start background task manager."""
        self.is_running = True
        logger.info("Background task manager started")
    
    async def stop(self) -> Any:
        """Stop all background tasks."""
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Background task manager stopped")
    
    async def add_task(self, coro) -> Any:
        """Add a new background task."""
        if self.is_running:
            task = asyncio.create_task(coro)
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return task

def setup_signal_handlers(app: FastAPI):
    """Set up signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # The lifespan context manager will handle cleanup
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# MODEL LOADING FUNCTIONS (Simplified)
# ============================================================================

async def load_video_model():
    """Load video generation model."""
    logger.info("Loading video generation model...")
    await asyncio.sleep(1)  # Simulate model loading
    return {"model": "video_generation_model", "loaded": True}

async def load_text_model():
    """Load text processing model."""
    logger.info("Loading text processing model...")
    await asyncio.sleep(0.5)  # Simulate model loading
    return {"model": "text_processing_model", "loaded": True}

async def unload_video_model(model) -> Any:
    """Unload video generation model."""
    logger.info("Unloading video generation model...")
    await asyncio.sleep(0.5)  # Simulate model unloading

async def unload_text_model(model) -> Any:
    """Unload text processing model."""
    logger.info("Unloading text processing model...")
    await asyncio.sleep(0.3)  # Simulate model unloading

# ============================================================================
# ADVANCED LIFESPAN PATTERNS
# ============================================================================

@asynccontextmanager
async def lifespan_with_health_checks(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with health checks and recovery."""
    
    # Startup
    logger.info("Starting application with health checks...")
    
    try:
        # Initialize core services
        await initialize_core_services(app)
        
        # Run health checks
        health_status = await run_health_checks(app)
        if not health_status["healthy"]:
            raise RuntimeError(f"Health checks failed: {health_status['errors']}")
        
        # Start monitoring
        await start_monitoring(app)
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        await cleanup_services(app)

@asynccontextmanager
async def lifespan_with_retry(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan with retry logic for critical services."""
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Startup attempt {attempt + 1}/{max_retries}")
            
            # Initialize services
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

# ============================================================================
# HELPER FUNCTIONS FOR ADVANCED PATTERNS
# ============================================================================

async def initialize_core_services(app: FastAPI):
    """Initialize core application services."""
    logger.info("Initializing core services...")
    
    # Database
    app.state.db = await create_database_connection()
    
    # Cache
    app.state.cache = await create_cache_connection()
    
    # Models
    app.state.models = await load_models()
    
    logger.info("Core services initialized")

async def run_health_checks(app: FastAPI) -> Dict[str, Any]:
    """Run health checks on all services."""
    logger.info("Running health checks...")
    
    errors = []
    
    # Database health check
    try:
        await app.state.db.execute("SELECT 1")
    except Exception as e:
        errors.append(f"Database: {e}")
    
    # Cache health check
    try:
        await app.state.cache.ping()
    except Exception as e:
        errors.append(f"Cache: {e}")
    
    # Model health check
    try:
        # Check if models are loaded
        if not app.state.models:
            errors.append("Models not loaded")
    except Exception as e:
        errors.append(f"Models: {e}")
    
    healthy = len(errors) == 0
    
    return {
        "healthy": healthy,
        "errors": errors,
        "timestamp": time.time()
    }

async def start_monitoring(app: FastAPI):
    """Start application monitoring."""
    logger.info("Starting application monitoring...")
    
    # Start background monitoring task
    app.state.monitoring_task = asyncio.create_task(
        monitor_application(app)
    )

async def monitor_application(app: FastAPI):
    """Background application monitoring."""
    while True:
        try:
            # Collect metrics
            metrics = await collect_application_metrics(app)
            
            # Log metrics
            logger.info(f"Application metrics: {metrics}")
            
            # Check for issues
            if metrics["memory_usage"] > 90:
                logger.warning("High memory usage detected")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(5)

async def initialize_services_with_retry(app: FastAPI, attempt: int):
    """Initialize services with retry logic."""
    logger.info(f"Initializing services (attempt {attempt + 1})...")
    
    # Initialize services with exponential backoff
    delay = 2 ** attempt
    
    # Database
    try:
        app.state.db = await create_database_connection()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        await asyncio.sleep(delay)
        raise
    
    # Cache
    try:
        app.state.cache = await create_cache_connection()
    except Exception as e:
        logger.error(f"Cache initialization failed: {e}")
        await asyncio.sleep(delay)
        raise

async def cleanup_services(app: FastAPI):
    """Clean up application services."""
    logger.info("Cleaning up services...")
    
    # Stop monitoring
    if hasattr(app.state, 'monitoring_task'):
        app.state.monitoring_task.cancel()
        try:
            await app.state.monitoring_task
        except asyncio.CancelledError:
            pass
    
    # Close database
    if hasattr(app.state, 'db'):
        await app.state.db.close()
    
    # Close cache
    if hasattr(app.state, 'cache'):
        await app.state.cache.close()
    
    logger.info("Services cleaned up")

async def create_database_connection():
    """Create database connection."""
    # Simulate database connection
    await asyncio.sleep(0.1)
    return {"type": "database", "connected": True}

async def create_cache_connection():
    """Create cache connection."""
    # Simulate cache connection
    await asyncio.sleep(0.1)
    return {"type": "cache", "connected": True}

async def load_models():
    """Load AI models."""
    # Simulate model loading
    await asyncio.sleep(1)
    return {"video_model": "loaded", "text_model": "loaded"}

async def collect_application_metrics(app: FastAPI) -> Dict[str, Any]:
    """Collect application metrics."""
    return {
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "active_requests": getattr(app.state, 'active_requests', 0),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
    }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_lifespan():
    """Example of basic lifespan usage."""
    
    app = FastAPI(lifespan=lifespan)
    
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return {"status": "healthy", "uptime": time.time() - app.state.startup_time}
    
    return app

def example_advanced_lifespan():
    """Example of advanced lifespan with health checks."""
    
    app = FastAPI(lifespan=lifespan_with_health_checks)
    
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return await run_health_checks(app)
    
    return app

def example_retry_lifespan():
    """Example of lifespan with retry logic."""
    
    app = FastAPI(lifespan=lifespan_with_retry)
    
    @app.get("/status")
    async def status():
        
    """status function."""
return {"status": "running", "retries": "configured"}
    
    return app

if __name__ == "__main__":
    # Example usage
    app = create_app()
    
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "AI Video API with lifespan management"}
    
    @app.get("/metrics")
    async def metrics():
        
    """metrics function."""
if hasattr(app.state, 'performance_monitor'):
            return await app.state.performance_monitor.collect_metrics()
        return {"error": "Performance monitor not available"}
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 