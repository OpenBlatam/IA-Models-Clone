from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import torch
import psutil
import signal
import sys
    import uvicorn
from typing import Any, List, Dict, Optional
"""
🔄 LIFESPAN MIGRATION EXAMPLES
==============================

Practical examples showing migration from @app.on_event() to lifespan context managers.
"""


logger = logging.getLogger(__name__)

# ============================================================================
# EXAMPLE 1: BASIC MIGRATION
# ============================================================================

# ❌ OLD WAY - Don't use this
"""
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    
    """startup_event function."""
app.state.redis = redis.Redis()
    app.state.db = create_engine("postgresql://...")
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    
    """shutdown_event function."""
await app.state.redis.close()
    app.state.db.dispose()
    logger.info("Application shutdown")
"""

# ✅ NEW WAY - Use lifespan context manager
@asynccontextmanager
async def basic_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Basic lifespan context manager."""
    
    # Startup
    logger.info("Starting application...")
    
    try:
        # Initialize Redis
        app.state.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("Redis connected")
        
        # Initialize database
        app.state.db_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost/ai_video",
            echo=False
        )
        app.state.db_sessionmaker = async_sessionmaker(
            app.state.db_engine,
            class_=AsyncSession
        )
        logger.info("Database connected")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        
        if hasattr(app.state, 'redis'):
            await app.state.redis.close()
            logger.info("Redis disconnected")
        
        if hasattr(app.state, 'db_engine'):
            await app.state.db_engine.dispose()
            logger.info("Database disconnected")

# ============================================================================
# EXAMPLE 2: AI MODEL MANAGEMENT
# ============================================================================

@asynccontextmanager
async def ai_model_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan for AI model management."""
    
    logger.info("Loading AI models...")
    
    try:
        # Load video generation model
        logger.info("Loading video generation model...")
        app.state.video_model = await load_video_model()
        logger.info("Video model loaded")
        
        # Load text processing model
        logger.info("Loading text processing model...")
        app.state.text_model = await load_text_model()
        logger.info("Text model loaded")
        
        # Load diffusion pipeline
        logger.info("Loading diffusion pipeline...")
        app.state.diffusion_pipeline = await load_diffusion_pipeline()
        logger.info("Diffusion pipeline loaded")
        
        # Set model configurations
        app.state.model_config = {
            "video_model": "stable-diffusion-v1-5",
            "text_model": "gpt-3.5-turbo",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        yield
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise
    
    finally:
        logger.info("Unloading AI models...")
        
        # Unload models
        if hasattr(app.state, 'video_model'):
            await unload_model(app.state.video_model)
        
        if hasattr(app.state, 'text_model'):
            await unload_model(app.state.text_model)
        
        if hasattr(app.state, 'diffusion_pipeline'):
            await unload_model(app.state.diffusion_pipeline)
        
        logger.info("AI models unloaded")

# ============================================================================
# EXAMPLE 3: BACKGROUND TASK MANAGEMENT
# ============================================================================

@asynccontextmanager
async def background_task_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan for background task management."""
    
    logger.info("Starting background task manager...")
    
    try:
        # Initialize task manager
        app.state.task_manager = BackgroundTaskManager()
        await app.state.task_manager.start()
        
        # Start monitoring task
        app.state.monitoring_task = asyncio.create_task(
            monitor_system_resources(app)
        )
        
        # Start cleanup task
        app.state.cleanup_task = asyncio.create_task(
            cleanup_old_files(app)
        )
        
        yield
        
    except Exception as e:
        logger.error(f"Background task startup failed: {e}")
        raise
    
    finally:
        logger.info("Stopping background tasks...")
        
        # Stop task manager
        if hasattr(app.state, 'task_manager'):
            await app.state.task_manager.stop()
        
        # Cancel monitoring task
        if hasattr(app.state, 'monitoring_task'):
            app.state.monitoring_task.cancel()
            try:
                await app.state.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel cleanup task
        if hasattr(app.state, 'cleanup_task'):
            app.state.cleanup_task.cancel()
            try:
                await app.state.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background tasks stopped")

# ============================================================================
# EXAMPLE 4: COMPREHENSIVE LIFESPAN
# ============================================================================

@asynccontextmanager
async def comprehensive_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Comprehensive lifespan with all features."""
    
    logger.info("🚀 Starting comprehensive AI Video application...")
    start_time = time.time()
    
    try:
        # 1. Initialize application state
        app.state.startup_time = start_time
        app.state.is_healthy = True
        app.state.active_requests = 0
        app.state.version = "1.0.0"
        
        # 2. Initialize database
        logger.info("📊 Initializing database...")
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
        logger.info("✅ Database initialized")
        
        # 3. Initialize Redis
        logger.info("🔴 Initializing Redis...")
        app.state.redis = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("✅ Redis initialized")
        
        # 4. Load AI models
        logger.info("🤖 Loading AI models...")
        app.state.video_model = await load_video_model()
        app.state.text_model = await load_text_model()
        app.state.diffusion_pipeline = await load_diffusion_pipeline()
        logger.info("✅ AI models loaded")
        
        # 5. Initialize performance monitoring
        logger.info("📈 Initializing performance monitoring...")
        app.state.performance_monitor = PerformanceMonitor()
        await app.state.performance_monitor.start()
        logger.info("✅ Performance monitoring started")
        
        # 6. Start background tasks
        logger.info("🔄 Starting background tasks...")
        app.state.task_manager = BackgroundTaskManager()
        await app.state.task_manager.start()
        
        app.state.monitoring_task = asyncio.create_task(
            monitor_system_resources(app)
        )
        app.state.cleanup_task = asyncio.create_task(
            cleanup_old_files(app)
        )
        logger.info("✅ Background tasks started")
        
        # 7. Set up signal handlers
        setup_signal_handlers(app)
        
        # 8. Run health checks
        logger.info("🏥 Running health checks...")
        health_status = await run_health_checks(app)
        if not health_status["healthy"]:
            raise RuntimeError(f"Health checks failed: {health_status['errors']}")
        logger.info("✅ Health checks passed")
        
        startup_time = time.time() - start_time
        logger.info(f"🎉 Application startup completed in {startup_time:.2f}s")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        logger.info("🛑 Shutting down application...")
        shutdown_start = time.time()
        
        try:
            # Stop background tasks
            if hasattr(app.state, 'task_manager'):
                await app.state.task_manager.stop()
            
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
            
            # Stop performance monitoring
            if hasattr(app.state, 'performance_monitor'):
                await app.state.performance_monitor.stop()
            
            # Unload AI models
            if hasattr(app.state, 'video_model'):
                await unload_model(app.state.video_model)
            
            if hasattr(app.state, 'text_model'):
                await unload_model(app.state.text_model)
            
            if hasattr(app.state, 'diffusion_pipeline'):
                await unload_model(app.state.diffusion_pipeline)
            
            # Close Redis
            if hasattr(app.state, 'redis'):
                await app.state.redis.close()
            
            # Close database
            if hasattr(app.state, 'db_engine'):
                await app.state.db_engine.dispose()
            
            shutdown_time = time.time() - shutdown_start
            logger.info(f"✅ Application shutdown completed in {shutdown_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

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

# ============================================================================
# MODEL LOADING FUNCTIONS
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

async def load_diffusion_pipeline():
    """Load diffusion pipeline."""
    logger.info("Loading diffusion pipeline...")
    await asyncio.sleep(1.5)  # Simulate pipeline loading
    return {"pipeline": "stable_diffusion", "loaded": True}

async def unload_model(model) -> Any:
    """Unload AI model."""
    logger.info("Unloading model...")
    await asyncio.sleep(0.3)  # Simulate model unloading

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def monitor_system_resources(app: FastAPI):
    """Monitor system resources."""
    while True:
        try:
            metrics = await collect_system_metrics(app)
            
            # Log metrics
            logger.info(f"System metrics: {metrics}")
            
            # Check for issues
            if metrics["memory_usage"] > 90:
                logger.warning("High memory usage detected")
            
            if metrics["cpu_usage"] > 80:
                logger.warning("High CPU usage detected")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            await asyncio.sleep(5)

async def cleanup_old_files(app: FastAPI):
    """Clean up old temporary files."""
    while True:
        try:
            logger.info("Cleaning up old files...")
            # Simulate file cleanup
            await asyncio.sleep(300)  # Run every 5 minutes
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(60)

async def collect_system_metrics(app: FastAPI) -> Dict[str, Any]:
    """Collect system metrics."""
    return {
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "disk_usage": psutil.disk_usage('/').percent,
        "active_requests": getattr(app.state, 'active_requests', 0),
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
    }

# ============================================================================
# HEALTH CHECKS AND UTILITIES
# ============================================================================

async def run_health_checks(app: FastAPI) -> Dict[str, Any]:
    """Run health checks on all services."""
    logger.info("Running health checks...")
    
    errors = []
    
    # Database health check
    try:
        async with app.state.db_engine.begin() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        errors.append(f"Database: {e}")
    
    # Redis health check
    try:
        await app.state.redis.ping()
    except Exception as e:
        errors.append(f"Redis: {e}")
    
    # Model health check
    try:
        if not hasattr(app.state, 'video_model'):
            errors.append("Video model not loaded")
        if not hasattr(app.state, 'text_model'):
            errors.append("Text model not loaded")
    except Exception as e:
        errors.append(f"Models: {e}")
    
    healthy = len(errors) == 0
    
    return {
        "healthy": healthy,
        "errors": errors,
        "timestamp": time.time()
    }

def setup_signal_handlers(app: FastAPI):
    """Set up signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# FASTAPI APPLICATION EXAMPLES
# ============================================================================

def create_basic_app() -> FastAPI:
    """Create basic FastAPI app with lifespan."""
    app = FastAPI(
        title="AI Video API - Basic",
        lifespan=basic_lifespan
    )
    
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return {"status": "healthy"}
    
    return app

def create_ai_model_app() -> FastAPI:
    """Create FastAPI app with AI model lifespan."""
    app = FastAPI(
        title="AI Video API - AI Models",
        lifespan=ai_model_lifespan
    )
    
    @app.get("/models")
    async def list_models():
        
    """list_models function."""
return {
            "video_model": app.state.video_model,
            "text_model": app.state.text_model,
            "diffusion_pipeline": app.state.diffusion_pipeline
        }
    
    return app

def create_comprehensive_app() -> FastAPI:
    """Create comprehensive FastAPI app with full lifespan."""
    app = FastAPI(
        title="AI Video API - Comprehensive",
        lifespan=comprehensive_lifespan
    )
    
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return await run_health_checks(app)
    
    @app.get("/metrics")
    async def get_metrics():
        
    """get_metrics function."""
if hasattr(app.state, 'performance_monitor'):
            return await app.state.performance_monitor.collect_metrics()
        return {"error": "Performance monitor not available"}
    
    @app.get("/status")
    async def get_status():
        
    """get_status function."""
return {
            "status": "running",
            "uptime": time.time() - app.state.startup_time,
            "version": app.state.version,
            "active_requests": app.state.active_requests
        }
    
    return app

# ============================================================================
# MIGRATION UTILITIES
# ============================================================================

def migrate_from_on_event_to_lifespan():
    """Example of migrating from @app.on_event to lifespan."""
    
    # ❌ OLD CODE (to be replaced)
    old_code = '''
    app = FastAPI()

    @app.on_event("startup")
    async def startup_event():
        
    """startup_event function."""
app.state.redis = redis.Redis()
        app.state.db = create_engine("postgresql://...")
        logger.info("Application started")

    @app.on_event("shutdown")
    async def shutdown_event():
        
    """shutdown_event function."""
await app.state.redis.close()
        app.state.db.dispose()
        logger.info("Application shutdown")
    '''
    
    # ✅ NEW CODE (replacement)
    new_code = '''
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
    '''
    
    return {
        "old_code": old_code,
        "new_code": new_code,
        "benefits": [
            "Better error handling",
            "Cleaner resource management",
            "Easier testing",
            "More maintainable code"
        ]
    }

if __name__ == "__main__":
    # Example usage
    app = create_comprehensive_app()
    
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "AI Video API with comprehensive lifespan management"}
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 