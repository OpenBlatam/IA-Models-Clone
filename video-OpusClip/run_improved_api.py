"""
Run Improved Video-OpusClip API

Complete integration script that demonstrates all improvements:
- Configuration management
- Database initialization
- Middleware setup
- Performance monitoring
- Health checking
- Graceful shutdown
"""

import asyncio
import sys
import signal
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import improved modules
from config import settings, validate_configuration
from database import db_manager, DatabaseMigrator, VideoRepository
from cache import CacheManager, CacheConfig
from monitoring import PerformanceMonitor, HealthChecker, MonitoringConfig
from middleware import create_middleware_registry
from error_handling import handle_processing_errors
from models import HealthResponse

# Configure structured logging
logger = structlog.get_logger("main")

# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with startup and shutdown."""
    logger.info("Starting Video-OpusClip API", version=settings.app_version)
    
    try:
        # Validate configuration
        if not validate_configuration():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Initialize database
        logger.info("Initializing database")
        await db_manager.initialize()
        
        # Run database migrations
        logger.info("Running database migrations")
        migrator = DatabaseMigrator(db_manager)
        await migrator.run_migrations()
        
        # Initialize cache
        logger.info("Initializing cache")
        cache_config = CacheConfig(
            enable_fallback=settings.cache_enable_fallback,
            fallback_max_size=settings.cache_fallback_max_size
        )
        app.state.cache = CacheManager(cache_config)
        await app.state.cache.initialize()
        
        # Initialize monitoring
        logger.info("Initializing monitoring")
        monitoring_config = MonitoringConfig(
            enable_performance_monitoring=settings.enable_performance_monitoring,
            enable_health_checks=settings.enable_health_checks
        )
        app.state.performance_monitor = PerformanceMonitor(monitoring_config)
        await app.state.performance_monitor.start()
        
        app.state.health_checker = HealthChecker(monitoring_config)
        await app.state.health_checker.initialize()
        
        # Initialize repositories
        app.state.video_repository = VideoRepository(db_manager)
        
        logger.info("Video-OpusClip API started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        sys.exit(1)
    
    finally:
        # Shutdown sequence
        logger.info("Shutting down Video-OpusClip API")
        
        try:
            # Stop monitoring
            if hasattr(app.state, 'performance_monitor'):
                await app.state.performance_monitor.stop()
            
            if hasattr(app.state, 'health_checker'):
                await app.state.health_checker.close()
            
            # Close cache
            if hasattr(app.state, 'cache'):
                await app.state.cache.close()
            
            # Close database
            await db_manager.close()
            
            logger.info("Video-OpusClip API shutdown completed")
            
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Setup routes
    setup_routes(app)
    
    return app

def setup_middleware(app: FastAPI):
    """Setup middleware for the application."""
    
    # Create middleware registry
    middleware_registry = create_middleware_registry()
    
    # Set performance monitor
    if hasattr(app.state, 'performance_monitor'):
        middleware_registry.set_performance_monitor(app.state.performance_monitor)
    
    # Apply middleware
    middleware_registry.apply_middleware(app)
    
    logger.info("Middleware setup completed")

def setup_routes(app: FastAPI):
    """Setup API routes."""
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            # Check database health
            db_health = await db_manager.health_check()
            
            # Check cache health
            cache_health = app.state.cache.get_health_status()
            
            # Check system health
            system_health = await app.state.health_checker.check_system_health()
            
            # Determine overall health
            is_healthy = (
                db_health.get('healthy', False) and
                cache_health.get('healthy', False) and
                system_health.is_healthy
            )
            
            return HealthResponse(
                status="healthy" if is_healthy else "unhealthy",
                timestamp=system_health.timestamp,
                system_metrics=system_health.system_metrics,
                gpu_metrics=system_health.gpu_metrics,
                database_health=db_health,
                cache_health=cache_health,
                issues=system_health.issues if not is_healthy else []
            )
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return HealthResponse(
                status="unhealthy",
                issues=[f"Health check error: {str(e)}"]
            )
    
    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Get application metrics."""
        try:
            # Get performance metrics
            performance_metrics = app.state.performance_monitor.get_metrics()
            
            # Get database stats
            db_stats = db_manager.get_stats()
            
            # Get cache stats
            cache_stats = app.state.cache.get_stats()
            
            return {
                "performance": performance_metrics,
                "database": db_stats,
                "cache": cache_stats,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            return {"error": str(e)}
    
    # API info endpoint
    @app.get("/")
    async def api_info():
        """API information endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": settings.app_description,
            "environment": settings.environment,
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }
    
    logger.info("Routes setup completed")

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def setup_signal_handlers(app: FastAPI):
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        # The lifespan context manager will handle cleanup
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Create application
    app = create_app()
    
    # Setup signal handlers
    setup_signal_handlers(app)
    
    # Run application
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers if settings.is_production else 1,
        reload=settings.reload,
        log_level=settings.log_level.value,
        access_log=True
    )

if __name__ == "__main__":
    main()






























