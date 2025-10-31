"""
Main FastAPI application with advanced features.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time

from .core.config import get_settings
from .core.logging import get_logger, setup_logging
from .core.metrics import MetricsMiddleware, get_metrics_collector
from .core.cache import get_cache_manager
from .api.v1 import analysis_router, plugin_router, system_router, advanced_router, optimized_router, lightning_router, ultra_speed_router, hyper_performance_router, ultimate_optimization_router, extreme_router, infinite_router

# Setup logging
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Ultra Modular AI History Comparison System")
    
    try:
        # Initialize cache system
        cache_manager = await get_cache_manager()
        logger.info("Cache system initialized")
        
        # Initialize metrics system
        metrics_collector = get_metrics_collector()
        logger.info("Metrics system initialized")
        
        # Initialize optimization system
        from .core.optimization import start_optimization
        await start_optimization()
        logger.info("Optimization system initialized")
        
        # Initialize speed optimization system
        from .core.speed_optimization import start_speed_optimization
        await start_speed_optimization()
        logger.info("Speed optimization system initialized")
        
        # Initialize ultra speed optimization system
        from .core.ultra_speed_engine import start_ultra_speed_optimization
        await start_ultra_speed_optimization()
        logger.info("Ultra speed optimization system initialized")
        
        # Initialize hyper performance optimization system
        from .core.hyper_performance_engine import start_hyper_performance_optimization
        await start_hyper_performance_optimization()
        logger.info("Hyper performance optimization system initialized")
        
               # Initialize ultimate optimization system
               from .core.ultimate_optimization_engine import start_ultimate_optimization
               await start_ultimate_optimization()
               logger.info("Ultimate optimization system initialized")

               # Initialize extreme optimization system
               from .core.extreme_optimization_engine import start_extreme_optimization
               await start_extreme_optimization()
               logger.info("Extreme optimization system initialized")

               # Initialize infinite optimization system
               from .core.infinite_optimization_engine import start_infinite_optimization
               await start_infinite_optimization()
               logger.info("Infinite optimization system initialized")
        
        # Initialize pool system
        from .core.async_pool import get_pool_manager
        pool_manager = get_pool_manager()
        logger.info("Pool system initialized")
        
        # Warm up system
        if settings.warmup_on_startup:
            logger.info("Warming up system...")
            # Add warmup tasks here
            logger.info("System warmup completed")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
           try:
               # Shutdown infinite optimization system
               from .core.infinite_optimization_engine import stop_infinite_optimization
               await stop_infinite_optimization()
               logger.info("Infinite optimization system shutdown completed")

               # Shutdown extreme optimization system
               from .core.extreme_optimization_engine import stop_extreme_optimization
               await stop_extreme_optimization()
               logger.info("Extreme optimization system shutdown completed")

               # Shutdown ultimate optimization system
               from .core.ultimate_optimization_engine import stop_ultimate_optimization
               await stop_ultimate_optimization()
               logger.info("Ultimate optimization system shutdown completed")
        
        # Shutdown hyper performance optimization system
        from .core.hyper_performance_engine import stop_hyper_performance_optimization
        await stop_hyper_performance_optimization()
        logger.info("Hyper performance optimization system shutdown completed")
        
        # Shutdown ultra speed optimization system
        from .core.ultra_speed_engine import stop_ultra_speed_optimization
        await stop_ultra_speed_optimization()
        logger.info("Ultra speed optimization system shutdown completed")
        
        # Shutdown speed optimization system
        from .core.speed_optimization import stop_speed_optimization
        await stop_speed_optimization()
        logger.info("Speed optimization system shutdown completed")
        
        # Shutdown optimization system
        from .core.optimization import stop_optimization
        await stop_optimization()
        logger.info("Optimization system shutdown completed")
        
        # Shutdown pool system
        from .core.async_pool import shutdown_all_pools
        await shutdown_all_pools()
        logger.info("Pool system shutdown completed")
        
        # Shutdown cache system
        cache_manager = await get_cache_manager()
        await cache_manager.shutdown()
        logger.info("Cache system shutdown completed")
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Application shutdown failed: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Ultra Modular AI History Comparison System - Infinite Edition",
        description="Advanced AI content analysis and comparison system with infinite optimization capabilities and plugin architecture",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(MetricsMiddleware)
    
    # Add custom middleware for request timing
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Include routers
    app.include_router(
        analysis_router,
        prefix="/api/v1/analysis",
        tags=["Analysis"]
    )
    
    app.include_router(
        plugin_router,
        prefix="/api/v1/plugins",
        tags=["Plugins"]
    )
    
    app.include_router(
        system_router,
        prefix="/api/v1/system",
        tags=["System"]
    )
    
    app.include_router(
        advanced_router,
        prefix="/api/v1/advanced",
        tags=["Advanced Features"]
    )
    
    app.include_router(
        optimized_router,
        prefix="/api/v1/optimized",
        tags=["Ultra Optimized Features"]
    )
    
    app.include_router(
        lightning_router,
        prefix="/api/v1/lightning",
        tags=["Lightning Fast Features"]
    )
    
    app.include_router(
        ultra_speed_router,
        prefix="/api/v1/ultra-speed",
        tags=["Ultra Speed Features"]
    )
    
    app.include_router(
        hyper_performance_router,
        prefix="/api/v1/hyper-performance",
        tags=["Hyper Performance Features"]
    )
    
app.include_router(
    ultimate_optimization_router,
    prefix="/api/v1/ultimate-optimization",
    tags=["Ultimate Optimization Features"]
)

app.include_router(
    extreme_router,
    prefix="/api/v1/extreme",
    tags=["Extreme Features"]
)

app.include_router(
    infinite_router,
    prefix="/api/v1/infinite",
    tags=["Infinite Features"]
)
    
    # Add exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {exc.errors()} for URL: {request.url}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "body": exc.body,
                "message": "Request validation failed"
            },
        )
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"message": "Endpoint not found", "path": str(request.url)}
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with system information."""
        return {
            "message": "Ultra Modular AI History Comparison System",
            "version": settings.app_version,
            "environment": settings.environment,
            "status": "operational",
            "docs_url": "/docs" if settings.debug else None,
            "api_version": "v1"
        }
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint."""
        try:
            # Check cache system
            cache_manager = await get_cache_manager()
            cache_stats = await cache_manager.get_stats()
            
            # Check metrics system
            metrics_collector = get_metrics_collector()
            
            return {
                "status": "healthy",
                "version": settings.app_version,
                "environment": settings.environment,
                "cache_status": "connected" if cache_stats.get("redis_connected") else "memory_only",
                "metrics_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "version": settings.app_version
                }
            )
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )