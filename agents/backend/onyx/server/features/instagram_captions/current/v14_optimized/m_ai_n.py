from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from core.async_database import (
from core.shared_resources import (
from core.optimized_engine import (
from core.middleware import create_middleware_stack
from core.exceptions import (
from core.schemas import (
from config import settings
from api.routes import captions_router, batch_router, health_router, stats_router
from routes.lazy_loading_routes import lazy_loading_router
from routes.shared_resources_routes import shared_resources_router
from routes.async_flow_routes import async_flow_router
from routes.enhanced_async_routes import enhanced_async_router
import logging
from typing import Any, List, Dict, Optional
"""
FastAPI Application for Instagram Captions API v14.0

Ultra-optimized with:
- Non-blocking async I/O operations
- Advanced middleware stack
- Comprehensive error handling
- Performance monitoring
- Modular router architecture
"""



# Import async I/O components
    initialize_async_io, cleanup_async_io,
    db_pool, api_client, io_monitor
)

# Import shared resources
    initialize_shared_resources, shutdown_shared_resources,
    ResourceConfig
)
    engine, OptimizedRequest, OptimizedResponse,
    generate_caption_optimized, generate_batch_captions_optimized,
    get_engine_stats
)
    ValidationError, AIGenerationError, CacheError,
    handle_validation_error, handle_ai_error, handle_cache_error
)
    CaptionRequest, CaptionResponse, BatchRequest, BatchResponse,
    PerformanceStats, ErrorResponse
)

# Import configuration

# Import routers


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with async I/O and shared resources initialization"""
    # Startup
    logger.info("🚀 Starting Instagram Captions API v14.0")
    
    try:
        # Initialize shared resources
        resource_config = ResourceConfig(
            database_url="postgresql+asyncpg://user:pass@localhost/instagram_captions",
            redis_url="redis://localhost:6379",
            enable_monitoring=True
        )
        await initialize_shared_resources(resource_config)
        logger.info("✅ Shared resources initialized")
        
        # Initialize async I/O components
        await initialize_async_io()
        logger.info("✅ Async I/O components initialized")
        
        # Initialize engine
        await engine._initialize_models()
        logger.info("✅ AI engine initialized")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Instagram Captions API v14.0")
        
        try:
            # Shutdown shared resources
            await shutdown_shared_resources()
            logger.info("✅ Shared resources cleaned up")
            
            # Cleanup async I/O components
            await cleanup_async_io()
            logger.info("✅ Async I/O components cleaned up")
            
            # Cleanup engine
            await engine.cleanup()
            logger.info("✅ AI engine cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Instagram Captions API v14.0",
    description="Ultra-optimized Instagram captions generation with async I/O",
    version="14.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware stack
middleware_stack = create_middleware_stack()
for middleware in middleware_stack:
    app.add_middleware(middleware)


# Global exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="validation_error",
            message=exc.message,
            details=exc.details,
            request_id=exc.request_id
        ).dict()
    )


@app.exception_handler(AIGenerationError)
async def ai_generation_exception_handler(request: Request, exc: AIGenerationError):
    """Handle AI generation errors"""
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="ai_generation_error",
            message=exc.message,
            details=exc.details,
            request_id=exc.request_id
        ).dict()
    )


@app.exception_handler(CacheError)
async def cache_exception_handler(request: Request, exc: CacheError):
    """Handle cache errors"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="cache_error",
            message=exc.message,
            details=exc.details,
            request_id=exc.request_id
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__},
            request_id=None
        ).dict()
    )


# Include routers
app.include_router(captions_router, prefix="/api/v14", tags=["captions"])
app.include_router(batch_router, prefix="/api/v14", tags=["batch"])
app.include_router(health_router, prefix="/api/v14", tags=["health"])
app.include_router(stats_router, prefix="/api/v14", tags=["stats"])
app.include_router(lazy_loading_router, prefix="/api/v14", tags=["lazy-loading"])
app.include_router(shared_resources_router, prefix="/api/v14", tags=["shared-resources"])
app.include_router(async_flow_router, prefix="/api/v14", tags=["async-flows"])
app.include_router(enhanced_async_router, prefix="/api/v14", tags=["enhanced-async"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Instagram Captions API v14.0",
        "version": "14.0.0",
        "status": "running",
        "features": [
            "Ultra-optimized async I/O",
            "Advanced caching system",
            "Lazy loading with background preloading",
            "Comprehensive error handling",
            "Performance monitoring",
            "Circuit breaker pattern",
            "Connection pooling"
        ],
        "documentation": "/docs",
        "health_check": "/api/v14/health"
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check with async I/O status"""
    try:
        # Check database connectivity
        db_stats = db_pool.get_stats()
        db_healthy = db_stats["total_queries"] >= 0
        
        # Check API client connectivity
        api_stats = api_client.get_stats()
        api_healthy = api_stats["total_requests"] >= 0
        
        # Check engine status
        engine_stats = get_engine_stats()
        engine_healthy = engine_stats["requests"] >= 0
        
        # Overall health
        healthy = db_healthy and api_healthy and engine_healthy
        
        return {
            "status": "healthy" if healthy else "unhealthy",
            "timestamp": time.time(),
            "components": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "total_queries": db_stats["total_queries"],
                    "cache_hit_rate": db_stats["cache_hit_rate"]
                },
                "api_client": {
                    "status": "healthy" if api_healthy else "unhealthy",
                    "total_requests": api_stats["total_requests"],
                    "success_rate": api_stats["success_rate"]
                },
                "engine": {
                    "status": "healthy" if engine_healthy else "unhealthy",
                    "total_requests": engine_stats["requests"],
                    "success_rate": engine_stats["success_rate"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


# Performance monitoring endpoint
@app.get("/api/v14/performance")
async def get_performance_stats():
    """Get comprehensive performance statistics"""
    try:
        # Get engine stats
        engine_stats = get_engine_stats()
        
        # Get I/O stats
        io_stats = io_monitor.get_performance_summary()
        
        # Get database stats
        db_stats = db_pool.get_stats()
        
        # Get API stats
        api_stats = api_client.get_stats()
        
        return {
            "timestamp": time.time(),
            "engine": engine_stats,
            "io_operations": io_stats,
            "database": db_stats,
            "api_client": api_stats,
            "summary": {
                "total_requests": engine_stats["requests"],
                "success_rate": engine_stats["success_rate"],
                "avg_response_time": engine_stats["avg_time"],
                "cache_hit_rate": engine_stats["cache"]["hit_rate"],
                "database_queries": db_stats["total_queries"],
                "api_requests": api_stats["total_requests"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dependency injection
async def get_current_user(request: Request) -> str:
    """Get current user from request (placeholder)"""
    # In a real application, this would extract user from JWT token
    return request.headers.get("X-User-ID", "anonymous")


# Request/Response middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record I/O operation
    io_monitor.record_operation(
        "http_request",
        process_time,
        response.status_code < 400
    )
    
    return response


# Startup event (deprecated, using lifespan instead)
@app.on_event("startup")
async def startup_event():
    """Startup event (deprecated - using lifespan context manager)"""
    logger.warning("Using deprecated startup event - migrate to lifespan context manager")


# Shutdown event (deprecated, using lifespan instead)
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event (deprecated - using lifespan context manager)"""
    logger.warning("Using deprecated shutdown event - migrate to lifespan context manager")


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level="info"
    ) 