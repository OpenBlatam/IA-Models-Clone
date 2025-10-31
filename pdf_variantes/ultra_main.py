"""
Ultra-Optimized Main Application
===============================

High-performance FastAPI application with all optimizations integrated.
Features ultra-fast processing, comprehensive monitoring, and production-ready setup.

Key Features:
- Ultra-efficient PDF processing
- Async caching with LRU and TTL
- Performance monitoring middleware
- Streaming responses
- Background task processing
- Health checks and metrics
- Error handling and recovery
- Production-ready configuration

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Optimized
License: MIT
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .performance_middleware import (
    setup_performance_middleware, 
    get_performance_summary,
    get_performance_dashboard_data,
    cleanup_performance_monitoring
)
from .async_cache_layer import (
    get_all_cache_stats,
    clear_all_caches,
    cleanup_all_caches,
    cleanup_caches
)
from .routers.ultra_optimized_router import router as ultra_router
from .optimized_processor import get_cache_stats as get_processor_cache_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with cleanup."""
    # Startup
    logger.info("Starting Ultra-Optimized PDF Variantes API")
    
    # Initialize performance monitoring
    setup_performance_middleware(app)
    logger.info("Performance monitoring initialized")
    
    # Warm up caches (optional)
    logger.info("Cache systems ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ultra-Optimized PDF Variantes API")
    
    # Cleanup resources
    await asyncio.gather(
        cleanup_performance_monitoring(),
        cleanup_caches(),
        return_exceptions=True
    )
    
    logger.info("Cleanup completed")

def create_ultra_app() -> FastAPI:
    """Create ultra-optimized FastAPI application."""
    
    app = FastAPI(
        title="PDF Variantes Ultra API",
        description="Ultra-optimized PDF processing system with advanced performance features",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include ultra-optimized router
    app.include_router(ultra_router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler with performance logging."""
        logger.error(f"Unhandled exception: {exc}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Get performance metrics
            perf_summary = await get_performance_summary()
            
            # Get cache stats
            cache_stats = await get_all_cache_stats()
            processor_cache_stats = get_processor_cache_stats()
            
            # Determine health status
            status = "healthy"
            if perf_summary["alerts"]:
                status = "degraded"
            
            return {
                "status": status,
                "service": "pdf-variantes-ultra",
                "version": "1.0.0",
                "timestamp": time.time(),
                "performance": perf_summary,
                "cache_stats": cache_stats,
                "processor_cache_stats": processor_cache_stats,
                "uptime": time.time() - perf_summary["metrics"]["global"]["uptime"]
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    # Performance dashboard endpoint
    @app.get("/dashboard", tags=["System"])
    async def performance_dashboard() -> Dict[str, Any]:
        """Performance dashboard data."""
        try:
            dashboard_data = await get_performance_dashboard_data()
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard data failed: {e}")
            raise HTTPException(status_code=500, detail=f"Dashboard failed: {e}")
    
    # Cache management endpoints
    @app.get("/cache/stats", tags=["System"])
    async def cache_statistics() -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            async_cache_stats = await get_all_cache_stats()
            processor_cache_stats = get_processor_cache_stats()
            
            return {
                "success": True,
                "async_cache": async_cache_stats,
                "processor_cache": processor_cache_stats,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cache stats failed: {e}")
    
    @app.post("/cache/clear", tags=["System"])
    async def clear_all_cache() -> Dict[str, Any]:
        """Clear all cache systems."""
        try:
            async_counts = await clear_all_caches()
            processor_stats = get_processor_cache_stats()
            
            return {
                "success": True,
                "message": "All caches cleared successfully",
                "async_cache_cleared": async_counts,
                "processor_cache_size": processor_stats["cache_size"],
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")
    
    @app.post("/cache/cleanup", tags=["System"])
    async def cleanup_expired_cache() -> Dict[str, Any]:
        """Cleanup expired cache entries."""
        try:
            cleaned_count = await cleanup_all_caches()
            
            return {
                "success": True,
                "message": f"Cleaned up {cleaned_count} expired cache entries",
                "cleaned_count": cleaned_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            raise HTTPException(status_code=500, detail=f"Cache cleanup failed: {e}")
    
    # Metrics endpoint
    @app.get("/metrics", tags=["System"])
    async def get_metrics() -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            perf_summary = await get_performance_summary()
            cache_stats = await get_all_cache_stats()
            
            return {
                "success": True,
                "performance": perf_summary,
                "cache": cache_stats,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Metrics failed: {e}")
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "message": "PDF Variantes Ultra API",
            "version": "1.0.0",
            "description": "Ultra-optimized PDF processing system",
            "features": [
                "Ultra-fast PDF processing with PyMuPDF",
                "Async caching with LRU and TTL",
                "Performance monitoring and metrics",
                "Streaming responses",
                "Background task processing",
                "Comprehensive error handling"
            ],
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "dashboard": "/dashboard",
                "metrics": "/metrics",
                "cache_stats": "/cache/stats",
                "ultra_processing": "/pdf-ultra"
            },
            "timestamp": time.time()
        }
    
    logger.info("Ultra-optimized PDF Variantes API created successfully")
    return app

# Create the app instance
app = create_ultra_app()

# Export for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ultra_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )