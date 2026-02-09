from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import time
from datetime import datetime
from .routes import (
from .version_control_middleware import (
from .dependencies.core import cleanup_resources
from .api.config.settings import Settings
from .utils.logging import setup_logging, get_logger
    import uvicorn
from typing import Any, List, Dict, Optional
import asyncio
"""
Structured Main Application

This module provides a well-structured FastAPI application with clear
route organization, dependency injection, and comprehensive middleware.
"""


# Import routes
    register_routers,
    get_all_routers,
    ROUTER_REGISTRY
)

# Import middleware
    RequestLoggingMiddleware,
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitingMiddleware
)

# Import dependencies

# Import configuration

# Import utilities

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Load settings
settings = Settings()

# Application state
app_state: Dict[str, Any] = {
    "startup_time": None,
    "request_count": 0,
    "active_connections": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Product Descriptions API...")
    app_state["startup_time"] = datetime.utcnow()
    
    # Initialize services
    try:
        # Initialize database connections
        logger.info("Initializing database connections...")
        
        # Initialize cache
        logger.info("Initializing cache...")
        
        # Initialize monitoring
        logger.info("Initializing monitoring services...")
        
        logger.info("Product Descriptions API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Product Descriptions API...")
    
    try:
        # Cleanup resources
        await cleanup_resources()
        logger.info("Resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Product Descriptions API shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Product Descriptions API",
    description="AI-powered product description generation and management API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add Gzip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitingMiddleware)

# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request statistics."""
    app_state["request_count"] += 1
    app_state["active_connections"] += 1
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        return response
    finally:
        app_state["active_connections"] -= 1
        duration = time.time() - start_time
        
        # Log slow requests
        if duration > 5.0:  # 5 seconds threshold
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.2f}s"
            )

# Register all routers
register_routers(app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Product Descriptions API",
        "version": "1.0.0",
        "status": "running",
        "uptime": str(datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "request_count": app_state["request_count"],
        "active_connections": app_state["active_connections"],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "status": "/api/v1/status"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Application info endpoint
@app.get("/info")
async def app_info():
    """Get application information."""
    return {
        "name": "Product Descriptions API",
        "version": "1.0.0",
        "description": "AI-powered product description generation and management",
        "environment": settings.ENVIRONMENT,
        "startup_time": app_state["startup_time"].isoformat() if app_state["startup_time"] else None,
        "uptime": str(datetime.utcnow() - app_state["startup_time"]) if app_state["startup_time"] else "unknown",
        "statistics": {
            "total_requests": app_state["request_count"],
            "active_connections": app_state["active_connections"]
        },
        "features": {
            "product_descriptions": True,
            "version_control": True,
            "performance_monitoring": True,
            "caching": True,
            "error_handling": True,
            "admin_panel": True
        }
    }

# Router information endpoint
@app.get("/routers")
async def list_routers():
    """List all registered routers."""
    router_info = {}
    
    for name, router in ROUTER_REGISTRY.items():
        router_info[name] = {
            "prefix": router.prefix,
            "tags": router.tags,
            "routes_count": len(router.routes)
        }
    
    return {
        "status": "success",
        "message": "Router information retrieved",
        "data": {
            "total_routers": len(ROUTER_REGISTRY),
            "routers": router_info
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return {
        "status": "error",
        "message": "Endpoint not found",
        "error_code": 404,
        "path": request.url.path,
        "method": request.method
    }

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return {
        "status": "error",
        "message": "Internal server error",
        "error_code": 500,
        "path": request.url.path,
        "method": request.method
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    logger.info("Application startup event triggered")
    
    # Log router registration
    logger.info(f"Registered {len(ROUTER_REGISTRY)} routers:")
    for name, router in ROUTER_REGISTRY.items():
        logger.info(f"  - {name}: {router.prefix} ({len(router.routes)} routes)")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks."""
    logger.info("Application shutdown event triggered")

# Development endpoints (only in development mode)
if settings.ENVIRONMENT == "development":
    
    @app.get("/dev/debug")
    async def debug_info():
        """Debug information for development."""
        return {
            "status": "success",
            "message": "Debug information",
            "data": {
                "app_state": app_state,
                "settings": {
                    "environment": settings.ENVIRONMENT,
                    "debug": settings.DEBUG,
                    "allowed_origins": settings.ALLOWED_ORIGINS
                },
                "routers": {
                    name: {
                        "prefix": router.prefix,
                        "tags": router.tags,
                        "routes": [
                            {
                                "path": route.path,
                                "methods": route.methods,
                                "name": route.name
                            }
                            for route in router.routes
                        ]
                    }
                    for name, router in ROUTER_REGISTRY.items()
                }
            }
        }
    
    @app.get("/dev/routes")
    async def list_all_routes():
        """List all registered routes for development."""
        all_routes = []
        
        for name, router in ROUTER_REGISTRY.items():
            for route in router.routes:
                all_routes.append({
                    "router": name,
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name,
                    "tags": router.tags
                })
        
        return {
            "status": "success",
            "message": "All routes listed",
            "data": {
                "total_routes": len(all_routes),
                "routes": all_routes
            }
        }

# Export the app
__all__ = ["app"]

if __name__ == "__main__":
    
    uvicorn.run(
        "structured_main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    ) 