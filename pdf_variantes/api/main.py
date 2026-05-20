"""
PDF Variantes API - Main FastAPI Application
Advanced PDF processing with AI capabilities, real-time collaboration, and enterprise features

Modular Architecture:
- lifecycle.py: Startup/shutdown logic
- dependencies.py: Dependency injection
- config.py: Application configuration
- routers.py: Router registration
- endpoints/: Root and health endpoints
- handlers/: Exception handlers
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..utils.logging_config import setup_logging

# Import organized modules
from .lifecycle import lifespan
from .config import (
    create_app_config,
    setup_middleware,
    create_openapi_schema
)
from .routers import register_routers
from .dependencies import get_services
from .handlers import setup_exception_handlers
from .endpoints import root_router, health_router

# Setup logging
setup_logging()


# ============================================================================
# Application Factory
# ============================================================================

def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    # Create FastAPI app with configuration
    app_config = create_app_config()
    app = FastAPI(**app_config, lifespan=lifespan)
    
    # Setup middleware (base middleware that doesn't need services)
    setup_middleware(app)
    
    # Register root endpoints
    app.include_router(root_router)
    app.include_router(health_router)
    
    # Register API routers
    register_routers(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Configure OpenAPI schema
    app.openapi = lambda: create_openapi_schema(app)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_application()


# ============================================================================
# Exports
# ============================================================================

__all__ = ["app", "get_services", "create_application"]
