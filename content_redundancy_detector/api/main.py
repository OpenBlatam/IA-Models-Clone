"""
Content Redundancy Detector - Modular FastAPI Application
Microservices-ready architecture with dependency injection
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from ..core.config import get_settings
from ..core.logging_config import setup_logging, get_logger
from ..infrastructure.service_registry import ServiceRegistry
from .middleware import (
    setup_security_middleware,
    setup_cors_middleware,
    setup_logging_middleware,
    setup_rate_limiting_middleware,
    setup_performance_middleware,
    setup_http_cache_middleware
)
from .routes import register_routers
from .exception_handlers import setup_exception_handlers
from ..core.telemetry import setup_tracing, instrument_fastapi, instrument_httpx

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager with modular service initialization"""
    settings = get_settings()
    
    # Setup logging first
    setup_logging()
    # Setup tracing (graceful fallback if libs are missing)
    try:
        setup_tracing(
            enabled=True,
            service_name=settings.app_name,
            otlp_endpoint=getattr(settings, "otlp_endpoint", None),
            sample_ratio=getattr(settings, "tracing_sample_ratio", 0.1),
        )
        instrument_httpx()
    except Exception:
        logger.debug("Tracing setup skipped")
    logger.info("=" * 60)
    logger.info("Starting Content Redundancy Detector API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Version: {settings.app_version}")
    logger.info("=" * 60)
    
    # Initialize service registry
    service_registry = ServiceRegistry()
    
    try:
        # Initialize infrastructure services
        logger.info("Initializing infrastructure services...")
        await service_registry.initialize_all()
        
        # Store registry in app state for dependency injection
        app.state.services = service_registry
        
        logger.info("✅ All services initialized successfully")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown services
        logger.info("=" * 60)
        logger.info("Shutting down services...")
        logger.info("=" * 60)
        
        try:
            await service_registry.shutdown_all()
            logger.info("✅ All services shut down successfully")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        
        logger.info("=" * 60)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    Follows dependency injection pattern for microservices
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Modular Content Redundancy Detector with AI/ML capabilities - Microservices-ready",
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Setup middleware in correct order
    setup_security_middleware(app)
    setup_cors_middleware(app)
    setup_logging_middleware(app)
    setup_rate_limiting_middleware(app)
    setup_performance_middleware(app)
    setup_http_cache_middleware(app)
    # Instrument FastAPI
    try:
        instrument_fastapi(app)
    except Exception:
        pass
    
    # Register routers
    register_routers(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint with system information"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
            "docs": "/docs",
            "health": "/api/v1/health",
            "timestamp": time.time()
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    from ..core.config import get_settings
    
    settings = get_settings()
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )

