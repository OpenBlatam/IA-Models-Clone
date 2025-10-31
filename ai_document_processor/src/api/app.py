"""
FastAPI Application - Main Application Factory
============================================

Main FastAPI application with clean architecture and comprehensive setup.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routes import (
    documents_router,
    processing_router,
    health_router,
    metrics_router
)
from .middleware import (
    setup_cors,
    setup_logging,
    setup_error_handlers,
    setup_rate_limiting
)
from .dependencies import get_config_manager
from ..core.exceptions import AIProcessorError
from ..core.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting AI Document Processor API...")
    
    try:
        # Initialize services
        config_manager = get_config_manager()
        logger.info("✅ Configuration manager initialized")
        
        # Initialize other services here
        # await initialize_database()
        # await initialize_cache()
        # await initialize_ai_services()
        
        logger.info("✅ All services initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI Document Processor API...")
        
        # Cleanup services
        # await cleanup_database()
        # await cleanup_cache()
        # await cleanup_ai_services()
        
        logger.info("✅ Shutdown complete")


def create_app(config_manager: ConfigManager) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Configured FastAPI application
    """
    # Get server configuration
    server_config = config_manager.get('server')
    
    # Create FastAPI app
    app = FastAPI(
        title="AI Document Processor API",
        description="Ultra-fast AI document processing with modern architecture",
        version="3.0.0",
        lifespan=lifespan,
        docs_url="/docs" if server_config.reload else None,
        redoc_url="/redoc" if server_config.reload else None,
        openapi_url="/openapi.json" if server_config.reload else None
    )
    
    # Setup middleware
    setup_cors(app, config_manager)
    setup_logging(app, config_manager)
    setup_error_handlers(app)
    setup_rate_limiting(app, config_manager)
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(metrics_router, prefix="/api/v1", tags=["metrics"])
    app.include_router(documents_router, prefix="/api/v1", tags=["documents"])
    app.include_router(processing_router, prefix="/api/v1", tags=["processing"])
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "AI Document Processor API",
            "version": "3.0.0",
            "description": "Ultra-fast AI document processing with modern architecture",
            "status": "running",
            "docs_url": "/docs",
            "health_url": "/api/v1/health"
        }
    
    # Global exception handler
    @app.exception_handler(AIProcessorError)
    async def ai_processor_exception_handler(request: Request, exc: AIProcessorError):
        """Handle custom AI processor exceptions."""
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Log request
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    
    logger.info("✅ FastAPI application created and configured")
    return app


def run_app(config_manager: ConfigManager):
    """
    Run the FastAPI application.
    
    Args:
        config_manager: Configuration manager instance
    """
    app = create_app(config_manager)
    server_config = config_manager.get('server')
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        reload=server_config.reload,
        access_log=server_config.access_log,
        log_level="info"
    )


if __name__ == "__main__":
    import time
    from ..core.config import get_config_manager
    
    # Get configuration and run app
    config_manager = get_config_manager()
    run_app(config_manager)

















