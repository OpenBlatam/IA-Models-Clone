"""
FastAPI Application for OpusClip Improved
========================================

Advanced video processing and AI-powered content creation application.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

from .schemas import get_settings, ErrorResponse
from .routes import router
from .middleware import (
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware
)
from .exceptions import OpusClipException
from .database import init_database
from .ai_engine import ai_engine
from .services import opus_clip_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting OpusClip Improved API")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")
        
        # Initialize AI engine
        ai_engine._initialize_models()
        logger.info("AI engine initialized")
        
        # Initialize service
        opus_clip_service._ensure_directories()
        logger.info("Service initialized")
        
        logger.info("OpusClip Improved API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpusClip Improved API")
    
    try:
        # Cleanup resources
        logger.info("Cleaning up resources")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Get settings
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None
    )
    
    # Add middleware
    _add_middleware(app, settings)
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Include routers
    app.include_router(router)
    
    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint"""
        return {
            "message": "OpusClip Improved API",
            "version": settings.api_version,
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled"
        }
    
    # Add metrics endpoint
    if settings.enable_metrics:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    return app


def _add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the application"""
    
    # Error handling middleware (first)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Performance monitoring middleware
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=1.0)
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware, include_body=False)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
        )


def _add_exception_handlers(app: FastAPI) -> None:
    """Add exception handlers to the application"""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.error(f"Validation error: {exc}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code="VALIDATION_ERROR",
                error_message="Request validation failed",
                error_details={"errors": exc.errors()}
            ).model_dump()
        )
    
    @app.exception_handler(OpusClipException)
    async def opus_clip_exception_handler(request: Request, exc: OpusClipException):
        """Handle OpusClip exceptions"""
        logger.error(f"OpusClip exception: {exc}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_code=exc.error_code,
                error_message=exc.message,
                error_details=exc.details
            ).model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INTERNAL_ERROR",
                error_message="Internal server error",
                error_details={"error": str(exc)} if app.debug else {}
            ).model_dump()
        )


def main():
    """Main entry point"""
    settings = get_settings()
    
    logger.info(f"Starting OpusClip Improved API on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "opus_clip_improved.app:create_app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
        factory=True
    )


if __name__ == "__main__":
    main()






























