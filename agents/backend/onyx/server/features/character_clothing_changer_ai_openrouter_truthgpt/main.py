"""
Main Entry Point
================

FastAPI application for Character Clothing Changer AI with OpenRouter and TruthGPT.

This application provides:
- RESTful API for clothing change operations
- Integration with OpenRouter for prompt optimization
- Integration with TruthGPT for advanced processing
- ComfyUI workflow execution
- Health check endpoints
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.clothing_router import router as clothing_router
from api.health_router import router as health_router
from api.monitoring_router import router as monitoring_router
from api.analytics_router import router as analytics_router
from api.openapi_extensions import get_openapi_tags, get_openapi_info, get_openapi_servers
from config.settings import get_settings
from config.advanced_settings import get_advanced_settings
from middleware.logging_middleware import LoggingMiddleware
from middleware.error_handler_middleware import ErrorHandlerMiddleware
from middleware.rate_limit_middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "Character Clothing Changer AI - OpenRouter & TruthGPT"
APP_DESCRIPTION = """
Advanced AI-powered character clothing changer with:
- **OpenRouter**: Intelligent prompt optimization
- **TruthGPT**: Advanced processing and analytics
- **ComfyUI**: High-quality image generation workflows

## Features

* Prompt optimization with LLM models
* Advanced query enhancement
* Flux Fill inpainting workflows
* Real-time status tracking
* Comprehensive analytics
"""
APP_VERSION = "1.0.0"


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    logger.info("=" * 60)
    
    settings = get_settings()
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  - Host: {settings.host}")
    logger.info(f"  - Port: {settings.port}")
    logger.info(f"  - Debug: {settings.debug}")
    logger.info(f"  - OpenRouter: {'Enabled' if settings.openrouter_enabled else 'Disabled'}")
    logger.info(f"  - TruthGPT: {'Enabled' if settings.truthgpt_enabled else 'Disabled'}")
    logger.info(f"  - ComfyUI URL: {settings.comfyui_api_url}")
    
    # Verify critical services
    if not settings.comfyui_api_url:
        logger.warning("ComfyUI URL not configured - service may not function properly")
    
    logger.info("Application started successfully")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)
    
    # Cleanup resources
    # Add any cleanup logic here (close connections, etc.)
    
    logger.info("Application shutdown complete")


# Get advanced settings
advanced_settings = get_advanced_settings()

# Create FastAPI application with OpenAPI extensions
app = FastAPI(
    title=get_openapi_info()["title"],
    description=get_openapi_info()["description"],
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=get_openapi_tags(),
    servers=get_openapi_servers() if not get_settings().debug else None
)


# CORS Configuration
def configure_cors(app: FastAPI, settings) -> None:
    """
    Configure CORS middleware.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # In production, replace "*" with specific origins
    allowed_origins = ["*"] if settings.debug else [
        "http://localhost:3000",
        "http://localhost:8000",
        # Add production origins here
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Health-Status"],
    )


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation exception
        
    Returns:
        JSONResponse with error details
    """
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle general exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception that occurred
        
    Returns:
        JSONResponse with error details
    """
    logger.error(
        f"Unhandled exception on {request.url.path}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health"]
)
app.include_router(
    clothing_router,
    prefix="/api/v1",
    tags=["Clothing"]
)
app.include_router(
    monitoring_router,
    prefix="/api/v1",
    tags=["Monitoring"]
)
app.include_router(
    analytics_router,
    prefix="/api/v1",
    tags=["Analytics"]
)


# Root endpoint
@app.get(
    "/",
    summary="Root Endpoint",
    description="API root endpoint with basic information",
    tags=["Root"]
)
async def root() -> dict:
    """
    Root endpoint providing API information.
    
    Returns:
        Dictionary with API information
    """
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health"
    }


# Add middleware
if advanced_settings.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=advanced_settings.rate_limit_default,
        default_window=advanced_settings.rate_limit_window
    )

if advanced_settings.log_level.value in ["DEBUG", "INFO"]:
    app.add_middleware(
        LoggingMiddleware,
        log_requests=True,
        log_responses=True,
        log_body=advanced_settings.log_level == advanced_settings.log_level.DEBUG
    )

app.add_middleware(ErrorHandlerMiddleware)

# Configure CORS
settings = get_settings()
configure_cors(app, settings)


# Main entry point
if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )

