"""
FastAPI Application Factory
==========================

Clean, production-ready FastAPI application following best practices.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings, get_api_settings, get_security_settings, get_logging_settings
from .routes import router
from .services import get_copywriting_service, cleanup_copywriting_service
from .exceptions import CopywritingException
from .schemas import ErrorResponse
from .middleware import (
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware
)
from .advanced.routes import advanced_router
from .enterprise.routes import enterprise_router

# Configure logging
logging_settings = get_logging_settings()
logging.basicConfig(
    level=getattr(logging, logging_settings.level),
    format=logging_settings.format,
    handlers=[
        logging.StreamHandler(),
        *([logging.FileHandler(logging_settings.file_path)] if logging_settings.file_path else [])
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Copywriting Service...")
    start_time = time.time()
    
    try:
        # Initialize services
        service = await get_copywriting_service()
        logger.info("Services initialized successfully")
        
        # Initialize monitoring if enabled
        settings = get_settings()
        if settings.monitoring.enabled:
            instrumentator = Instrumentator()
            instrumentator.instrument(app).expose(app, endpoint=settings.monitoring.metrics_endpoint)
            logger.info("Monitoring initialized")
        
        startup_time = time.time() - start_time
        logger.info(f"Application started in {startup_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Copywriting Service...")
    try:
        await cleanup_copywriting_service()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    api_settings = get_api_settings()
    security_settings = get_security_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="Copywriting Service",
        description="High-performance copywriting generation API",
        version="2.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        openapi_url="/openapi.json" if settings.environment != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware
    _add_middleware(app, api_settings, security_settings)
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Include routers
    app.include_router(router)
    app.include_router(advanced_router)
    app.include_router(enterprise_router)
    
    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint"""
        return {
            "message": "Copywriting Service API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs" if settings.environment != "production" else "disabled"
        }
    
    return app


def _add_middleware(app: FastAPI, api_settings, security_settings) -> None:
    """Add middleware to the application"""
    
    # Custom middleware (order matters - first added is outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=1.0)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware, include_body=False)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=api_settings.cors_methods,
        allow_headers=api_settings.cors_headers,
    )
    
    # GZip middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware
    if api_settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual hosts in production
        )


def _add_exception_handlers(app: FastAPI) -> None:
    """Add exception handlers to the application"""
    
    @app.exception_handler(CopywritingException)
    async def copywriting_exception_handler(request: Request, exc: CopywritingException):
        """Handle custom copywriting exceptions"""
        logger.error(f"Copywriting exception: {exc.error_code} - {exc.message}")
        
        error_response = ErrorResponse(
            error_code=exc.error_code,
            error_message=exc.message,
            error_details=exc.details,
            request_id=exc.request_id
        )
        
        return JSONResponse(
            status_code=400,  # Default status code
            content=error_response.model_dump()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc.errors()}")
        
        error_response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Request validation failed",
            error_details={"errors": exc.errors()}
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        
        error_response = ErrorResponse(
            error_code="HTTP_ERROR",
            error_message=str(exc.detail),
            error_details={"status_code": exc.status_code}
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {type(exc).__name__} - {exc}")
        
        error_response = ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="An internal error occurred",
            error_details={"exception_type": type(exc).__name__}
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    api_settings = get_api_settings()
    uvicorn.run(
        "app:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=api_settings.reload,
        log_level="info"
    )
