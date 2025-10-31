"""
Main FastAPI application with improved architecture
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import structlog

from .config.settings import get_settings
from .config.database import close_db_connections
from .core.exceptions import BlogSystemException
from .core.middleware import add_middleware
from .api.v1.api import api_router
from .utils.logging import configure_logging, get_logger

# Configure logging
settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up the application...")
    yield
    # Shutdown
    logger.info("Shutting down the application...")
    await close_db_connections()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add middleware
    add_middleware(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
        )
    
    # Include API router
    app.include_router(api_router, prefix="/api/v1")
    
    # Add exception handlers
    add_exception_handlers(app)
    
    # Add health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.api_version,
            "timestamp": time.time()
        }
    
    return app


def add_exception_handlers(app: FastAPI):
    """Add custom exception handlers."""
    
    @app.exception_handler(BlogSystemException)
    async def blog_system_exception_handler(request: Request, exc: BlogSystemException):
        """Handle custom blog system exceptions."""
        logger.error(
            "Blog system exception",
            error=exc.detail,
            error_code=exc.error_code,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_code": exc.error_code,
                "timestamp": time.time(),
                "path": request.url.path
            },
            headers=exc.headers
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "detail": exc.errors(),
                "timestamp": time.time(),
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(
            "Unhandled exception",
            error=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "timestamp": time.time(),
                "path": request.url.path
            }
        )


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )






























