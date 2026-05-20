"""
FastAPI Application for Facebook Posts API
Following functional programming principles and FastAPI best practices
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import asyncio
import logging
import time

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import structlog

from .api.routes import router as api_router
from .api.advanced_routes import router as advanced_router
from .api.ultimate_routes import router as ultimate_router
from .api.nextgen_routes import router as nextgen_router
from .core.config import get_settings, validate_environment
from .api.dependencies import get_service_lifespan

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Pure functions for application setup

def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration - pure function"""
    settings = get_settings()
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["*"]
    }


def get_trusted_hosts() -> list:
    """Get trusted hosts configuration - pure function"""
    settings = get_settings()
    return ["*"] if settings.debug else ["localhost", "127.0.0.1"]


def create_error_response(
    error: str,
    error_code: str,
    status_code: int,
    request: Request,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create standardized error response - pure function"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "error_code": error_code,
            "details": details or {},
            "path": str(request.url),
            "method": request.method,
            "request_id": getattr(request.state, "request_id", None),
            "timestamp": time.time()
        }
    )


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware - pure function"""
    # CORS middleware
    cors_config = get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"]
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=get_trusted_hosts()
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        response = await call_next(request)
        process_time = asyncio.get_event_loop().time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers - pure function"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return create_error_response(
            exc.detail,
            f"HTTP_{exc.status_code}",
            exc.status_code,
            request
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        return create_error_response(
            "Validation error",
            "VALIDATION_ERROR",
            422,
            request,
            {"errors": exc.errors()}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error("Unhandled exception", error=str(exc), exc_info=True)
        return create_error_response(
            "Internal server error",
            "INTERNAL_ERROR",
            500,
            request
        )


def setup_routes(app: FastAPI) -> None:
    """Setup application routes - pure function"""
    # Include API routers
    app.include_router(api_router)
    app.include_router(advanced_router)
    app.include_router(ultimate_router)
    app.include_router(nextgen_router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information"""
        settings = get_settings()
        return {
            "message": "Ultimate Facebook Posts API",
            "version": settings.api_version,
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled",
            "timestamp": time.time()
        }
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": get_settings().api_version
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Ultimate Facebook Posts System", version=get_settings().api_version)
    
    if not validate_environment():
        logger.error("Environment validation failed")
        raise RuntimeError("Environment validation failed")
    
    # Initialize services
    async with get_service_lifespan():
        yield
    
    # Shutdown
    logger.info("Shutting down Ultimate Facebook Posts System")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations - pure function"""
    settings = get_settings()
    
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
    
    # Setup components
    setup_middleware(app)
    setup_routes(app)
    setup_error_handlers(app)
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )