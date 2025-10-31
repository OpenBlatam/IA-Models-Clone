from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from .dependencies import create_app, lifespan, app_state
from .routes import video_router, system_router
from ..core.error_handler import error_handler, ErrorContext
from ..core.exceptions import AIVideoError
        import uuid
    import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
🚀 FASTAPI MAIN - AI VIDEO SYSTEM
=================================

Main FastAPI application for the AI Video system.
"""



# Import dependencies and routes

# Import core components

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 AI Video System starting up...")
    
    # Initialize resources (database, cache, etc.)
    # In a real application, you would:
    # - Initialize database connections
    # - Set up caching systems
    # - Load AI models
    # - Start background workers
    
    yield
    
    # Shutdown
    logger.info("🛑 AI Video System shutting down...")
    
    # Cleanup resources
    # In a real application, you would:
    # - Close database connections
    # - Stop background workers
    # - Save any pending data
    # - Clean up temporary files


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="AI Video System",
    description="High-performance video processing API with AI enhancement",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    responses={
        400: {"description": "Bad request", "model": ErrorResponse},
        401: {"description": "Unauthorized", "model": ErrorResponse},
        403: {"description": "Forbidden", "model": ErrorResponse},
        404: {"description": "Not found", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        429: {"description": "Too many requests", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        502: {"description": "Bad gateway", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
    }
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# Request/Response middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Middleware for request tracking and error handling."""
    # Increment active requests
    app_state.increment_requests()
    
    # Add request ID to headers if not present
    if "X-Request-ID" not in request.headers:
        request.headers.__dict__["_list"].append(
            (b"x-request-id", str(uuid.uuid4()).encode())
        )
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Handle unexpected errors
        error_context = ErrorContext(
            operation=request.url.path,
            user_id=request.headers.get("X-User-ID"),
            request_id=request.headers.get("X-Request-ID")
        )
        
        error = error_handler.handle_error(e, error_context)
        logger.error(f"Request failed: {error.message}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "details": {"type": type(e).__name__},
                "timestamp": asyncio.get_event_loop().time()
            }
        )
    finally:
        # Decrement active requests
        app_state.decrement_requests()


# Exception handlers
@app.exception_handler(AIVideoError)
async def ai_video_error_handler(request: Request, exc: AIVideoError):
    """Handle AI Video system errors."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    error = error_handler.handle_error(exc, error_context)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": error.message,
            "error_code": error.error_code or "AI_VIDEO_ERROR",
            "details": error.details,
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    logger.error(f"Validation Error: {exc.errors()}", extra={"context": error_context.to_dict()})
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": {"errors": exc.errors()},
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    logger.error(f"HTTP Exception: {exc.detail}", extra={"context": error_context.to_dict()})
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions."""
    error_context = ErrorContext(
        operation=request.url.path,
        user_id=request.headers.get("X-User-ID"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    error = error_handler.handle_error(exc, error_context)
    logger.error(f"Generic Exception: {error.message}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {"type": type(exc).__name__},
            "timestamp": asyncio.get_event_loop().time()
        }
    )


# ============================================================================
# ROUTER INCLUSION
# ============================================================================

# Include routers
app.include_router(video_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1/system", tags=["system"])


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        dict: API information and available endpoints
    """
    return {
        "message": "AI Video System API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "videos": "/api/v1/videos",
            "system": "/api/v1/system",
            "health": "/health"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "version": "1.0.0",
        "uptime": app_state.get_uptime(),
        "active_requests": app_state.active_requests
    }


# OpenAPI customization
def custom_openapi():
    """Customize OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AI Video System API",
        version="1.0.0",
        description="Advanced AI video generation system with performance optimization",
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # Run the application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    ) 