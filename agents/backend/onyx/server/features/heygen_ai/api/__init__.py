from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from .core.error_handling import (
from .core.database import init_database, close_database
from .core.auth import setup_auth_middleware
from .routers import (
from .utils.helpers import generate_request_id
    from pydantic import ValidationError as PydanticValidationError
    from fastapi import HTTPException
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Application
Main API module with comprehensive error handling, middleware, and router configuration.
"""


    HeyGenBaseError,
    heygen_exception_handler,
    pydantic_validation_handler,
    http_exception_handler,
    general_exception_handler,
    ErrorCategory,
    ErrorSeverity
)
    video_router,
    user_router,
    auth_router,
    admin_router,
    analytics_router
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting HeyGen AI API...")
    await init_database()
    logger.info("HeyGen AI API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HeyGen AI API...")
    await close_database()
    logger.info("HeyGen AI API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="HeyGen AI API",
        description="AI-powered video generation API with comprehensive error handling",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add request middleware for request ID and timing
    setup_request_middleware(app)
    
    # Mount static files
    setup_static_files(app)
    
    # Include routers
    include_routers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:8080",
            "https://heygen.ai",
            "https://app.heygen.ai"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "localhost",
            "127.0.0.1",
            "heygen.ai",
            "*.heygen.ai"
        ]
    )
    
    # Setup authentication middleware
    setup_auth_middleware(app)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for comprehensive error handling"""
    
    # Register custom exception handlers
    app.add_exception_handler(HeyGenBaseError, heygen_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Register Pydantic validation error handler
    app.add_exception_handler(PydanticValidationError, pydantic_validation_handler)
    
    # Register HTTP exception handler
    app.add_exception_handler(HTTPException, http_exception_handler)


async def setup_request_middleware(app: FastAPI) -> None:
    """Setup request middleware for request tracking and timing"""
    
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        """Add request context and timing"""
        # Generate request ID
        request_id = generate_request_id()
        request.state.request_id = request_id
        
        # Add request start time
        start_time = time.time()
        
        # Add request context to state
        request.state.start_time = start_time
        request.state.user_agent = request.headers.get("user-agent", "")
        request.state.client_ip = request.client.host if request.client else "unknown"
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.state.client_ip,
                "user_agent": request.state.user_agent
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add processing time to response headers
            response.headers["X-Processing-Time"] = str(processing_time)
            response.headers["X-Request-ID"] = request_id
            
            # Log successful request
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time": processing_time
                }
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "processing_time": processing_time,
                    "error": str(e)
                },
                exc_info=True
            )
            
            # Re-raise the exception to be handled by exception handlers
            raise


def setup_static_files(app: FastAPI) -> None:
    """Setup static file serving"""
    
    # Mount static files for video outputs
    app.mount("/static/videos", StaticFiles(directory="static/videos"), name="videos")
    
    # Mount static files for thumbnails
    app.mount("/static/thumbnails", StaticFiles(directory="static/thumbnails"), name="thumbnails")
    
    # Mount static files for documentation
    app.mount("/static/docs", StaticFiles(directory="static/docs"), name="docs")


def include_routers(app: FastAPI) -> None:
    """Include all API routers"""
    
    # Include main routers
    app.include_router(
        video_router,
        prefix="/api/v1",
        tags=["videos"]
    )
    
    app.include_router(
        user_router,
        prefix="/api/v1/users",
        tags=["users"]
    )
    
    app.include_router(
        auth_router,
        prefix="/api/v1/auth",
        tags=["authentication"]
    )
    
    app.include_router(
        admin_router,
        prefix="/api/v1/admin",
        tags=["admin"]
    )
    
    app.include_router(
        analytics_router,
        prefix="/api/v1/analytics",
        tags=["analytics"]
    )


# Health check endpoint
def add_health_check(app: FastAPI) -> None:
    """Add health check endpoint"""
    
    @app.get("/health", tags=["health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
            "service": "HeyGen AI API"
        }


# Create application instance
app = create_app()

# Add health check
add_health_check(app)

# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "message": "Welcome to HeyGen AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "timestamp": time.time()
    }


# Named exports
__all__ = [
    "app",
    "create_app",
    "setup_middleware",
    "setup_exception_handlers",
    "setup_request_middleware",
    "setup_static_files",
    "include_routers",
    "add_health_check"
] 