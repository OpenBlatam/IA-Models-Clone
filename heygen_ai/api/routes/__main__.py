from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import structlog
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from . import RouteRegistry, DependencyContainer, RouteCategory
from .base import BaseRoute
from .users import UserRoutes
from .videos import VideoRoutes
from ..models import (
from ..middleware import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
HeyGen AI FastAPI Main Application
FastAPI best practices for main application setup with models and middleware integration.
"""


# Import our custom modules
    UserCreate, UserResponse, UserListResponse, UserAuthResponse,
    VideoCreate, VideoResponse, VideoListResponse, VideoStatusResponse
)
    setup_default_middleware, RequestLoggingMiddleware, 
    ErrorHandlingMiddleware, AuthenticationMiddleware, RateLimitingMiddleware
)

logger = structlog.get_logger()

# =============================================================================
# FastAPI Application Configuration
# =============================================================================

def create_app(
    title: str = "HeyGen AI API",
    description: str = """
    Advanced AI-powered video generation and processing API.
    
    ## Features
    
    * **Video Generation**: Create AI-generated videos from text prompts
    * **Video Processing**: Upload and process existing videos
    * **User Management**: Complete user authentication and profile management
    * **Analytics**: Comprehensive video and user analytics
    
    ## Authentication
    
    This API uses Bearer token authentication. Include your token in the Authorization header:
    
    ```
    Authorization: Bearer your-token-here
    ```
    """,
    version: str = "1.0.0",
    debug: bool = False,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json"
) -> FastAPI:
    """Create FastAPI application following best practices."""
    
    # Create FastAPI app with proper configuration
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        docs_url=docs_url if not debug else None,
        redoc_url=redoc_url if not debug else None,
        openapi_url=openapi_url,
        contact={
            "name": "HeyGen AI Support",
            "email": "support@heygen.ai",
            "url": "https://heygen.ai/support"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    )
    
    # =============================================================================
    # Middleware Setup
    # =============================================================================
    
    # Configure middleware following FastAPI best practices
    middleware_config = {
        "cors_origins": [
            "https://app.heygen.ai",
            "https://api.heygen.ai",
            "http://localhost:3000"  # Development only
        ],
        "cors_credentials": True,
        "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "cors_headers": [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept"
        ],
        "compression_min_size": 1000,
        "log_request_body": debug,
        "log_request_headers": True,
        "include_traceback": debug,
        "auth_exclude_paths": [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/v1/users/login",
            "/api/v1/users/register"
        ],
        "rate_limit_requests": 60,
        "rate_limit_burst": 10
    }
    
    # Setup default middleware
    setup_default_middleware(app, middleware_config)
    
    # =============================================================================
    # Dependency Container Setup
    # =============================================================================
    
    # Initialize dependency container
    dependency_container = DependencyContainer()
    
    # Register dependencies
    dependency_container.register("database", get_database_session)
    dependency_container.register("external_api", get_external_api_client)
    dependency_container.register("file_storage", get_file_storage)
    dependency_container.register("cache", get_cache_client)
    
    # Store in app state
    app.state.dependencies = dependency_container
    
    # =============================================================================
    # Route Registry Setup
    # =============================================================================
    
    # Initialize route registry
    route_registry = RouteRegistry()
    
    # Register route categories
    route_registry.register_category(RouteCategory.USERS, "User Management")
    route_registry.register_category(RouteCategory.VIDEOS, "Video Processing")
    route_registry.register_category(RouteCategory.SYSTEM, "System Operations")
    
    # Store in app state
    app.state.routes = route_registry
    
    # =============================================================================
    # Route Registration
    # =============================================================================
    
    # Register user routes
    user_routes = UserRoutes(dependency_container)
    app.include_router(
        user_routes.router,
        prefix="/api/v1/users",
        tags=["users"],
        responses={
            400: {"description": "Bad request"},
            401: {"description": "Unauthorized"},
            403: {"description": "Forbidden"},
            404: {"description": "Not found"},
            500: {"description": "Internal server error"}
        }
    )
    
    # Register video routes
    video_routes = VideoRoutes(dependency_container)
    app.include_router(
        video_routes.router,
        prefix="/api/v1/videos",
        tags=["videos"],
        responses={
            400: {"description": "Bad request"},
            401: {"description": "Unauthorized"},
            403: {"description": "Forbidden"},
            404: {"description": "Not found"},
            500: {"description": "Internal server error"}
        }
    )
    
    # =============================================================================
    # System Routes
    # =============================================================================
    
    @app.get("/health", tags=["system"])
    async def health_check():
        """Health check endpoint following FastAPI best practices."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": version
        }
    
    @app.get("/api/info", tags=["system"])
    async def api_info():
        """API information endpoint following FastAPI best practices."""
        return {
            "name": title,
            "version": version,
            "description": "HeyGen AI API for video generation and processing",
            "endpoints": {
                "users": "/api/v1/users",
                "videos": "/api/v1/videos",
                "docs": docs_url,
                "health": "/health"
            }
        }
    
    @app.get("/api/metrics", tags=["system"])
    async def get_metrics():
        """API metrics endpoint following FastAPI best practices."""
        return {
            "requests_total": 0,  # Implement actual metrics
            "requests_per_minute": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "active_users": 0,
            "videos_processed": 0
        }
    
    # =============================================================================
    # Error Handlers
    # =============================================================================
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions following FastAPI best practices."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.detail,
                "error_code": f"HTTP_{exc.status_code}",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors following FastAPI best practices."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Validation error",
                "error_code": "VALIDATION_ERROR",
                "error_details": exc.errors(),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors following FastAPI best practices."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Data validation error",
                "error_code": "VALIDATION_ERROR",
                "error_details": exc.errors(),
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions following FastAPI best practices."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.error(
            "Unhandled exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            request_id=request_id,
            method=request.method,
            url=str(request.url)
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    # =============================================================================
    # Lifecycle Events
    # =============================================================================
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event following FastAPI best practices."""
        logger.info("Starting HeyGen AI API", version=version)
        
        # Initialize dependencies
        await dependency_container.startup()
        
        # Initialize route registry
        route_registry.initialize()
        
        logger.info("HeyGen AI API started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event following FastAPI best practices."""
        logger.info("Shutting down HeyGen AI API")
        
        # Cleanup dependencies
        await dependency_container.shutdown()
        
        logger.info("HeyGen AI API shutdown completed")
    
    # =============================================================================
    # Development Routes
    # =============================================================================
    
    if debug:
        @app.get("/debug/routes", tags=["debug"])
        async def debug_routes():
            """Debug routes endpoint for development."""
            return {
                "routes": route_registry.get_all_routes(),
                "categories": route_registry.get_categories(),
                "dependencies": dependency_container.get_registered_services()
            }
        
        @app.get("/debug/health", tags=["debug"])
        async def debug_health():
            """Detailed health check for development."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": version,
                "debug": True,
                "dependencies": {
                    "database": "connected",
                    "external_api": "connected",
                    "file_storage": "connected",
                    "cache": "connected"
                }
            }
    
    return app

# =============================================================================
# Dependency Functions
# =============================================================================

async def get_database_session():
    """Get database session following FastAPI best practices."""
    # Implement database session creation
    # This is a placeholder implementation
    return {"session": "database_session"}

async def get_external_api_client():
    """Get external API client following FastAPI best practices."""
    # Implement external API client creation
    # This is a placeholder implementation
    return {"client": "external_api_client"}

async def get_file_storage():
    """Get file storage client following FastAPI best practices."""
    # Implement file storage client creation
    # This is a placeholder implementation
    return {"storage": "file_storage_client"}

async def get_cache_client():
    """Get cache client following FastAPI best practices."""
    # Implement cache client creation
    # This is a placeholder implementation
    return {"cache": "cache_client"}

# =============================================================================
# Application Factory
# =============================================================================

def create_development_app() -> FastAPI:
    """Create development FastAPI application."""
    return create_app(
        title="HeyGen AI API (Development)",
        debug=True,
        docs_url="/docs",
        redoc_url="/redoc"
    )

def create_production_app() -> FastAPI:
    """Create production FastAPI application."""
    return create_app(
        title="HeyGen AI API",
        debug=False,
        docs_url=None,
        redoc_url=None
    )

# =============================================================================
# Main Application Instance
# =============================================================================

# Create the main application instance
app = create_app()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "create_app",
    "create_development_app",
    "create_production_app",
    "app"
] 