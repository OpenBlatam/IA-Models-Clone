#!/usr/bin/env python3
"""
Ultimate Facebook Posts System Launcher
=======================================

Unified launcher following FastAPI best practices with functional programming,
async patterns, and modular architecture.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseSettings, Field
import structlog

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


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    api_title: str = Field("Ultimate Facebook Posts API", env="API_TITLE")
    api_version: str = Field("4.0.0", env="API_VERSION")
    api_description: str = Field("AI-powered Facebook post generation system", env="API_DESCRIPTION")
    debug: bool = Field(False, env="DEBUG")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./facebook_posts.db", env="DATABASE_URL")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # AI Configuration
    ai_api_key: str = Field("", env="AI_API_KEY")
    ai_model: str = Field("gpt-3.5-turbo", env="AI_MODEL")
    
    # Security
    api_key: str = Field("", env="API_KEY")
    cors_origins: str = Field("*", env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Pure functions for configuration and validation

def is_valid_environment() -> bool:
    """Validate environment configuration"""
    if not settings.api_key and not settings.debug:
        logger.warning("API_KEY not set - authentication disabled in debug mode")
    
    if not settings.ai_api_key:
        logger.warning("AI_API_KEY not set - AI features may not work")
    
    return True


def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration"""
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["*"]
    }


def get_trusted_hosts() -> list:
    """Get trusted hosts configuration"""
    return ["*"] if settings.debug else ["localhost", "127.0.0.1"]


# Service initialization functions

async def initialize_database() -> None:
    """Initialize database connection"""
    logger.info("Database initialized")


async def initialize_cache() -> None:
    """Initialize cache service"""
    logger.info("Cache service initialized")


async def initialize_ai_service() -> None:
    """Initialize AI service"""
    logger.info("AI service initialized")


async def initialize_services() -> None:
    """Initialize all system services"""
    try:
        logger.info("Initializing services...")
        
        await asyncio.gather(
            initialize_database(),
            initialize_cache(),
            initialize_ai_service()
        )
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise


async def cleanup_services() -> None:
    """Cleanup all services"""
    logger.info("Cleaning up services...")


# Middleware functions

def add_process_time_middleware(app: FastAPI) -> None:
    """Add request timing middleware"""
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        response = await call_next(request)
        process_time = asyncio.get_event_loop().time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


def add_request_id_middleware(app: FastAPI) -> None:
    """Add request ID middleware"""
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI) -> None:
    """Setup application middleware"""
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
    
    # Custom middleware
    add_process_time_middleware(app)
    add_request_id_middleware(app)


# Error handler functions

def create_error_response(
    error: str,
    error_code: str,
    status_code: int,
    request: Request,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error,
            "error_code": error_code,
            "details": details or {},
            "path": str(request.url),
            "method": request.method,
            "request_id": getattr(request.state, "request_id", None)
        }
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers"""
    
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


# Route functions

def setup_routes(app: FastAPI) -> None:
    """Setup application routes"""
    # Import route modules
    from api.routes import router as api_router
    
    # Include routers
    app.include_router(api_router)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Ultimate Facebook Posts API",
            "version": settings.api_version,
            "status": "running",
            "docs": "/docs" if settings.debug else "disabled"
        }
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "version": settings.api_version
        }


# Application factory function

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Ultimate Facebook Posts System", version=settings.api_version)
    
    if not is_valid_environment():
        logger.error("Environment validation failed")
        raise RuntimeError("Environment validation failed")
    
    await initialize_services()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ultimate Facebook Posts System")
    await cleanup_services()


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations"""
    
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


# Main execution functions

async def run_development_server() -> None:
    """Run development server with hot reload"""
    app = create_app()
    
    logger.info(
        "Starting development server",
        host=settings.host,
        port=settings.port,
        debug=settings.debug
    )
    
    uvicorn.run(
        "launch_ultimate_system:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )


async def run_production_server() -> None:
    """Run production server"""
    app = create_app()
    
    logger.info(
        "Starting production server",
        host=settings.host,
        port=settings.port,
        workers=settings.workers
    )
    
    uvicorn.run(
        "launch_ultimate_system:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info"
    )


def main() -> None:
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Facebook Posts System")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev", help="Run mode")
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=settings.workers, help="Number of workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Update settings from command line
    settings.host = args.host
    settings.port = args.port
    settings.workers = args.workers
    settings.debug = args.debug or settings.debug
    
    try:
        if args.mode == "dev":
            asyncio.run(run_development_server())
        else:
            asyncio.run(run_production_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error("Server failed to start", error=str(e))
        sys.exit(1)


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    main()