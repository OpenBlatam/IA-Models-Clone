from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from ..config.settings import get_settings, get_api_config, get_security_config
from ..core.engine import CopywritingEngine
from .routes import router
from .middleware import RateLimitMiddleware, LoggingMiddleware
from .exceptions import setup_exception_handlers
        from ..core.engine import CopywritingEngine
        from ..config.settings import get_engine_config
from typing import Any, List, Dict, Optional
import asyncio
"""
FastAPI Application Factory
==========================

Main application factory for the copywriting system.
"""




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[CopywritingEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engine
    
    # Startup
    logger.info("Starting Copywriting System...")
    
    try:
        # Initialize engine
        
        config = get_engine_config()
        engine = CopywritingEngine(config)
        await engine.initialize()
        
        logger.info("Copywriting System started successfully")
        
    except Exception as e:
        logger.error(f"Error starting Copywriting System: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Copywriting System...")
    
    try:
        if engine:
            await engine.shutdown()
        
        logger.info("Copywriting System shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down Copywriting System: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Get configuration
    settings = get_settings()
    api_config = get_api_config()
    security_config = get_security_config()
    
    # Create FastAPI application
    app = FastAPI(
        title="Copywriting System API",
        description="High-performance copywriting generation system with advanced AI capabilities",
        version=settings.version,
        docs_url="/docs" if api_config.enable_docs else None,
        redoc_url="/redoc" if api_config.enable_docs else None,
        lifespan=lifespan
    )
    
    # Add middleware
    if api_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.cors_origins,
            allow_credentials=security_config.cors_credentials,
            allow_methods=security_config.cors_methods,
            allow_headers=security_config.cors_headers,
        )
    
    if api_config.enable_gzip:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    
    if api_config.enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Add root endpoint
    @app.get("/")
    async def root():
        
    """root function."""
return {
            "message": "Copywriting System API",
            "version": settings.version,
            "status": "running",
            "docs": "/docs" if api_config.enable_docs else None
        }
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    return app


def get_app() -> FastAPI:
    """Get the FastAPI application instance"""
    return create_app()


def get_engine() -> Optional[CopywritingEngine]:
    """Get the global engine instance"""
    return engine 