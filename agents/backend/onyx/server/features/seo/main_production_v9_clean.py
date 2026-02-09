from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvloop
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from shared.core.config import get_settings, settings
from shared.core.logging import setup_logging, get_logger, log_startup, log_shutdown
from shared.core.container import Container
from presentation.api.seo_routes import router as seo_router
from presentation.api.health_routes import router as health_router
from presentation.api.metrics_routes import router as metrics_routes
from presentation.middleware.performance_middleware import PerformanceMiddleware
from presentation.middleware.error_middleware import ErrorMiddleware
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Ultra-Optimized SEO Service v9 - Clean Architecture
Maximum Performance with Fastest Libraries
"""


# Ultra-fast imports

# Application imports

# Configure uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global variables
start_time = time.time()
container = Container()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    log_startup(
        version=settings.version,
        environment=settings.environment,
        host=settings.host,
        port=settings.port,
        workers=settings.workers
    )
    
    try:
        # Initialize container
        await container.initialize()
        logger.info("Application container initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Error during application startup", error=str(e))
        raise
    finally:
        # Shutdown
        try:
            await container.cleanup()
            logger.info("Application container cleaned up successfully")
        except Exception as e:
            logger.error("Error during application cleanup", error=str(e))
        
        log_shutdown(reason="normal_shutdown")


def create_app() -> FastAPI:
    """Create FastAPI application with all middleware and routes"""
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Ultra-Fast SEO Analysis Service with Clean Architecture",
        docs_url=settings.docs_url if settings.debug else None,
        redoc_url=settings.redoc_url if settings.debug else None,
        openapi_url=settings.openapi_url if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
    
    # Add GZip middleware
    if settings.enable_compression:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(ErrorMiddleware)
    
    # Add Prometheus instrumentation
    Instrumentator().instrument(app).expose(app, include_in_schema=False, should_gzip=True)
    
    # Include routers
    app.include_router(seo_router, prefix=settings.api_prefix)
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(metrics_routes, prefix=settings.api_prefix)
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "service": settings.app_name,
            "version": settings.version,
            "status": "running",
            "environment": settings.environment,
            "message": "Ultra-Fast SEO Service v9 - Clean Architecture"
        }
    
    return app


# Create application instance
app = create_app()


def signal_handler(signum, frame) -> Any:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Run with ultra-optimized settings
    uvicorn.run(
        "main_production_v9_clean:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        loop="uvloop",
        http="httptools",
        access_log=False,
        log_level="error" if not settings.debug else "info",
        server_header=False,
        date_header=False,
        forwarded_allow_ips="*",
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 