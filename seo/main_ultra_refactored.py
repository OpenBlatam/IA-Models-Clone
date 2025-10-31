from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
from config.loader import load_config
from services.seo_service_factory import SEOServiceFactory, cleanup_factory
from api.routes import router as seo_router
from api.middleware import (
from typing import Any, List, Dict, Optional
import logging
"""
Main entry point for Ultra-Optimized SEO Service.
Clean architecture implementation with dependency injection.
"""


# Import our modules
    RequestLoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware
)

# Prometheus metrics
REQUEST_COUNT = Counter('seo_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('seo_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('seo_active_requests', 'Active requests')
ERROR_COUNT = Counter('seo_errors_total', 'Total errors', ['type'])

# Global variables
app: FastAPI = None
factory: SEOServiceFactory = None
config: Dict[str, Any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Ultra-Optimized SEO Service...")
    
    # Load configuration
    global config
    config = load_config()
    logger.info(f"📋 Configuration loaded: {config.get('app', {}).get('environment', 'development')}")
    
    # Initialize factory
    global factory
    factory = SEOServiceFactory(config)
    logger.info("🏭 SEO Service Factory initialized")
    
    # Health check
    try:
        seo_service = factory.get_seo_service()
        health = await seo_service.health_check()
        logger.info(f"✅ Health check passed: {health}")
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        sys.exit(1)
    
    logger.info("✅ SEO Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SEO Service...")
    
    try:
        await cleanup_factory()
        logger.info("✅ Factory cleanup completed")
    except Exception as e:
        logger.error(f"❌ Factory cleanup failed: {e}")
    
    logger.info("👋 SEO Service shutdown completed")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app_config = config.get('app', {})
    
    app = FastAPI(
        title=app_config.get('name', 'SEO Service Ultra-Optimized'),
        version=app_config.get('version', '2.0.0'),
        description="Ultra-optimized SEO analysis service with clean architecture",
        docs_url="/docs" if app_config.get('debug', False) else None,
        redoc_url="/redoc" if app_config.get('debug', False) else None,
        lifespan=lifespan
    )
    
    # Add middleware
    add_middleware(app)
    
    # Add routes
    add_routes(app)
    
    # Add exception handlers
    add_exception_handlers(app)
    
    return app


def add_middleware(app: FastAPI):
    """Add middleware to the application."""
    # CORS
    cors_config = config.get('cors', {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get('origins', ["*"]),
        allow_credentials=cors_config.get('credentials', True),
        allow_methods=cors_config.get('methods', ["*"]),
        allow_headers=cors_config.get('headers', ["*"]),
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RateLimitMiddleware, config=config.get('rate_limit', {}))
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("🔧 Middleware configured")


def add_routes(app: FastAPI):
    """Add routes to the application."""
    # Include SEO routes
    app.include_router(seo_router, prefix="/api/v2", tags=["SEO"])
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check."""
        return {
            "status": "healthy",
            "service": "SEO Service Ultra-Optimized",
            "version": config.get('app', {}).get('version', '2.0.0'),
            "environment": config.get('app', {}).get('environment', 'development')
        }
    
    # Detailed health check
    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with component status."""
        if factory:
            return factory.get_health_status()
        return {"status": "factory_not_initialized"}
    
    # Metrics endpoint
    @app.get("/metrics", tags=["Metrics"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            prometheus_client.generate_latest(),
            media_type="text/plain"
        )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "SEO Service Ultra-Optimized",
            "version": config.get('app', {}).get('version', '2.0.0'),
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }
    
    logger.info("🛣️ Routes configured")


def add_exception_handlers(app: FastAPI):
    """Add exception handlers to the application."""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        ERROR_COUNT.labels(type=type(exc).__name__).inc()
        
        logger.error(f"Unhandled exception: {exc}")
        
        return {
            "error": "Internal server error",
            "message": str(exc) if config.get('app', {}).get('debug', False) else "Something went wrong",
            "status_code": 500
        }
    
    logger.info("🛡️ Exception handlers configured")


async def shutdown_handler(signum, frame) -> Any:
    """Handle shutdown signals."""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    
    if factory:
        await cleanup_factory()
    
    sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info("📡 Signal handlers configured")


def setup_logging():
    """Setup structured logging."""
    log_config = config.get('logging', {})
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
        level=log_config.get('level', 'INFO'),
        colorize=True
    )
    
    # Add file handler
    if log_config.get('file_enabled', True):
        logger.add(
            log_config.get('file_path', 'logs/seo-service.log'),
            rotation=log_config.get('rotation', '100 MB'),
            retention=log_config.get('retention', '30 days'),
            compression=log_config.get('compression', 'zip'),
            format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"),
            level=log_config.get('level', 'INFO')
        )
    
    # Add JSON handler for production
    if config.get('app', {}).get('environment') == 'production':
        logger.add(
            log_config.get('json_path', 'logs/seo-service.json'),
            format="{time} | {level} | {extra}",
            serialize=True,
            level=log_config.get('level', 'INFO')
        )
    
    logger.info("📝 Logging configured")


def main():
    """Main entry point."""
    try:
        # Setup logging first
        setup_logging()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Load configuration
        global config
        config = load_config()
        
        # Create application
        global app
        app = create_app()
        
        # Server configuration
        server_config = config.get('server', {})
        
        # Start server
        uvicorn.run(
            app,
            host=server_config.get('host', '0.0.0.0'),
            port=server_config.get('port', 8000),
            workers=server_config.get('workers', 1),
            log_level=config.get('logging', {}).get('level', 'info').lower(),
            access_log=True,
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        sys.exit(1)


match __name__:
    case "__main__":
    main() 