from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from ...shared.config import EnterpriseConfig
from ...infrastructure import (
from ..middleware import EnterpriseMiddlewareStack
from ..endpoints import HealthEndpoints, MetricsEndpoints, APIEndpoints
import logging
from typing import Any, List, Dict, Optional
"""
Enterprise API Factory
=====================

Main factory for creating the enterprise FastAPI application with all layers integrated.
"""


# Import layers
    MultiTierCacheService,
    PrometheusMetricsService,
    CircuitBreakerService,
    HealthCheckService,
    RedisRateLimitService
)

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Dependency injection container for all services."""
    
    def __init__(self, config: EnterpriseConfig):
        
    """__init__ function."""
self.config = config
        
        # Initialize services
        self.cache_service = MultiTierCacheService(
            redis_url=config.redis_url,
            max_memory_items=config.cache_max_size
        )
        
        self.metrics_service = PrometheusMetricsService()
        
        self.circuit_breaker = CircuitBreakerService(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        
        self.health_service = HealthCheckService(config)
        
        self.rate_limit_service = RedisRateLimitService(
            redis_url=config.redis_url,
            requests_per_window=config.rate_limit_requests,
            window_size=config.rate_limit_window
        )
        
    async def initialize(self) -> Any:
        """Initialize all services."""
        await self.cache_service.initialize()
        await self.rate_limit_service.initialize()
        self.health_service.register_default_checks()
        
        # Register cache health check
        self.health_service.register_check(
            "cache",
            lambda: {"healthy": True, "message": "Cache service operational"}
        )
        
        logger.info("All services initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting Enterprise API v2.0.0")
    
    try:
        await app.state.container.initialize()
        logger.info("✅ All services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enterprise API")


def create_enterprise_app(config: EnterpriseConfig = None) -> FastAPI:
    """Create and configure the enterprise FastAPI application."""
    
    if config is None:
        config = EnterpriseConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description="🚀 Enterprise API - Clean Architecture Implementation",
        lifespan=lifespan,
        debug=config.debug
    )
    
    # Initialize service container
    app.state.container = ServiceContainer(config)
    
    # Add middleware stack
    middleware_stack = EnterpriseMiddlewareStack(
        config=config,
        metrics_service=app.state.container.metrics_service,
        rate_limit_service=app.state.container.rate_limit_service
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        **config.get_cors_config()
    )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware
    app.middleware("http")(middleware_stack.request_id_middleware)
    app.middleware("http")(middleware_stack.performance_monitoring_middleware)
    app.middleware("http")(middleware_stack.security_headers_middleware)
    app.middleware("http")(middleware_stack.rate_limiting_middleware)
    
    # Register endpoints
    health_endpoints = HealthEndpoints(app.state.container.health_service)
    metrics_endpoints = MetricsEndpoints(app.state.container.metrics_service)
    api_endpoints = APIEndpoints(
        cache_service=app.state.container.cache_service,
        circuit_breaker=app.state.container.circuit_breaker
    )
    
    # Health routes
    app.include_router(health_endpoints.router, tags=["Health"])
    
    # Metrics routes (if enabled)
    if config.enable_metrics:
        app.include_router(metrics_endpoints.router, tags=["Metrics"])
    
    # API routes
    app.include_router(api_endpoints.router, prefix="/api/v1", tags=["API"])
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint showing service information."""
        return {
            "service": config.app_name,
            "version": config.app_version,
            "status": "operational",
            "architecture": "Clean Architecture",
            "features": [
                "Multi-tier caching",
                "Circuit breaker protection", 
                "Rate limiting",
                "Health checks",
                "Metrics collection",
                "Enterprise middleware stack"
            ],
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics" if config.enable_metrics else None,
                "api": "/api/v1",
                "docs": "/docs"
            }
        }
    
    # Global exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            }
        )
    
    logger.info(f"✅ Enterprise API created - Environment: {config.environment}")
    return app 