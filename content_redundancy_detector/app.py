"""
Content Redundancy Detector - Functional FastAPI Application
Following best practices: functional programming, RORO pattern, async operations

REFACTORED: Now uses centralized core modules for better organization
"""

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Core modules (refactored)
from core.config import settings
from core.logging_config import get_logger
from core.error_handlers import register_exception_handlers
from core.dependencies import get_cache_manager, get_rate_limiter, get_health_checker
from core.initialization import lifespan
# Middleware
from middleware import (
    LoggingMiddleware, ErrorHandlingMiddleware,
    SecurityMiddleware, PerformanceMiddleware, RateLimitMiddleware
)
from compression_middleware import compression_middleware, streaming_compression_middleware

# Advanced features
from health_checks_advanced import health_checker, create_health_response
from openapi_enhanced import create_enhanced_openapi_schema, create_enhanced_docs_html, create_enhanced_redoc_html

# Routers
from routers import router

# Advanced microservices (optional)
try:
    from api_gateway import APIGatewayHeaders
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "AI Ultimate Content Redundancy Detector with Advanced ML, "
            "Real-time Processing, Cloud Integration, Security, Monitoring, and Automation"
        ),
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # Register exception handlers (refactored)
    register_exception_handlers(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add correlation middleware first for request tracking
    try:
        from api.middleware.correlation import CorrelationMiddleware
        app.add_middleware(CorrelationMiddleware)
    except ImportError:
        logger.warning("CorrelationMiddleware not available, skipping")
    
    # Add custom middleware (order matters - last added is first executed)
    app.add_middleware(streaming_compression_middleware)
    app.add_middleware(compression_middleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add API Gateway middleware if available
    if ADVANCED_FEATURES_AVAILABLE:
        @app.middleware("http")
        async def api_gateway_middleware(request: Request, call_next):
            """API Gateway integration middleware"""
            response = await call_next(request)
            if hasattr(response, 'headers'):
                response = APIGatewayHeaders.add_gateway_headers(response, request)
            return response
    
    # Include routers
    app.include_router(router, prefix="/api/v1", tags=["API"])
    app.include_router(router, prefix="", tags=["Root"])  # Keep root endpoints
    
    # Include modular API routes
    try:
        from api.routes import api_router
        app.include_router(api_router, prefix="/api/v1", tags=["API Modular"])
    except ImportError:
        pass  # Fallback to old router if new structure not available
    
    # Prometheus metrics endpoint
    try:
        from prometheus_metrics import router as metrics_router
        app.include_router(metrics_router, tags=["Monitoring"])
    except ImportError:
        pass
    
    # Enhanced OpenAPI documentation
    app.openapi = lambda: create_enhanced_openapi_schema(app)
    
    # Custom documentation endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_docs():
        return create_enhanced_docs_html(app)
    
    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc():
        return create_enhanced_redoc_html(app)
    
    # Enhanced health check endpoint
    @app.get("/health/advanced", tags=["Health & Monitoring"])
    async def advanced_health_check():
        """Advanced health check with dependency monitoring"""
        health_checker_instance = get_health_checker()
        health_data = await health_checker_instance.run_all_checks()
        return create_health_response(health_data)
    
    # Cache statistics endpoint
    @app.get("/cache/stats", tags=["Monitoring"])
    async def cache_statistics():
        """Get cache statistics and performance metrics"""
        cache_manager_instance = get_cache_manager()
        stats = cache_manager_instance.get_stats()
        return {
            "success": True,
            "data": stats,
            "error": None,
            "timestamp": time.time()
        }
    
    # Policy guard endpoint
    @app.get("/policies/summary", tags=["Monitoring", "Policy"])
    async def policy_summary():
        """Get policy guardrails summary"""
        try:
            from policy_guard import default_policy_guard
            summary = default_policy_guard.get_policy_summary()
            return {
                "success": True,
                "data": summary,
                "error": None,
                "timestamp": time.time()
            }
        except ImportError:
            return {
                "success": False,
                "data": None,
                "error": "Policy guard not available",
                "timestamp": time.time()
            }
    
    # Rate limit status endpoint
    @app.get("/rate-limit/status", tags=["Monitoring"])
    async def rate_limit_status(request: Request):
        """Get current rate limit status"""
        from core.dependencies import get_ip_address
        
        rate_limiter_instance = get_rate_limiter()
        ip_address = get_ip_address(request)
        usage = rate_limiter_instance.get_usage(ip_address=ip_address)
        
        return {
            "success": True,
            "data": usage,
            "error": None,
            "timestamp": time.time()
        }
    
    logger.info("FastAPI application created and configured")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    import time
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )