"""
Burnout Prevention AI - Main Application
========================================
Servidor principal para el sistema de prevención de burnout.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

try:
    import structlog
    from structlog.stdlib import LoggerFactory
    structlog.configure(logger_factory=LoggerFactory())
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _has_slowapi = True
except ImportError:
    _has_slowapi = False

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _has_prometheus = True
except ImportError:
    _has_prometheus = False

from .config.app_config import get_config
from .api.routes import router
from .services.continuous_burnout_service import get_continuous_service

config = get_config()

# Note: Metrics are now in core.metrics module for better organization

# Rate limiter (if available)
if _has_slowapi:
    limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events.
    
    Inicializa y limpia el servicio de procesamiento continuo.
    Similar al manejo de estado en continuous-agent.
    """
    logger.info("Starting Burnout Prevention AI service")
    
    # Initialize continuous service
    continuous_service = get_continuous_service()
    logger.info("Continuous processing service initialized", started=False)
    
    # Cleanup expired cache entries periodically (background task)
    import asyncio
    from .core.cache import clear_expired, get_cache_stats
    
    async def periodic_cache_cleanup():
        """Periodically clean expired cache entries."""
        from .core.constants import CACHE_CLEANUP_INTERVAL
        from .core.logging_helpers import log_error, log_debug
        
        while True:
            try:
                await asyncio.sleep(CACHE_CLEANUP_INTERVAL)
                removed = clear_expired()
                stats = get_cache_stats()
                log_debug("Cache cleanup completed", context={**stats, "removed": removed})
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Cache cleanup error", e)
    
    cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Burnout Prevention AI service")
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Stop continuous service if active
    if continuous_service.is_active:
        logger.info("Stopping continuous processing")
        try:
            await continuous_service.stop()
        except Exception as e:
            from .core.logging_helpers import log_error
            log_error("Error stopping continuous service", e)
    
    # Close OpenRouter client if exists
    try:
        from .api.routes.burnout_routes import _openrouter_client_instance
        from .core.logging_helpers import log_error, log_info
        if _openrouter_client_instance is not None:
            try:
                await _openrouter_client_instance.close()
                log_info("OpenRouter client closed")
            except Exception as e:
                log_error("Error closing OpenRouter client", e)
    except ImportError:
        # Module might not be loaded yet
        pass
    
    logger.info("Shutdown complete")


app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="AI-powered burnout prevention and wellness assistant",
    lifespan=lifespan,
    default_response_class=ORJSONResponse  # Faster JSON responses
)

# CORS (more secure configuration)
cors_origins = ["*"] if config.debug else []  # Restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# Rate limiting (if available)
if _has_slowapi:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    """Add security headers to all responses."""
    from .core.security import get_security_headers
    
    response = await call_next(request)
    
    # Add security headers
    security_headers = get_security_headers()
    for header, value in security_headers.items():
        response.headers[header] = value
    
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """
    Log all requests with timing (optimized).
    
    Excludes health check and metrics endpoints from verbose logging
    to reduce log noise while maintaining observability.
    """
    # Skip verbose logging for health/metrics endpoints
    path = request.url.path
    skip_verbose_logging = path in ("/health", "/metrics", "/")
    
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request (use debug for health/metrics, info for others)
    log_level = logger.debug if skip_verbose_logging else logger.info
    log_level(
        "Request processed",
        method=request.method,
        path=path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")[:100]  # Limit length
    )
    
    # Prometheus metrics (if available) - always record
    if _has_prometheus:
        from .core.metrics import record_api_request
        record_api_request(
            endpoint=path,
            method=request.method,
            status=response.status_code,
            duration=duration
        )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(round(duration, 3))
    
    return response

# Routes
app.include_router(router)


@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with service information.
    
    Returns service metadata, available features, and API endpoints.
    """
    return {
        "service": config.app_name,
        "version": config.app_version,
        "status": "running",
        "features": {
            "rate_limiting": _has_slowapi,
            "prometheus": _has_prometheus,
            "structured_logging": True
        },
        "endpoints": {
            "assess": "/api/v1/assess",
            "wellness-check": "/api/v1/wellness-check",
            "coping-strategies": "/api/v1/coping-strategies",
            "chat": "/api/v1/chat",
            "progress": "/api/v1/progress",
            "trends": "/api/v1/trends",
            "resources": "/api/v1/resources",
            "personalized-plan": "/api/v1/personalized-plan",
            "health": "/api/v1/health",
            "docs": "/docs",
            "metrics": "/metrics" if _has_prometheus else None
        }
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns Prometheus-formatted metrics for monitoring.
    Requires prometheus-client to be installed.
    """
    if not _has_prometheus:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prometheus client not available"
        )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    # Use uvloop if available (faster on Linux/macOS)
    try:
        import uvloop
        uvloop.install()
        logger.info("Using uvloop for better performance")
    except ImportError:
        pass
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_config=None,  # Use structlog if available
        access_log=True
    )

