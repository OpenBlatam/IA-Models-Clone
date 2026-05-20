"""
Middleware configuration for the FastAPI application.

Extracted from main.py for better organization and separation of concerns.
"""

import logging
from typing import Optional, Type
from fastapi import FastAPI

from config.settings import settings, Environment

logger = logging.getLogger(__name__)

PROMETHEUS_AVAILABLE = False


def configure_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    _configure_cors(app)
    _configure_rate_limiting(app)
    _configure_security_middleware(app)
    _configure_tracing_middleware(app)
    _configure_monitoring_middleware(app)


def _configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware.
    
    Args:
        app: FastAPI application instance
    """
    from fastapi.middleware.cors import CORSMiddleware
    
    allowed_origins = (
        settings.security.allowed_origins
        if settings.security.allowed_origins != ["*"]
        else ["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-*"],
        max_age=3600,
    )


def _configure_rate_limiting(app: FastAPI) -> None:
    """Configure rate limiting middleware"""
    middleware = _safe_import_middleware(
        module_path="middleware.rate_limit_middleware",
        class_name="RateLimitMiddleware",
        dependency_module="utils.rate_limiter",
        dependency_class="RateLimiter",
        warning_message="Rate limiting middleware not available"
    )
    
    if middleware:
        from utils.rate_limiter import RateLimiter
        
        rate_limiter = RateLimiter(
            max_requests=settings.rate_limit.max_requests,
            window_seconds=settings.rate_limit.window_seconds
        )
        app.add_middleware(middleware, rate_limiter=rate_limiter)


def _configure_security_middleware(app: FastAPI) -> None:
    """Configure security middleware"""
    if not settings.security.require_auth:
        return
    
    middleware = _safe_import_middleware(
        module_path="middleware.security_middleware",
        class_name="SecurityMiddleware",
        warning_message="Security middleware not available"
    )
    
    if middleware:
        app.add_middleware(middleware)


def _configure_tracing_middleware(app: FastAPI) -> None:
    """Configure tracing middleware"""
    if settings.environment != Environment.PRODUCTION:
        return
    
    middleware = _safe_import_middleware(
        module_path="middleware.tracing_middleware",
        class_name="TracingMiddleware",
        warning_message="Tracing middleware not available",
        log_level="debug"
    )
    
    if middleware:
        app.add_middleware(middleware)


def _configure_monitoring_middleware(app: FastAPI) -> None:
    """Configure monitoring middleware"""
    global PROMETHEUS_AVAILABLE
    
    try:
        from middleware.monitoring_middleware import MonitoringMiddleware, PROMETHEUS_AVAILABLE as prom_avail
        app.add_middleware(MonitoringMiddleware)
        PROMETHEUS_AVAILABLE = prom_avail
    except ImportError:
        logger.warning("Monitoring middleware not available")
        PROMETHEUS_AVAILABLE = False


def _safe_import_middleware(
    module_path: str,
    class_name: str,
    dependency_module: Optional[str] = None,
    dependency_class: Optional[str] = None,
    warning_message: str = "Middleware not available",
    log_level: str = "warning"
) -> Optional[Type]:
    """
    Safely import and return a middleware class.
    
    Args:
        module_path: Path to the module containing the middleware
        class_name: Name of the middleware class
        dependency_module: Optional module path for dependencies
        dependency_class: Optional class name for dependencies
        warning_message: Message to log if import fails
        log_level: Log level for the warning (warning or debug)
    
    Returns:
        Middleware class if import succeeds, None otherwise
    """
    try:
        if dependency_module:
            __import__(dependency_module)
        
        module = __import__(module_path, fromlist=[class_name])
        middleware_class = getattr(module, class_name)
        return middleware_class
    except ImportError as e:
        if log_level == "debug":
            logger.debug(f"{warning_message}: {e}")
        else:
            logger.warning(f"{warning_message}: {e}")
        return None

