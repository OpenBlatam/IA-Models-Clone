"""
Middleware configuration for FastAPI application
"""

import logging
from typing import Any
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI, config: Any) -> None:
    """
    Setup all middleware for the application
    
    Middleware order matters - last added is first executed
    """
    from middleware.error_handler import ErrorHandlerMiddleware
    app.add_middleware(ErrorHandlerMiddleware)
    
    setup_performance_middleware(app, config)
    setup_aws_middleware(app, config)
    setup_rate_limiting_middleware(app, config)
    
    from middleware.performance import PerformanceMonitoringMiddleware
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    from middleware.logging_middleware import LoggingMiddleware
    app.add_middleware(LoggingMiddleware)


def setup_performance_middleware(app: FastAPI, config: Any) -> None:
    """Setup performance optimization middleware"""
    try:
        from performance.async_optimizer import get_async_optimizer
        from performance.memory_optimizer import get_memory_optimizer
        from middleware.speed_middleware import SpeedMiddleware
        from middleware.ultra_speed_middleware import UltraSpeedMiddleware
        
        async_optimizer = get_async_optimizer()
        async_optimizer.enable_uvloop()
        
        memory_optimizer = get_memory_optimizer()
        
        try:
            from middleware.performance_integrator import PerformanceIntegratorMiddleware
            redis_url = get_redis_url(config)
            
            app.add_middleware(
                PerformanceIntegratorMiddleware,
                enable_all=True,
                redis_url=redis_url
            )
            logger.info("✅ Performance integrator middleware enabled")
        except ImportError:
            pass
        
        redis_url = get_redis_url(config)
        
        try:
            app.add_middleware(
                UltraSpeedMiddleware,
                redis_url=redis_url,
                enable_brotli=True,
                enable_coalescing=True,
                enable_prefetch=True
            )
            app.add_middleware(SpeedMiddleware)
        except ImportError:
            pass
    except ImportError:
        logger.debug("Performance middleware not available")


def setup_aws_middleware(app: FastAPI, config: Any) -> None:
    """Setup AWS-specific middleware"""
    try:
        from config.aws_settings import get_aws_settings
        from middleware.aws_observability import AWSObservabilityMiddleware
        from middleware.opentelemetry_middleware import OpenTelemetryMiddleware
        from middleware.oauth2_middleware import OAuth2Middleware
        from middleware.performance_middleware import PerformanceMiddleware, ConnectionPoolMiddleware
        from middleware.security_advanced import (
            SecurityHeadersMiddleware,
            DDoSProtectionMiddleware,
            InputValidationMiddleware
        )
        from aws.prometheus_metrics import PrometheusMetricsMiddleware, get_metrics_endpoint
        from optimization.cold_start import init_cold_start
        
        aws_settings = get_aws_settings()
        
        if aws_settings.is_lambda:
            init_cold_start()
        
        app.add_middleware(SecurityHeadersMiddleware, strict_csp=True)
        app.add_middleware(InputValidationMiddleware)
        app.add_middleware(DDoSProtectionMiddleware, requests_per_minute=60, requests_per_hour=1000)
        
        if aws_settings.is_lambda:
            app.add_middleware(OpenTelemetryMiddleware)
            app.add_middleware(AWSObservabilityMiddleware)
            app.add_middleware(PrometheusMetricsMiddleware)
            app.add_middleware(PerformanceMiddleware, enable_compression=True)
            app.add_middleware(ConnectionPoolMiddleware)
            
            app.add_middleware(OAuth2Middleware, public_paths=[
                "/", "/docs", "/redoc", "/openapi.json",
                "/health", "/metrics",
                "/recovery/health",
                "/recovery/auth/login",
                "/recovery/auth/register"
            ])
            
            from fastapi import APIRouter
            metrics_router = APIRouter()
            metrics_router.add_api_route("/metrics", get_metrics_endpoint(), methods=["GET"])
            app.include_router(metrics_router)
        else:
            app.add_middleware(PerformanceMiddleware, enable_compression=True)
            app.add_middleware(ConnectionPoolMiddleware)
    except ImportError:
        logger.debug("AWS middleware not available")


def setup_rate_limiting_middleware(app: FastAPI, config: Any) -> None:
    """Setup rate limiting middleware"""
    try:
        from middleware.throttling_middleware import ThrottlingMiddleware
        app.add_middleware(ThrottlingMiddleware)
    except ImportError:
        try:
            from middleware.rate_limit import RateLimitMiddleware
            app.add_middleware(
                RateLimitMiddleware,
                requests_per_minute=config.rate_limit_per_minute,
                requests_per_hour=config.rate_limit_per_hour
            )
        except ImportError:
            logger.debug("Rate limiting middleware not available")


def get_redis_url(config: Any) -> Any:
    """Get Redis URL from config if available"""
    try:
        return config.redis_url if hasattr(config, 'redis_url') else None
    except Exception:
        return None

