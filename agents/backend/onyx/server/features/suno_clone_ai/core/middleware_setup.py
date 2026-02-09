"""
Middleware setup module for centralized middleware configuration
"""

import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from middleware.logging_middleware import LoggingMiddleware
from middleware.rate_limiter import RateLimiterMiddleware
from middleware.error_handler_middleware import ErrorHandlerMiddleware
from middleware.auth_middleware import AuthMiddleware
from middleware.opentelemetry_middleware import OpenTelemetryMiddleware, instrument_fastapi
from middleware.security_headers_middleware import SecurityHeadersMiddleware
from middleware.service_mesh_middleware import ServiceMeshMiddleware
from middleware.load_balancer_middleware import LoadBalancerMiddleware
from middleware.performance_middleware import PerformanceMiddleware
from middleware.api_gateway_middleware import APIGatewayMiddleware
from middleware.compression_middleware import CompressionMiddleware
from middleware.response_cache_middleware import ResponseCacheMiddleware
from middleware.request_optimizer_middleware import RequestOptimizerMiddleware
from utils.prometheus_metrics import PrometheusMiddleware

logger = logging.getLogger(__name__)


def _get_service_name() -> str:
    """Get normalized service name for middleware"""
    return settings.app_name.lower().replace(" ", "-")


def _setup_observability_middleware(app: FastAPI) -> None:
    """Setup observability middleware (tracing, metrics)"""
    if settings.enable_tracing:
        try:
            app.add_middleware(
                OpenTelemetryMiddleware,
                service_name=_get_service_name(),
                enabled=settings.enable_tracing
            )
            instrument_fastapi(app, service_name=_get_service_name())
            logger.info("OpenTelemetry tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")

    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(LoggingMiddleware)


def _setup_infrastructure_middleware(app: FastAPI) -> None:
    """Setup infrastructure middleware (service mesh, API gateway, load balancer)"""
    app.add_middleware(
        ServiceMeshMiddleware,
        service_name=_get_service_name(),
        enable_istio=True,
        enable_consul=False,
        enable_linkerd=False
    )

    app.add_middleware(
        APIGatewayMiddleware,
        gateway_type=settings.api_gateway_type,
        enable_rate_limiting=True
    )

    app.add_middleware(LoadBalancerMiddleware)


def _setup_performance_middleware(app: FastAPI) -> None:
    """Setup performance middleware (caching, compression, optimization)"""
    app.add_middleware(
        RequestOptimizerMiddleware,
        enable_prefetch=True,
        enable_early_response=False
    )

    app.add_middleware(
        ResponseCacheMiddleware,
        ttl=settings.response_cache_ttl,
        max_size=1000
    )

    app.add_middleware(
        CompressionMiddleware,
        minimum_size=settings.compression_min_size,
        compress_level=settings.compression_level
    )

    app.add_middleware(
        PerformanceMiddleware,
        enable_caching=True,
        cache_ttl=settings.response_cache_ttl
    )


def _setup_security_middleware(app: FastAPI) -> None:
    """Setup security middleware (CORS, security headers, auth, rate limiting)"""
    app.add_middleware(SecurityHeadersMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.enable_auth:
        app.add_middleware(AuthMiddleware)
        logger.info("Authentication middleware enabled")

    if not settings.debug:
        app.add_middleware(RateLimiterMiddleware)
        logger.info("Rate limiting enabled (production mode)")


def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the application in the correct order.
    
    Middleware order matters in FastAPI - they execute in reverse order
    of registration (last registered = first executed).
    
    Order of execution (from first to last):
    1. Error handling (last registered, first executed)
    2. Security (auth, rate limiting, CORS)
    3. Observability (logging, metrics, tracing)
    4. Performance (caching, compression, optimization)
    5. Infrastructure (service mesh, API gateway, load balancer)
    """
    try:
        _setup_observability_middleware(app)
        _setup_infrastructure_middleware(app)
        _setup_performance_middleware(app)
        _setup_security_middleware(app)
        app.add_middleware(ErrorHandlerMiddleware)
        
        logger.info("All middleware configured successfully")
    except Exception as e:
        logger.error(f"Failed to setup middleware: {e}", exc_info=True)
        raise

