"""
Middleware for Multi-Model API
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.asyncio import AsyncioIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        'multi_model_requests_total',
        'Total number of requests',
        ['method', 'endpoint', 'status']
    )
    REQUEST_DURATION = Histogram(
        'multi_model_request_duration_seconds',
        'Request duration in seconds',
        ['method', 'endpoint']
    )
    ACTIVE_REQUESTS = Gauge(
        'multi_model_active_requests',
        'Number of active requests'
    )


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)
        
        method = request.method
        endpoint = request.url.path
        
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
            
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                time.time() - start_time
            )
            
            return response
        except Exception as e:
            status = "500"
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2)
                }
            )
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(duration, 3))
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2)
                },
                exc_info=True
            )
            raise


def init_sentry(dsn: str = None, environment: str = "production"):
    """Initialize Sentry for error tracking"""
    if not SENTRY_AVAILABLE:
        logger.warning("Sentry SDK not available")
        return
    
    if not dsn:
        dsn = None
        logger.info("Sentry DSN not provided, skipping initialization")
        return
    
    try:
        sentry_sdk.init(
            dsn=dsn,
            integrations=[
                FastApiIntegration(),
                AsyncioIntegration(),
            ],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            environment=environment,
            before_send=_sentry_before_send
        )
        logger.info("Sentry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")


def _sentry_before_send(event, hint):
    """Filter events before sending to Sentry"""
    if not SENTRY_AVAILABLE:
        return event
    
    if 'rate_limit' in str(event).lower():
        return None
    
    if 'cache_miss' in str(event).lower():
        return None
    
    event.setdefault('tags', {})['component'] = 'multi_model_api'
    event.setdefault('tags', {})['version'] = '2.1.0'
    
    return event

