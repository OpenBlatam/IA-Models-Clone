"""
Advanced Observability Middleware
Integrates OpenTelemetry, structured logging, and metrics
"""

import time
import logging
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    from observability import http_requests_total, http_request_duration
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ...shared.response import create_error_response

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive observability middleware
    - Distributed tracing (OpenTelemetry)
    - Structured logging
    - Prometheus metrics
    - Request correlation
    """
    
    def __init__(self, app, service_name: str = "content-redundancy-detector"):
        super().__init__(app)
        self.service_name = service_name
        self.tracer = None
        
        if OPENTELEMETRY_AVAILABLE:
            try:
                self.tracer = trace.get_tracer(service_name)
            except Exception as e:
                logger.warning(f"OpenTelemetry tracer not available: {e}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start trace span
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                f"{request.method} {request.url.path}",
                kind=trace.SpanKind.SERVER
            )
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.scheme", request.url.scheme)
            span.set_attribute("correlation.id", correlation_id)
        
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Structured logging
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update span
            if span:
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.request.duration", duration)
                span.set_status(Status(StatusCode.OK))
            
            # Record Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                try:
                    http_requests_total.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code
                    ).inc()
                    
                    http_request_duration.labels(
                        method=request.method,
                        endpoint=request.url.path
                    ).observe(duration)
                except Exception as e:
                    logger.warning(f"Metrics recording error: {e}")
            
            # Add correlation ID to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Response-Time"] = f"{duration:.4f}"
            
            # Structured logging
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Update span with error
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            # Record error metric
            if PROMETHEUS_AVAILABLE:
                try:
                    http_requests_total.labels(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=500
                    ).inc()
                except Exception:
                    pass
            
            # Structured error logging
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration": duration
                },
                exc_info=True
            )
            
            raise
        finally:
            if span:
                span.end()






