"""
OpenTelemetry Tracing Middleware for distributed tracing
"""

import os
import time
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi")


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for distributed tracing using OpenTelemetry.
    Falls back to simple logging if OpenTelemetry is not available.
    """
    
    def __init__(self, app, service_name: str = "dermatology-ai"):
        super().__init__(app)
        self.service_name = service_name
        self.tracer = None
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        try:
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "6.0.0",
            })
            
            provider = TracerProvider(resource=resource)
            
            # Try to use OTLP exporter (for Jaeger, Tempo, etc.)
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                    insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except Exception as e:
                logger.warning(f"OTLP exporter not available, using console: {e}")
            
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            
        except Exception as e:
            logger.warning(f"Tracing setup failed: {e}. Using fallback logging.")
            self.tracer = None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing"""
        start_time = time.time()
        
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or f"req_{int(time.time() * 1000)}"
        request.state.request_id = request_id
        
        # Create span if tracing is available
        if self.tracer:
            with self.tracer.start_as_current_span(
                f"{request.method} {request.url.path}",
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.route": request.url.path,
                    "request.id": request_id,
                }
            ) as span:
                try:
                    response = await call_next(request)
                    duration = time.time() - start_time
                    
                    span.set_attribute("http.status_code", response.status_code)
                    span.set_attribute("http.duration_ms", duration * 1000)
                    
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Response-Time"] = f"{duration:.3f}s"
                    
                    return response
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        else:
            # Fallback: simple logging
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                logger.info(
                    f"Request: {request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Duration: {duration:.3f}s - "
                    f"Request-ID: {request_id}"
                )
                
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                
                return response
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Request failed: {request.method} {request.url.path} - "
                    f"Error: {e} - Duration: {duration:.3f}s - "
                    f"Request-ID: {request_id}",
                    exc_info=True
                )
                raise

