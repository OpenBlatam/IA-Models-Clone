"""
Distributed Tracing Middleware
================================

Implements OpenTelemetry distributed tracing.
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class TracingMiddleware(BaseHTTPMiddleware):
    """Distributed tracing middleware using OpenTelemetry."""

    def __init__(
        self,
        app: ASGIApp,
        service_name: str = "manuales-hogar-ai",
        otlp_endpoint: Optional[str] = None,
    ):
        super().__init__(app)
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.tracer = None

        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            # Create tracer provider
            provider = TracerProvider()
            trace.set_tracer_provider(provider)

            # Add OTLP exporter if endpoint is provided
            if self.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to setup tracing: {e}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with distributed tracing."""
        if not self.tracer:
            return await call_next(request)

        # Start span
        with self.tracer.start_as_current_span(
            f"{request.method} {request.url.path}"
        ) as span:
            # Add attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")

            # Add request ID if available
            if hasattr(request.state, "request_id"):
                span.set_attribute("request.id", request.state.request_id)

            start_time = time.time()

            try:
                # Process request
                response = await call_next(request)

                # Calculate duration
                duration = time.time() - start_time

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.duration_ms", duration * 1000)

                if response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))

                return response

            except Exception as e:
                duration = time.time() - start_time
                span.set_attribute("http.duration_ms", duration * 1000)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise




