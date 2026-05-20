"""
OpenTelemetry Middleware for Distributed Tracing
Advanced observability with OpenTelemetry (works with AWS X-Ray, Jaeger, etc.)
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.boto3 import Boto3Instrumentor
    from opentelemetry.propagators.aws import AwsXRayPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()


class OpenTelemetryMiddleware(BaseHTTPMiddleware):
    """
    OpenTelemetry middleware for distributed tracing
    
    Supports:
    - AWS X-Ray
    - Jaeger
    - Zipkin
    - OTLP (OpenTelemetry Protocol)
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
            return
        
        # Initialize OpenTelemetry
        self._setup_opentelemetry()
    
    def _setup_opentelemetry(self) -> None:
        """Setup OpenTelemetry tracing"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": "addiction-recovery-ai",
                "service.version": "3.3.0",
                "deployment.environment": aws_settings.environment,
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add OTLP exporter (for AWS X-Ray, Jaeger, etc.)
            otlp_endpoint = aws_settings.aws_region  # Can be configured via env
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    insecure=True  # Use TLS in production
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            # Use AWS X-Ray propagator if in AWS
            if aws_settings.is_lambda:
                from opentelemetry.propagators import composite
                from opentelemetry.propagators.b3 import B3Format
                trace.set_textmap_propagator(
                    composite.CompositeHTTPPropagator([
                        AwsXRayPropagator(),
                        B3Format()
                    ])
                )
            
            # Instrument FastAPI
            FastAPIInstrumentor.instrument()
            
            # Instrument HTTP clients
            HTTPXClientInstrumentor().instrument()
            
            # Instrument AWS SDK
            Boto3Instrumentor().instrument()
            
            logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {str(e)}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with OpenTelemetry tracing"""
        if not OPENTELEMETRY_AVAILABLE:
            return await call_next(request)
        
        tracer = trace.get_tracer(__name__)
        
        with tracer.start_as_current_span(
            name=f"{request.method} {request.url.path}",
            kind=trace.SpanKind.SERVER
        ) as span:
            # Add attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.scheme", request.url.scheme)
            
            # Add headers as attributes
            if "user-agent" in request.headers:
                span.set_attribute("http.user_agent", request.headers["user-agent"])
            if "x-forwarded-for" in request.headers:
                span.set_attribute("http.client_ip", request.headers["x-forwarded-for"])
            
            start_time = time.time()
            
            try:
                # Process request
                response = await call_next(request)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_time", duration)
                
                # Add status
                if response.status_code >= 500:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Server error"))
                elif response.status_code >= 400:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, "Client error"))
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))
                
                return response
                
            except Exception as e:
                # Record exception
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise















