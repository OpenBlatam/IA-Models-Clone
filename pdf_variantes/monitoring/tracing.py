"""
PDF Variantes - Distributed Tracing with OpenTelemetry
"""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry (optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")


class TracingService:
    """Distributed tracing service"""
    
    def __init__(self, service_name: str = "pdf-variantes-api", enabled: bool = True):
        self.service_name = service_name
        self.enabled = enabled and OPENTELEMETRY_AVAILABLE
        
        if self.enabled:
            self._setup_tracer()
        else:
            self.tracer = None
            logger.info("Tracing disabled or OpenTelemetry not available")
    
    def _setup_tracer(self):
        """Setup OpenTelemetry tracer"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "2.0.0"
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            
            # Add exporter based on environment
            otlp_endpoint = os.getenv("OTLP_ENDPOINT")
            if otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"Tracing enabled with OTLP exporter: {otlp_endpoint}")
            else:
                # Console exporter for development
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))
                logger.info("Tracing enabled with console exporter")
            
            self.tracer = trace.get_tracer(self.service_name)
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
            self.enabled = False
            self.tracer = None
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI app"""
        if self.enabled and OPENTELEMETRY_AVAILABLE:
            try:
                FastAPIInstrumentor.instrument_app(app)
                logger.info("FastAPI instrumentation enabled")
            except Exception as e:
                logger.error(f"Failed to instrument FastAPI: {e}")
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a span context manager"""
        if self.enabled and self.tracer:
            span = self.tracer.start_as_current_span(name)
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                span.end()
        else:
            yield None
    
    def get_trace_context(self) -> Optional[Dict[str, str]]:
        """Get current trace context for propagation"""
        if self.enabled and self.tracer:
            span = trace.get_current_span()
            if span:
                ctx = span.get_span_context()
                return {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "trace_flags": format(ctx.trace_flags, "02x")
                }
        return None


# Global tracing service
_tracing_service: Optional[TracingService] = None


def get_tracing_service() -> TracingService:
    """Get global tracing service"""
    global _tracing_service
    if _tracing_service is None:
        enabled = os.getenv("ENABLE_TRACING", "false").lower() == "true"
        _tracing_service = TracingService(enabled=enabled)
    return _tracing_service


def setup_tracing(service_name: str = "pdf-variantes-api", enabled: bool = True):
    """Setup tracing globally"""
    global _tracing_service
    _tracing_service = TracingService(service_name=service_name, enabled=enabled)
    return _tracing_service






