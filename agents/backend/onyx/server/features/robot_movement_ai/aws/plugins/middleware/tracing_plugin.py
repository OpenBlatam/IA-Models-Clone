"""
OpenTelemetry Tracing Plugin
============================
"""

import os
import logging
from typing import Dict, Any
from fastapi import FastAPI
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class TracingMiddlewarePlugin(MiddlewarePlugin):
    """OpenTelemetry distributed tracing plugin."""
    
    def get_name(self) -> str:
        return "tracing"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_tracing", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup OpenTelemetry tracing."""
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            
            middleware_config = config.get("middleware", {})
            service_name = config.get("app_name", "robot-movement-ai")
            
            resource = Resource.create({
                "service.name": service_name,
                "service.version": config.get("app_version", "1.0.0"),
            })
            
            otlp_endpoint = middleware_config.get("otlp_endpoint", "http://localhost:4317")
            otlp_insecure = middleware_config.get("otlp_insecure", True)
            
            trace.set_tracer_provider(TracerProvider(resource=resource))
            tracer = trace.get_tracer(__name__)
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=otlp_insecure
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # Instrument FastAPI and HTTP clients
            FastAPIInstrumentor.instrument_app(app)
            HTTPXClientInstrumentor.instrument()
            
            logger.info(f"OpenTelemetry tracing enabled: {otlp_endpoint}")
            
        except ImportError:
            logger.warning("OpenTelemetry not installed. Tracing disabled.")
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry: {e}")
        
        return app















