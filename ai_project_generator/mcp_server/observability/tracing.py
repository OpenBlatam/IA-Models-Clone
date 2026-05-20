"""
MCP Tracing - Tracing OpenTelemetry para MCP
=============================================
"""

import logging
from typing import Dict, Any, Optional, ContextManager
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


class MCPTracer:
    """
    Tracer OpenTelemetry para MCP
    
    Crea spans para operaciones MCP y los exporta a OTLP o consola.
    """
    
    def __init__(self, otlp_endpoint: Optional[str] = None):
        """
        Inicializa el tracer
        
        Args:
            otlp_endpoint: Endpoint OTLP (opcional, si no se usa consola)
        """
        # Crear provider
        resource = Resource.create({"service.name": "mcp-server"})
        provider = TracerProvider(resource=resource)
        
        # Configurar exporter
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                logger.info(f"OTLP tracing enabled: {otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available, using console")
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
        else:
            # Usar consola por defecto
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            logger.info("Console tracing enabled")
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
    
    @contextmanager
    def start_span(self, name: str, **attributes) -> ContextManager:
        """
        Inicia un span
        
        Args:
            name: Nombre del span
            **attributes: Atributos del span
            
        Yields:
            Span
        """
        with self.tracer.start_as_current_span(name) as span:
            # Agregar atributos
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

