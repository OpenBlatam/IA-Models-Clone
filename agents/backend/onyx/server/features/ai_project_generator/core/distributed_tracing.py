"""
Distributed Tracing - Tracing Distribuido con OpenTelemetry
==========================================================

Tracing distribuido completo con OpenTelemetry:
- Span creation y management
- Context propagation
- Trace export
- Integration con FastAPI
"""

import logging
from typing import Optional, Dict, Any, Callable, ContextManager
from functools import wraps
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TracingBackend(str, Enum):
    """Backends de tracing"""
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    CLOUDWATCH = "cloudwatch"


class DistributedTracer:
    """
    Tracer distribuido con OpenTelemetry.
    """
    
    def __init__(
        self,
        service_name: str,
        backend: TracingBackend = TracingBackend.OTLP,
        **kwargs: Any
    ) -> None:
        self.service_name = service_name
        self.backend = backend
        self._tracer: Optional[Any] = None
        self._setup_tracing(**kwargs)
    
    def _setup_tracing(self, **kwargs: Any) -> None:
        """Configura OpenTelemetry"""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource
            
            # Crear resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": kwargs.get("version", "1.0.0")
            })
            
            # Configurar provider
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            
            # Configurar exporter según backend
            if self.backend == TracingBackend.JAEGER:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                exporter = JaegerExporter(
                    agent_host_name=kwargs.get("jaeger_host", "localhost"),
                    agent_port=kwargs.get("jaeger_port", 6831)
                )
            elif self.backend == TracingBackend.ZIPKIN:
                from opentelemetry.exporter.zipkin.json import ZipkinExporter
                exporter = ZipkinExporter(
                    endpoint=kwargs.get("zipkin_endpoint", "http://localhost:9411/api/v2/spans")
                )
            elif self.backend == TracingBackend.OTLP:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(
                    endpoint=kwargs.get("otlp_endpoint", "http://localhost:4317")
                )
            else:
                # Default: Console exporter
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                exporter = ConsoleSpanExporter()
            
            # Agregar processor
            provider.add_span_processor(BatchSpanProcessor(exporter))
            
            # Obtener tracer
            self._tracer = trace.get_tracer(self.service_name)
            
            logger.info(f"Distributed tracing configured for {self.service_name}")
        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}")
            self._tracer = None
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> ContextManager[Any]:
        """Inicia un span"""
        if not self._tracer:
            from contextlib import nullcontext
            return nullcontext()
        
        span = self._tracer.start_as_current_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        return span
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Agrega evento al span actual"""
        if not self._tracer:
            return
        
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Establece atributo en span actual"""
        if not self._tracer:
            return
        
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, str(value))
    
    def trace_function(
        self,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Decorator para trazar funciones"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.start_span(span_name, attributes):
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_name = name or f"{func.__module__}.{func.__name__}"
                with self.start_span(span_name, attributes):
                    return func(*args, **kwargs)
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


def get_distributed_tracer(
    service_name: str,
    backend: TracingBackend = TracingBackend.OTLP,
    **kwargs: Any
) -> DistributedTracer:
    """Obtiene tracer distribuido"""
    return DistributedTracer(service_name, backend, **kwargs)

