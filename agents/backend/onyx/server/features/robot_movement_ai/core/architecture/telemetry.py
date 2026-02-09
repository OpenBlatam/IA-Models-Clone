"""
Sistema de telemetría y tracing para Robot Movement AI v2.0
Distributed tracing con OpenTelemetry
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime
import time

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None


class TelemetryManager:
    """Gestor de telemetría y tracing"""
    
    def __init__(
        self,
        service_name: str = "robot-movement-ai",
        endpoint: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Inicializar gestor de telemetría
        
        Args:
            service_name: Nombre del servicio
            endpoint: Endpoint de OTLP (opcional)
            enabled: Habilitar telemetría
        """
        self.service_name = service_name
        self.endpoint = endpoint
        self.enabled = enabled
        self.tracer: Optional[Any] = None
        
        if enabled and OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Configurar tracing con OpenTelemetry"""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "2.0.0"
        })
        
        provider = TracerProvider(resource=resource)
        
        if self.endpoint:
            exporter = OTLPSpanExporter(endpoint=self.endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(self.service_name)
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager para crear span
        
        Args:
            name: Nombre del span
            attributes: Atributos adicionales
        """
        if not self.enabled or not self.tracer:
            yield
            return
        
        span = self.tracer.start_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        finally:
            span.end()
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Agregar evento al span actual"""
        if not self.enabled or not self.tracer:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes or {})


# Instancia global
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Obtener instancia global del gestor de telemetría"""
    global _telemetry_manager
    if _telemetry_manager is None:
        from core.architecture.config import get_config
        config = get_config()
        _telemetry_manager = TelemetryManager(
            service_name="robot-movement-ai",
            enabled=getattr(config, 'enable_tracing', False)
        )
    return _telemetry_manager


def trace_function(name: Optional[str] = None):
    """
    Decorator para trazar funciones
    
    Args:
        name: Nombre del span (opcional)
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_telemetry_manager()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with manager.span(span_name):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_telemetry_manager()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with manager.span(span_name):
                return func(*args, **kwargs)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




