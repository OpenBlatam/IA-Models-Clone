"""
Tracing Service - Servicio de trazabilidad
"""
from .tracer import Tracer
from .logger import StructuredLogger
from .metrics import MetricsCollector
from .span_manager import SpanManager


class TracingService:
    """Servicio centralizado de tracing y observabilidad"""
    
    def __init__(self):
        self.tracer = Tracer()
        self.logger = StructuredLogger()
        self.metrics = MetricsCollector()
        self.span_manager = SpanManager()
    
    async def trace_operation(self, operation_name: str, **kwargs):
        """Traza una operación"""
        async with self.tracer.trace(operation_name, **kwargs):
            pass
    
    def log(self, level: str, message: str, **kwargs):
        """Registra un log"""
        self.logger.log(level, message, **kwargs)
    
    def metric(self, name: str, value: float, tags: dict = None):
        """Registra una métrica"""
        self.metrics.record(name, value, tags or {})

