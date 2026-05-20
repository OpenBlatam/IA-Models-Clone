"""
Observability Utilities
=======================
Utilidades para observabilidad y tracing.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime
import uuid
from functools import wraps
import time

from .logger import get_logger

logger = get_logger(__name__)


class TraceContext:
    """Contexto de tracing para requests."""
    
    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.start_time = datetime.now()
        self.tags: Dict[str, Any] = {}
        self.baggage: Dict[str, Any] = {}
    
    def add_tag(self, key: str, value: Any):
        """Agregar tag al trace."""
        self.tags[key] = value
    
    def add_baggage(self, key: str, value: Any):
        """Agregar baggage al trace."""
        self.baggage[key] = value
    
    def get_duration(self) -> float:
        """Obtener duración del trace."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "start_time": self.start_time.isoformat(),
            "duration": self.get_duration(),
            "tags": self.tags,
            "baggage": self.baggage
        }


class Span:
    """Span para distributed tracing."""
    
    def __init__(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.tags: Dict[str, Any] = {}
        self.logs: list = []
        self.status = "started"
    
    def finish(self, status: str = "completed"):
        """Finalizar span."""
        self.end_time = datetime.now()
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Agregar tag."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Agregar log al span."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
            **kwargs
        })
    
    def get_duration(self) -> Optional[float]:
        """Obtener duración."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_duration(),
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs
        }


def trace_function(operation_name: Optional[str] = None):
    """
    Decorador para tracing de funciones.
    
    Args:
        operation_name: Nombre de la operación
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            trace_id = str(uuid.uuid4())
            span = Span(name=name, trace_id=trace_id)
            
            try:
                span.add_tag("function", func.__name__)
                span.add_tag("args_count", len(args))
                
                start = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                span.add_tag("duration", duration)
                span.add_tag("success", True)
                span.finish("completed")
                
                logger.debug("span_completed", **span.to_dict())
                
                return result
            except Exception as e:
                span.add_tag("success", False)
                span.add_tag("error", str(e))
                span.add_log(f"Error: {str(e)}", level="error")
                span.finish("error")
                
                logger.error("span_error", **span.to_dict())
                
                raise
        
        return wrapper
    return decorator


class MetricsCollector:
    """Colector de métricas personalizadas."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        })
    
    def get_metric(self, name: str) -> list:
        """Obtener métrica."""
        return self.metrics.get(name, [])
    
    def get_summary(self, name: str) -> Dict[str, Any]:
        """Obtener resumen de métrica."""
        values = [m["value"] for m in self.metrics.get(name, [])]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }


# Instancia global
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtener instancia global del colector de métricas."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

