"""
Observability Utilities - Utilidades de observabilidad
======================================================

Utilidades para métricas, trazas, y monitoreo avanzado.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict, deque
from threading import Lock

logger = logging.getLogger(__name__)


class Metric:
    """Métrica individual."""
    
    def __init__(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Inicializar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            tags: Tags adicionales (opcional)
        """
        self.name = name
        self.value = value
        self.tags = tags or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsCollector:
    """
    Colector de métricas.
    
    Recolecta y almacena métricas con soporte para contadores,
    gauges, e histogramas.
    """
    
    def __init__(self, max_metrics: int = 1000):
        """
        Inicializar colector.
        
        Args:
            max_metrics: Número máximo de métricas a almacenar
        """
        self.max_metrics = max_metrics
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._metrics: deque = deque(maxlen=max_metrics)
        self._lock = Lock()
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Incrementar contador.
        
        Args:
            name: Nombre del contador
            value: Valor a incrementar
            tags: Tags adicionales (opcional)
        """
        with self._lock:
            self._counters[name] += value
            self._metrics.append(Metric(f"{name}.counter", self._counters[name], tags))
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Establecer gauge.
        
        Args:
            name: Nombre del gauge
            value: Valor del gauge
            tags: Tags adicionales (opcional)
        """
        with self._lock:
            self._gauges[name] = value
            self._metrics.append(Metric(f"{name}.gauge", value, tags))
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Registrar valor en histograma.
        
        Args:
            name: Nombre del histograma
            value: Valor a registrar
            tags: Tags adicionales (opcional)
        """
        with self._lock:
            self._histograms[name].append(value)
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
            self._metrics.append(Metric(f"{name}.histogram", value, tags))
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las métricas.
        
        Returns:
            Lista de métricas como diccionarios
        """
        with self._lock:
            return [metric.to_dict() for metric in self._metrics]
    
    def get_counters(self) -> Dict[str, float]:
        """Obtener todos los contadores."""
        with self._lock:
            return self._counters.copy()
    
    def get_gauges(self) -> Dict[str, float]:
        """Obtener todos los gauges."""
        with self._lock:
            return self._gauges.copy()
    
    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Obtener estadísticas de histograma.
        
        Args:
            name: Nombre del histograma
        
        Returns:
            Diccionario con estadísticas o None si no existe
        """
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return None
            
            values = self._histograms[name]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)],
            }
    
    def clear(self) -> None:
        """Limpiar todas las métricas."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._metrics.clear()


class TraceSpan:
    """Span de traza."""
    
    def __init__(self, name: str, parent: Optional['TraceSpan'] = None):
        """
        Inicializar span.
        
        Args:
            name: Nombre del span
            parent: Span padre (opcional)
        """
        self.name = name
        self.parent = parent
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.tags: Dict[str, Any] = {}
        self.logs: List[Dict[str, Any]] = []
        self.children: List['TraceSpan'] = []
    
    def finish(self) -> None:
        """Finalizar span."""
        self.end_time = time.time()
    
    def add_tag(self, key: str, value: Any) -> None:
        """Agregar tag."""
        self.tags[key] = value
    
    def add_log(self, message: str, **kwargs: Any) -> None:
        """Agregar log al span."""
        self.logs.append({
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
    
    def duration(self) -> float:
        """Obtener duración en segundos."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration(),
            "tags": self.tags,
            "logs": self.logs,
            "children": [child.to_dict() for child in self.children],
        }


class Tracer:
    """
    Tracer para trazas distribuidas.
    
    Permite crear y gestionar trazas con spans anidados.
    """
    
    def __init__(self):
        """Inicializar tracer."""
        self._spans: List[TraceSpan] = []
        self._current_span: Optional[TraceSpan] = None
    
    @contextmanager
    def span(self, name: str, **tags: Any):
        """
        Crear span como context manager.
        
        Args:
            name: Nombre del span
            **tags: Tags iniciales
        
        Example:
            with tracer.span("operation", component="api"):
                # código
        """
        parent = self._current_span
        span = TraceSpan(name, parent=parent)
        
        for key, value in tags.items():
            span.add_tag(key, value)
        
        if parent:
            parent.children.append(span)
        else:
            self._spans.append(span)
        
        self._current_span = span
        
        try:
            yield span
        finally:
            span.finish()
            if self._current_span == span:
                self._current_span = parent
    
    def get_spans(self) -> List[Dict[str, Any]]:
        """
        Obtener todas las trazas.
        
        Returns:
            Lista de spans como diccionarios
        """
        return [span.to_dict() for span in self._spans]
    
    def clear(self) -> None:
        """Limpiar todas las trazas."""
        self._spans.clear()
        self._current_span = None


# Instancias globales
_metrics_collector = MetricsCollector()
_tracer = Tracer()


def get_metrics_collector() -> MetricsCollector:
    """Obtener instancia global de MetricsCollector."""
    return _metrics_collector


def get_tracer() -> Tracer:
    """Obtener instancia global de Tracer."""
    return _tracer


@contextmanager
def measure_time(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Medir tiempo de operación y registrar como métrica.
    
    Args:
        operation_name: Nombre de la operación
        tags: Tags adicionales (opcional)
    
    Example:
        with measure_time("api_call", tags={"endpoint": "/users"}):
            # código
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        collector = get_metrics_collector()
        collector.record_histogram(f"{operation_name}.duration", duration, tags)
        collector.increment(f"{operation_name}.count", tags=tags)


__all__ = [
    "Metric",
    "MetricsCollector",
    "TraceSpan",
    "Tracer",
    "get_metrics_collector",
    "get_tracer",
    "measure_time",
]

