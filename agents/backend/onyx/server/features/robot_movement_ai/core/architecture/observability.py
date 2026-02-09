"""
Observability
=============

Sistema de observabilidad: métricas, tracing y logging estructurado.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica."""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat()
        }


class MetricsCollector:
    """
    Colector de métricas.
    """
    
    def __init__(self):
        """Inicializar colector."""
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """
        Incrementar contador.
        
        Args:
            name: Nombre de la métrica
            value: Valor a incrementar
            tags: Tags (opcional, para futura implementación)
        """
        with self._lock:
            self._counters[name] += value
    
    def decrement(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """
        Decrementar contador.
        
        Args:
            name: Nombre de la métrica
            value: Valor a decrementar
            tags: Tags (opcional)
        """
        with self._lock:
            self._counters[name] -= value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Establecer gauge.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            tags: Tags (opcional)
        """
        with self._lock:
            self._gauges[name] = value
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Registrar valor en histograma.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            tags: Tags (opcional)
        """
        with self._lock:
            self._histograms[name].append(value)
            # Mantener solo últimos 1000 valores
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
    
    def get_counter(self, name: str) -> float:
        """Obtener valor de contador."""
        with self._lock:
            return self._counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Obtener valor de gauge."""
        with self._lock:
            return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        Obtener estadísticas de histograma.
        
        Args:
            name: Nombre de la métrica
            
        Returns:
            Estadísticas (min, max, mean, count) o None
        """
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return None
            
            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
                "p50": sorted(values)[len(values) // 2] if values else 0,
                "p95": sorted(values)[int(len(values) * 0.95)] if values else 0,
                "p99": sorted(values)[int(len(values) * 0.99)] if values else 0
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: self.get_histogram_stats(name)
                    for name in self._histograms.keys()
                }
            }
    
    def reset(self):
        """Resetear todas las métricas."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


@dataclass
class Span:
    """Span para tracing."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Duración del span."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs
        }


class Tracer:
    """
    Tracer para distributed tracing.
    """
    
    def __init__(self):
        """Inicializar tracer."""
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
    
    def start_span(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Iniciar span.
        
        Args:
            name: Nombre del span
            tags: Tags (opcional)
            
        Returns:
            ID del span
        """
        import uuid
        span_id = str(uuid.uuid4())
        
        span = Span(
            name=name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self._lock:
            self._active_spans[span_id] = span
        
        return span_id
    
    def end_span(self, span_id: str):
        """
        Finalizar span.
        
        Args:
            span_id: ID del span
        """
        with self._lock:
            span = self._active_spans.pop(span_id, None)
            if span:
                span.end_time = time.time()
                self._spans.append(span)
    
    def add_tag(self, span_id: str, key: str, value: str):
        """
        Agregar tag a span.
        
        Args:
            span_id: ID del span
            key: Clave del tag
            value: Valor del tag
        """
        with self._lock:
            span = self._active_spans.get(span_id)
            if span:
                span.tags[key] = value
    
    def add_log(self, span_id: str, log: Dict[str, Any]):
        """
        Agregar log a span.
        
        Args:
            span_id: ID del span
            log: Log
        """
        with self._lock:
            span = self._active_spans.get(span_id)
            if span:
                span.logs.append(log)
    
    def get_spans(self, limit: int = 100) -> List[Span]:
        """
        Obtener spans.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de spans
        """
        with self._lock:
            return self._spans[-limit:]
    
    def clear(self):
        """Limpiar spans."""
        with self._lock:
            self._spans.clear()
            self._active_spans.clear()


class ObservabilityManager:
    """
    Gestor de observabilidad.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        self._enabled = True
    
    def enable(self):
        """Habilitar observabilidad."""
        self._enabled = True
    
    def disable(self):
        """Deshabilitar observabilidad."""
        self._enabled = False
    
    def get_health(self) -> Dict[str, Any]:
        """
        Obtener estado de salud.
        
        Returns:
            Estado de salud
        """
        return {
            "status": "healthy" if self._enabled else "disabled",
            "metrics": {
                "total_requests": self.metrics.get_counter("requests.total"),
                "total_errors": self.metrics.get_counter("responses.error"),
                "total_success": self.metrics.get_counter("responses.success")
            },
            "tracing": {
                "active_spans": len(self.tracer._active_spans),
                "total_spans": len(self.tracer._spans)
            }
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Obtener datos para dashboard.
        
        Returns:
            Datos del dashboard
        """
        return {
            "metrics": self.metrics.get_all_metrics(),
            "recent_spans": [span.to_dict() for span in self.tracer.get_spans(limit=50)],
            "health": self.get_health()
        }

