"""
Metrics Utilities
=================

Utilidades para métricas avanzadas.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


class MetricsCollector:
    """Colector de métricas."""
    
    def __init__(self):
        """Inicializar colector."""
        self.metrics: List[Metric] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._logger = logger
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Incrementar contador.
        
        Args:
            name: Nombre del contador
            value: Valor a incrementar
            tags: Tags adicionales
        """
        key = self._make_key(name, tags)
        self.counters[key] += value
        
        self.record_metric(
            name=f"{name}_total",
            value=self.counters[key],
            tags=tags or {}
        )
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Establecer gauge.
        
        Args:
            name: Nombre del gauge
            value: Valor
            tags: Tags adicionales
        """
        key = self._make_key(name, tags)
        self.gauges[key] = value
        
        self.record_metric(
            name=name,
            value=value,
            tags=tags or {}
        )
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Registrar valor en histograma.
        
        Args:
            name: Nombre del histograma
            value: Valor
            tags: Tags adicionales
        """
        key = self._make_key(name, tags)
        self.histograms[key].append(value)
        
        # Mantener solo últimos 1000 valores
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self.record_metric(
            name=name,
            value=value,
            tags=tags or {}
        )
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """
        Registrar métrica.
        
        Args:
            name: Nombre
            value: Valor
            tags: Tags
            unit: Unidad
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.metrics.append(metric)
        
        # Mantener solo últimos 10000 métricas
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Obtener estadísticas de histograma.
        
        Args:
            name: Nombre
            tags: Tags
        
        Returns:
            Estadísticas (min, max, avg, p50, p95, p99)
        """
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.50)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Obtener resumen de métricas.
        
        Args:
            hours: Horas hacia atrás
        
        Returns:
            Resumen
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff]
        
        # Agrupar por nombre
        by_name = defaultdict(list)
        for metric in recent_metrics:
            by_name[metric.name].append(metric.value)
        
        summary = {}
        for name, values in by_name.items():
            summary[name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
        
        return {
            "period_hours": hours,
            "total_metrics": len(recent_metrics),
            "metrics": summary,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Crear clave única."""
        if tags:
            tag_str = "_".join(f"{k}:{v}" for k, v in sorted(tags.items()))
            return f"{name}_{tag_str}"
        return name
    
    def reset(self):
        """Resetear todas las métricas."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()




