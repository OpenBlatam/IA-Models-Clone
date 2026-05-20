"""
Advanced Metrics
================

Métricas avanzadas con agregación y exportación.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Métricas avanzadas."""
    
    def __init__(self, history_size: int = 1000):
        """
        Inicializar métricas.
        
        Args:
            history_size: Tamaño del historial
        """
        self.history_size = history_size
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._logger = logger
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Incrementar contador.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor a incrementar
            tags: Tags adicionales
        """
        key = self._build_key(metric_name, tags)
        self.counters[key] += value
    
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Establecer gauge.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        """
        key = self._build_key(metric_name, tags)
        self.gauges[key] = value
    
    def record_histogram(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Registrar en histograma.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        """
        key = self._build_key(metric_name, tags)
        self.histograms[key].append(value)
    
    def time_operation(
        self,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager para medir tiempo.
        
        Args:
            metric_name: Nombre de la métrica
            tags: Tags adicionales
        
        Returns:
            Timer context manager
        """
        return Timer(self, metric_name, tags)
    
    def _build_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Construir clave con tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "p95": self._percentile(values, 95) if values else None,
                    "p99": self._percentile(values, 99) if values else None
                }
                for name, values in self.histograms.items()
            },
            "timers": {
                name: {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times) if times else None,
                    "min": min(times) if times else None,
                    "max": max(times) if times else None
                }
                for name, times in self.timers.items()
            }
        }
    
    def _percentile(self, values: deque, percentile: int) -> Optional[float]:
        """Calcular percentil."""
        if not values:
            return None
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class Timer:
    """Context manager para medir tiempo."""
    
    def __init__(self, metrics: AdvancedMetrics, name: str, tags: Optional[Dict[str, str]]):
        """Inicializar timer."""
        self.metrics = metrics
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        """Iniciar timer."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalizar timer."""
        duration = time.time() - self.start_time
        key = self.metrics._build_key(self.name, self.tags)
        self.metrics.timers[key].append(duration)
        
        # También registrar en histograma
        self.metrics.record_histogram(self.name, duration, self.tags)




