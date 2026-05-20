"""
Metrics - Sistema de métricas y estadísticas
============================================

Recopila y gestiona métricas del agente.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica individual"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Recopilador de métricas"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.start_time = datetime.now()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registrar una métrica"""
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(metric)
        logger.debug(f"📊 Metric recorded: {name}={value}")
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Incrementar contador"""
        self.counters[name] += value
        self.record(f"{name}_count", self.counters[name], tags)
    
    def decrement(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Decrementar contador"""
        self.counters[name] = max(0, self.counters[name] - value)
        self.record(f"{name}_count", self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Establecer gauge"""
        self.gauges[name] = value
        self.record(f"{name}_gauge", value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registrar valor en histograma"""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        self.record(f"{name}_histogram", value, tags)
    
    def get_metric(self, name: str, window_seconds: Optional[int] = None) -> List[Metric]:
        """Obtener métricas por nombre"""
        if name not in self.metrics:
            return []
        
        metrics = list(self.metrics[name])
        
        if window_seconds:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        return metrics
    
    def get_counter(self, name: str) -> int:
        """Obtener valor de contador"""
        return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Obtener valor de gauge"""
        return self.gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Obtener estadísticas de histograma"""
        if name not in self.histograms or not self.histograms[name]:
            return {}
        
        values = self.histograms[name]
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / n,
            "median": sorted_values[n // 2] if n > 0 else 0,
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de todas las métricas.
        
        Returns:
            Diccionario con resumen completo de métricas
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calcular tasas por segundo
        rates = {}
        for name, counter_value in self.counters.items():
            if uptime > 0:
                rates[f"{name}_per_second"] = round(counter_value / uptime, 4)
        
        return {
            "uptime_seconds": round(uptime, 2),
            "uptime_human": str(timedelta(seconds=int(uptime))),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "rates": rates,
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self.histograms.keys()
            },
            "metric_names": list(self.metrics.keys()),
            "total_metrics_recorded": sum(len(metrics) for metrics in self.metrics.values()),
            "unique_metric_names": len(self.metrics)
        }
    
    def get_metric_summary(self, name: str, window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtener resumen de una métrica específica.
        
        Args:
            name: Nombre de la métrica
            window_seconds: Ventana de tiempo en segundos (opcional)
            
        Returns:
            Diccionario con estadísticas de la métrica
        """
        metrics = self.get_metric(name, window_seconds)
        
        if not metrics:
            return {
                "name": name,
                "count": 0,
                "exists": False
            }
        
        values = [m.value for m in metrics]
        sorted_values = sorted(values)
        n = len(values)
        
        return {
            "name": name,
            "count": n,
            "exists": True,
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / n, 4),
            "median": sorted_values[n // 2] if n > 0 else 0,
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
            "first_timestamp": metrics[0].timestamp.isoformat(),
            "last_timestamp": metrics[-1].timestamp.isoformat()
        }
    
    def reset(self):
        """Resetear todas las métricas"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.start_time = datetime.now()
        logger.info("🔄 Metrics reset")


class Timer:
    """Context manager para medir tiempo de ejecución"""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.collector.record(f"{self.metric_name}_duration", elapsed, self.tags)
        if exc_type is None:
            self.collector.increment(f"{self.metric_name}_success", tags=self.tags)
        else:
            self.collector.increment(f"{self.metric_name}_error", tags=self.tags)
        return False


