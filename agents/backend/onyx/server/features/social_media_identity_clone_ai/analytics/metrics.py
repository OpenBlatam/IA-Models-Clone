"""
Sistema de métricas y monitoreo
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Valor de métrica con timestamp"""
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Colector de métricas para el sistema"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)
        self.lock = Lock()
        self.start_time = datetime.now()
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Incrementa un contador"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            self.counters[key] += value
            self._record_metric(metric_name, value, tags, "counter")
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Establece un gauge (valor actual)"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            self.gauges[key] = value
            self._record_metric(metric_name, value, tags, "gauge")
    
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registra un valor en histograma"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            self.histograms[key].append(value)
            if len(self.histograms[key]) > self.max_history:
                self.histograms[key] = self.histograms[key][-self.max_history:]
            self._record_metric(metric_name, value, tags, "histogram")
    
    def timer(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager para medir tiempo"""
        return TimerContext(self, metric_name, tags)
    
    def _build_key(self, metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        """Construye clave única para métrica con tags"""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_str}]"
    
    def _record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]], metric_type: str):
        """Registra métrica en historial"""
        metric_value = MetricValue(
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        key = self._build_key(metric_name, tags)
        self.metrics[key].append(metric_value)
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Obtiene valor de contador"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            return self.counters.get(key, 0)
    
    def get_gauge(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Obtiene valor de gauge"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            return self.gauges.get(key)
    
    def get_histogram_stats(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Obtiene estadísticas de histograma"""
        with self.lock:
            key = self._build_key(metric_name, tags)
            values = self.histograms.get(key, [])
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }
    
    def _percentile(self, values: list, percentile: int) -> float:
        """Calcula percentil"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtiene todas las métricas"""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return {
                "uptime_seconds": uptime,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    k: self.get_histogram_stats(k.split("[")[0]) 
                    for k in self.histograms.keys()
                },
                "metric_count": len(self.metrics),
            }
    
    def reset(self):
        """Resetea todas las métricas"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.start_time = datetime.now()


class TimerContext:
    """Context manager para medir tiempo de operaciones"""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, tags: Optional[Dict[str, str]]):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.collector.histogram(self.metric_name, elapsed * 1000, self.tags)  # en milisegundos
        return False


# Singleton global
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtiene instancia singleton del colector de métricas"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector




