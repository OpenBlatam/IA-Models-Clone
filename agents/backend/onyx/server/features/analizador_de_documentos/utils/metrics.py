"""
Sistema de Métricas y Monitoring
=================================

Sistema para recopilar y exportar métricas de rendimiento,
uso y calidad del sistema.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import json

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica individual"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Recopilador de métricas"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()
        self.max_history = max_history
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registrar una métrica"""
        metric = Metric(
            name=name,
            value=value,
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Incrementar contador"""
        with self.lock:
            self.counters[name] += value
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Establecer gauge"""
        with self.lock:
            self.gauges[name] = value
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registrar valor en histograma"""
        with self.lock:
            self.histograms[name].append(value)
            # Mantener solo últimos N valores
            if len(self.histograms[name]) > self.max_history:
                self.histograms[name] = self.histograms[name][-self.max_history:]
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager para medir tiempo"""
        return Timer(self, name, tags)
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Obtener estadísticas de una métrica"""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            if not values:
                return {}
            
            hist_values = self.histograms.get(name, [])
            if hist_values:
                values.extend(hist_values)
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "sum": sum(values),
                "last_value": values[-1] if values else None
            }
    
    def get_counter(self, name: str) -> int:
        """Obtener valor de contador"""
        with self.lock:
            return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Obtener valor de gauge"""
        with self.lock:
            return self.gauges.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "mean": sum(values) / len(values) if values else None
                    }
                    for name, values in self.histograms.items()
                },
                "metrics": {
                    name: self.get_stats(name)
                    for name in self.metrics.keys()
                }
            }
    
    def reset(self):
        """Resetear todas las métricas"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


class Timer:
    """Context manager para medir tiempo"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.collector.record(f"{self.name}.duration", elapsed, self.tags)
        self.collector.histogram(f"{self.name}.duration", elapsed, self.tags)
        return False


class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.request_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.success_count = 0
    
    def record_request(self, endpoint: str, duration: float, success: bool):
        """Registrar una petición"""
        self.collector.record(
            "api.request.duration",
            duration,
            tags={"endpoint": endpoint, "status": "success" if success else "error"}
        )
        self.collector.histogram("api.request.duration", duration)
        self.request_times.append(duration)
        
        if success:
            self.success_count += 1
            self.collector.increment("api.request.success", tags={"endpoint": endpoint})
        else:
            self.error_count += 1
            self.collector.increment("api.request.error", tags={"endpoint": endpoint})
    
    def record_analysis(self, task: str, duration: float, success: bool):
        """Registrar análisis"""
        self.collector.record(
            "analysis.duration",
            duration,
            tags={"task": task, "status": "success" if success else "error"}
        )
        
        if success:
            self.collector.increment("analysis.success", tags={"task": task})
        else:
            self.collector.increment("analysis.error", tags={"task": task})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        if not self.request_times:
            return {}
        
        durations = list(self.request_times)
        return {
            "total_requests": len(durations),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p50": self._percentile(durations, 50),
            "p95": self._percentile(durations, 95),
            "p99": self._percentile(durations, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcular percentil"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtener todas las estadísticas"""
        return {
            "performance": self.get_performance_stats(),
            "metrics": self.collector.get_all_metrics()
        }


# Instancia global
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Obtener instancia global del monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
















