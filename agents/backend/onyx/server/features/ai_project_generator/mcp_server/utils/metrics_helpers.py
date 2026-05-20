"""
Metrics Helpers - Utilidades para métricas y observabilidad
===========================================================

Funciones helper para facilitar el registro y exposición de métricas.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Colector de métricas simple sin dependencias externas.
    
    Útil cuando Prometheus no está disponible.
    """
    
    def __init__(self):
        """Inicializar colector de métricas"""
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = {}
        self.timers: Dict[str, list] = {}
    
    def increment(self, metric_name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Incrementar contador.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor a incrementar
            labels: Labels adicionales (opcional)
        """
        key = self._make_key(metric_name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Establecer gauge.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor del gauge
            labels: Labels adicionales (opcional)
        """
        key = self._make_key(metric_name, labels)
        self.gauges[key] = value
    
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Registrar valor en histograma.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor a registrar
            labels: Labels adicionales (opcional)
        """
        key = self._make_key(metric_name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    def record_timing(self, metric_name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Registrar tiempo de ejecución.
        
        Args:
            metric_name: Nombre de la métrica
            duration: Duración en segundos
            labels: Labels adicionales (opcional)
        """
        key = self._make_key(metric_name, labels)
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(duration)
    
    def _make_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Crear clave única para métrica con labels"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{metric_name}{{{label_str}}}"
        return metric_name
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener todas las métricas.
        
        Returns:
            Diccionario con todas las métricas
        """
        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0
                }
                for k, v in self.histograms.items()
            },
            "timers": {
                k: {
                    "count": len(v),
                    "total": sum(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0
                }
                for k, v in self.timers.items()
            }
        }
    
    def reset(self) -> None:
        """Resetear todas las métricas"""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()


# Colector global
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Obtener colector de métricas global.
    
    Returns:
        MetricsCollector instance
    """
    return _global_collector


def record_metric(
    metric_name: str,
    metric_type: str = "counter",
    value: float = 1.0,
    labels: Optional[Dict[str, str]] = None
) -> None:
    """
    Registrar métrica genérica.
    
    Args:
        metric_name: Nombre de la métrica
        metric_type: Tipo (counter, gauge, histogram)
        value: Valor
        labels: Labels adicionales
    """
    collector = get_metrics_collector()
    
    if metric_type == "counter":
        collector.increment(metric_name, int(value), labels)
    elif metric_type == "gauge":
        collector.set_gauge(metric_name, value, labels)
    elif metric_type == "histogram":
        collector.record_histogram(metric_name, value, labels)
    else:
        logger.warning(f"Unknown metric type: {metric_type}")


@contextmanager
def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager para medir tiempo de ejecución.
    
    Args:
        metric_name: Nombre de la métrica
        labels: Labels adicionales
    
    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        get_metrics_collector().record_timing(metric_name, duration, labels)


def timed_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator para medir tiempo de ejecución de funciones.
    
    Args:
        metric_name: Nombre de la métrica
        labels: Labels adicionales
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with measure_time(metric_name, labels):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with measure_time(metric_name, labels):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def increment_counter(metric_name: str, value: int = 1, **labels) -> None:
    """
    Incrementar contador de métrica.
    
    Args:
        metric_name: Nombre de la métrica
        value: Valor a incrementar
        **labels: Labels adicionales
    """
    get_metrics_collector().increment(metric_name, value, labels if labels else None)


def set_gauge(metric_name: str, value: float, **labels) -> None:
    """
    Establecer valor de gauge.
    
    Args:
        metric_name: Nombre de la métrica
        value: Valor
        **labels: Labels adicionales
    """
    get_metrics_collector().set_gauge(metric_name, value, labels if labels else None)


def record_histogram(metric_name: str, value: float, **labels) -> None:
    """
    Registrar valor en histograma.
    
    Args:
        metric_name: Nombre de la métrica
        value: Valor
        **labels: Labels adicionales
    """
    get_metrics_collector().record_histogram(metric_name, value, labels if labels else None)


def format_prometheus_metrics(metrics: Dict[str, Any]) -> str:
    """
    Formatear métricas en formato Prometheus.
    
    Args:
        metrics: Diccionario con métricas
    
    Returns:
        String en formato Prometheus
    """
    lines = []
    
    # Counters
    if "counters" in metrics:
        for key, value in metrics["counters"].items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
    
    # Gauges
    if "gauges" in metrics:
        for key, value in metrics["gauges"].items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
    
    # Histograms
    if "histograms" in metrics:
        for key, stats in metrics["histograms"].items():
            metric_base = key.split('{')[0]
            lines.append(f"# TYPE {metric_base} histogram")
            lines.append(f"{key}_count {stats['count']}")
            lines.append(f"{key}_sum {stats['sum']}")
            lines.append(f"{key}_min {stats['min']}")
            lines.append(f"{key}_max {stats['max']}")
            lines.append(f"{key}_avg {stats['avg']}")
    
    return "\n".join(lines)

