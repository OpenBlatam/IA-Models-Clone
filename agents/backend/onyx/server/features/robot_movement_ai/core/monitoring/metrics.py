"""
Metrics Collection
=================

Sistema de recolección de métricas.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricValue:
    """Valor de métrica con timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)


class Counter:
    """Contador de métricas."""
    
    def __init__(self, name: str):
        """Inicializar contador."""
        self.name = name
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, value: int = 1):
        """Incrementar contador."""
        with self._lock:
            self._value += value
    
    def get(self) -> int:
        """Obtener valor actual."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Resetear contador."""
        with self._lock:
            self._value = 0


class Gauge:
    """Gauge (valor que sube y baja)."""
    
    def __init__(self, name: str):
        """Inicializar gauge."""
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()
    
    def set(self, value: float):
        """Establecer valor."""
        with self._lock:
            self._value = value
    
    def inc(self, value: float = 1.0):
        """Incrementar valor."""
        with self._lock:
            self._value += value
    
    def dec(self, value: float = 1.0):
        """Decrementar valor."""
        with self._lock:
            self._value -= value
    
    def get(self) -> float:
        """Obtener valor actual."""
        with self._lock:
            return self._value


class Histogram:
    """Histograma de valores."""
    
    def __init__(self, name: str, buckets: Optional[List[float]] = None):
        """
        Inicializar histograma.
        
        Args:
            name: Nombre
            buckets: Buckets (opcional)
        """
        self.name = name
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._values = deque(maxlen=10000)  # Limitar a 10k valores
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Observar valor."""
        with self._lock:
            self._values.append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas."""
        with self._lock:
            if not self._values:
                return {}
            
            values = list(self._values)
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            return {
                "count": n,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / n,
                "p50": sorted_values[n // 2] if n > 0 else 0,
                "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
            }
    
    def get_bucket_counts(self) -> Dict[str, int]:
        """Obtener conteos por bucket."""
        with self._lock:
            if not self._values:
                return {}
            
            counts = defaultdict(int)
            for value in self._values:
                for bucket in self.buckets:
                    if value <= bucket:
                        counts[f"le_{bucket}"] += 1
                        break
                else:
                    counts["le_inf"] += 1
            
            return dict(counts)


class Timer:
    """Timer para medir duración."""
    
    def __init__(self, name: str):
        """Inicializar timer."""
        self.name = name
        self._start_time: Optional[float] = None
        self._histogram = Histogram(f"{name}_duration")
    
    def start(self):
        """Iniciar timer."""
        self._start_time = time.time()
    
    def stop(self) -> float:
        """
        Detener timer y retornar duración.
        
        Returns:
            Duración en segundos
        """
        if self._start_time is None:
            raise ValueError("Timer no iniciado")
        
        duration = time.time() - self._start_time
        self._histogram.observe(duration)
        self._start_time = None
        return duration
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas."""
        return self._histogram.get_stats()


class MetricsCollector:
    """
    Recolector de métricas.
    """
    
    def __init__(self):
        """Inicializar recolector."""
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()
    
    def counter(self, name: str) -> Counter:
        """Obtener o crear contador."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name)
            return self._counters[name]
    
    def gauge(self, name: str) -> Gauge:
        """Obtener o crear gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name)
            return self._gauges[name]
    
    def histogram(self, name: str, buckets: Optional[List[float]] = None) -> Histogram:
        """Obtener o crear histograma."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, buckets)
            return self._histograms[name]
    
    def timer(self, name: str) -> Timer:
        """Obtener o crear timer."""
        with self._lock:
            if name not in self._timers:
                self._timers[name] = Timer(name)
            return self._timers[name]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas."""
        with self._lock:
            return {
                "counters": {name: counter.get() for name, counter in self._counters.items()},
                "gauges": {name: gauge.get() for name, gauge in self._gauges.items()},
                "histograms": {
                    name: hist.get_stats() for name, hist in self._histograms.items()
                },
                "timers": {
                    name: timer.get_stats() for name, timer in self._timers.items()
                }
            }
    
    def reset_all(self):
        """Resetear todas las métricas."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.set(0.0)
            self._histograms.clear()
            self._timers.clear()


# Instancia global
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Obtener recolector global."""
    return _global_collector

