"""
Metrics System
==============

Sistema de métricas para observabilidad del sistema de movimiento robótico.
"""

import time
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Valor de métrica con timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Métrica con historial."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    lock: Lock = field(default_factory=Lock)
    unit: str = ""
    description: str = ""
    
    def add_value(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Agregar valor a la métrica."""
        with self.lock:
            self.values.append(MetricValue(
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            ))
    
    def get_latest(self) -> Optional[float]:
        """Obtener último valor."""
        with self.lock:
            if not self.values:
                return None
            return self.values[-1].value
    
    def get_average(self, window_seconds: Optional[float] = None) -> Optional[float]:
        """Obtener promedio en ventana de tiempo."""
        with self.lock:
            if not self.values:
                return None
            
            if window_seconds is None:
                # Promedio de todos los valores
                return sum(v.value for v in self.values) / len(self.values)
            
            # Promedio en ventana de tiempo
            cutoff_time = time.time() - window_seconds
            recent_values = [v.value for v in self.values if v.timestamp >= cutoff_time]
            
            if not recent_values:
                return None
            
            return sum(recent_values) / len(recent_values)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la métrica."""
        with self.lock:
            if not self.values:
                return {
                    "name": self.name,
                    "count": 0,
                    "latest": None,
                    "average": None,
                    "min": None,
                    "max": None
                }
            
            values = [v.value for v in self.values]
            return {
                "name": self.name,
                "count": len(self.values),
                "latest": values[-1],
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "unit": self.unit,
                "description": self.description
            }


class MetricsCollector:
    """
    Recolector de métricas del sistema.
    
    Proporciona métricas para:
    - Performance de algoritmos
    - Tiempos de ejecución
    - Uso de recursos
    - Errores y excepciones
    - Estado del robot
    """
    
    def __init__(self):
        """Inicializar recolector de métricas."""
        self.metrics: Dict[str, Metric] = {}
        self.lock = Lock()
        self.start_time = time.time()
    
    def register_metric(
        self,
        name: str,
        unit: str = "",
        description: str = ""
    ) -> Metric:
        """
        Registrar nueva métrica.
        
        Args:
            name: Nombre de la métrica
            unit: Unidad de medida
            description: Descripción
            
        Returns:
            Métrica registrada
        """
        with self.lock:
            if name in self.metrics:
                logger.warning(f"Metric {name} already registered, overwriting")
            
            metric = Metric(
                name=name,
                unit=unit,
                description=description
            )
            self.metrics[name] = metric
            return metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Obtener métrica por nombre."""
        with self.lock:
            return self.metrics.get(name)
    
    def record_value(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Registrar valor en métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor a registrar
            tags: Tags opcionales
        """
        metric = self.get_metric(name)
        if metric is None:
            # Auto-registrar si no existe
            metric = self.register_metric(name)
        
        metric.add_value(value, tags)
    
    def increment_counter(self, name: str, amount: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Incrementar contador."""
        metric = self.get_metric(name)
        if metric is None:
            metric = self.register_metric(name, unit="count", description=f"Counter: {name}")
        
        current = metric.get_latest() or 0.0
        metric.add_value(current + amount, tags)
    
    def record_timing(
        self,
        name: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Registrar tiempo de ejecución."""
        self.record_value(name, duration, tags)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las métricas con estadísticas."""
        with self.lock:
            return {
                name: metric.get_statistics()
                for name, metric in self.metrics.items()
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        all_metrics = self.get_all_metrics()
        
        # Categorizar métricas
        counters = {}
        timings = {}
        gauges = {}
        
        for name, stats in all_metrics.items():
            if "count" in name.lower() or "counter" in stats.get("description", "").lower():
                counters[name] = stats
            elif "time" in name.lower() or "duration" in name.lower() or "latency" in name.lower():
                timings[name] = stats
            else:
                gauges[name] = stats
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_metrics": len(all_metrics),
            "counters": counters,
            "timings": timings,
            "gauges": gauges,
            "all_metrics": all_metrics
        }
    
    def reset_metric(self, name: str):
        """Resetear métrica."""
        metric = self.get_metric(name)
        if metric:
            with metric.lock:
                metric.values.clear()


# Instancia global
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtener instancia global del recolector de métricas."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Función helper para registrar métrica."""
    get_metrics_collector().record_value(name, value, tags)


def increment_counter(name: str, amount: float = 1.0, tags: Optional[Dict[str, str]] = None):
    """Función helper para incrementar contador."""
    get_metrics_collector().increment_counter(name, amount, tags)


def record_timing(name: str, duration: float, tags: Optional[Dict[str, str]] = None):
    """Función helper para registrar tiempo."""
    get_metrics_collector().record_timing(name, duration, tags)






