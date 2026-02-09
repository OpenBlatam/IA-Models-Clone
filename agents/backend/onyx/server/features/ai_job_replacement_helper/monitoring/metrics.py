"""
Metrics Collection
"""

import logging
import time
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Recolector de métricas"""
    
    def __init__(self):
        """Inicializar recolector"""
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, list] = defaultdict(list)
        self.gauges: Dict[str, float] = {}
        logger.info("MetricsCollector initialized")
    
    def increment(self, metric_name: str, value: int = 1):
        """Incrementar contador"""
        self.counters[metric_name] += value
    
    def record_timing(self, metric_name: str, duration: float):
        """Registrar tiempo de ejecución"""
        self.timers[metric_name].append(duration)
        # Mantener solo los últimos 1000 valores
        if len(self.timers[metric_name]) > 1000:
            self.timers[metric_name] = self.timers[metric_name][-1000:]
    
    def set_gauge(self, metric_name: str, value: float):
        """Establecer gauge"""
        self.gauges[metric_name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        metrics = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timers": {},
        }
        
        # Calcular estadísticas de timers
        for metric_name, timings in self.timers.items():
            if timings:
                metrics["timers"][metric_name] = {
                    "count": len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "avg": sum(timings) / len(timings),
                    "p95": sorted(timings)[int(len(timings) * 0.95)] if timings else 0,
                }
        
        return metrics
    
    def reset(self):
        """Resetear métricas"""
        self.counters.clear()
        self.timers.clear()
        self.gauges.clear()


# Instancia global
metrics_collector = MetricsCollector()




