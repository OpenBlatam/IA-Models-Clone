"""
Model Performance Monitor - Monitor de performance en producción
===================================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrica de performance"""
    timestamp: datetime
    latency_ms: float
    throughput_qps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float


class ProductionMonitor:
    """Monitor de performance en producción"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        memory_usage_mb: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None
    ):
        """Registra una request"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        self.total_latency += latency_ms
        
        # Calcular throughput (simplificado)
        throughput = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        # Calcular error rate
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            throughput_qps=throughput,
            error_rate=error_rate,
            memory_usage_mb=memory_usage_mb or 0.0,
            cpu_usage_percent=cpu_usage_percent or 0.0
        )
        
        self.metrics.append(metric)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Obtiene métricas actuales"""
        if not self.metrics:
            return {}
        
        latest = self.metrics[-1]
        
        return {
            "latency_ms": latest.latency_ms,
            "throughput_qps": latest.throughput_qps,
            "error_rate": latest.error_rate,
            "memory_usage_mb": latest.memory_usage_mb,
            "cpu_usage_percent": latest.cpu_usage_percent
        }
    
    def get_average_metrics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Obtiene métricas promedio en ventana de tiempo"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_latency_ms": sum(m.latency_ms for m in recent_metrics) / len(recent_metrics),
            "avg_throughput_qps": sum(m.throughput_qps for m in recent_metrics) / len(recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage_mb": sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
            "avg_cpu_usage_percent": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            "total_requests": self.request_count,
            "total_errors": self.error_count
        }
    
    def check_health(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Verifica salud del sistema basado en umbrales"""
        current = self.get_current_metrics()
        health = {}
        
        for metric_name, threshold in thresholds.items():
            if metric_name in current:
                if metric_name in ["latency_ms", "error_rate", "memory_usage_mb", "cpu_usage_percent"]:
                    health[metric_name] = current[metric_name] <= threshold
                else:  # throughput_qps
                    health[metric_name] = current[metric_name] >= threshold
        
        return health




