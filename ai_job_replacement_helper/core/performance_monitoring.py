"""
Performance Monitoring Service - Monitoreo de rendimiento
=========================================================

Sistema para monitorear y analizar rendimiento del sistema.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot de rendimiento"""
    timestamp: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, float]


class PerformanceMonitoringService:
    """Servicio de monitoreo de rendimiento"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.metrics: List[PerformanceMetric] = []
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        logger.info("PerformanceMonitoringService initialized")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        tags: Optional[Dict[str, str]] = None
    ) -> PerformanceMetric:
        """Registrar métrica"""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
        )
        
        self.metrics.append(metric)
        
        # Mantener solo últimas 10000 métricas
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
        
        return metric
    
    def record_request_time(
        self,
        endpoint: str,
        method: str,
        duration_ms: float
    ):
        """Registrar tiempo de request"""
        key = f"{method}:{endpoint}"
        self.request_times[key].append(duration_ms)
        
        # Mantener solo últimos 1000 requests por endpoint
        if len(self.request_times[key]) > 1000:
            self.request_times[key] = self.request_times[key][-1000:]
    
    def get_endpoint_stats(
        self,
        endpoint: str,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Obtener estadísticas de endpoint"""
        key = f"{method}:{endpoint}"
        times = self.request_times.get(key, [])
        
        if not times:
            return {
                "endpoint": endpoint,
                "method": method,
                "total_requests": 0,
            }
        
        return {
            "endpoint": endpoint,
            "method": method,
            "total_requests": len(times),
            "avg_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": self._calculate_percentile(times, 95),
            "p99_time_ms": self._calculate_percentile(times, 99),
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calcular percentil"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_performance_snapshot(
        self,
        minutes: int = 5
    ) -> PerformanceSnapshot:
        """Obtener snapshot de rendimiento"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]
        
        # Agrupar por nombre de métrica
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        summary = {
            name: {
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "count": len(values),
            }
            for name, values in metric_groups.items()
        }
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            metrics=recent_metrics,
            summary=summary,
        )
    
    def detect_slow_endpoints(
        self,
        threshold_ms: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """Detectar endpoints lentos"""
        slow_endpoints = []
        
        for key, times in self.request_times.items():
            if not times:
                continue
            
            avg_time = sum(times) / len(times)
            if avg_time > threshold_ms:
                method, endpoint = key.split(":", 1)
                slow_endpoints.append({
                    "endpoint": endpoint,
                    "method": method,
                    "avg_time_ms": avg_time,
                    "total_requests": len(times),
                })
        
        # Ordenar por tiempo promedio
        slow_endpoints.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        
        return slow_endpoints
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de salud del sistema"""
        snapshot = self.get_performance_snapshot(minutes=1)
        
        # Calcular métricas clave
        total_requests = sum(
            len(times) for times in self.request_times.values()
        )
        
        slow_endpoints_count = len(self.detect_slow_endpoints())
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests_last_minute": total_requests,
            "active_endpoints": len(self.request_times),
            "slow_endpoints": slow_endpoints_count,
            "metrics_summary": snapshot.summary,
        }




