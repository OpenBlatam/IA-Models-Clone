"""
Sistema de métricas en tiempo real
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import threading


@dataclass
class MetricPoint:
    """Punto de métrica"""
    timestamp: float
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class RealtimeMetrics:
    """Sistema de métricas en tiempo real"""
    
    def __init__(self, window_seconds: int = 300):
        """
        Inicializa el sistema
        
        Args:
            window_seconds: Ventana de tiempo en segundos
        """
        self.window_seconds = window_seconds
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Registra una métrica
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        """
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            self.metrics[metric_name].append(point)
    
    def get_metric(self, metric_name: str, window_seconds: Optional[int] = None) -> List[MetricPoint]:
        """
        Obtiene métricas en una ventana
        
        Args:
            metric_name: Nombre de la métrica
            window_seconds: Ventana de tiempo
            
        Returns:
            Lista de puntos de métrica
        """
        window = window_seconds or self.window_seconds
        cutoff = time.time() - window
        
        with self.lock:
            points = list(self.metrics.get(metric_name, deque()))
            return [p for p in points if p.timestamp >= cutoff]
    
    def get_statistics(self, metric_name: str, window_seconds: Optional[int] = None) -> Dict:
        """
        Obtiene estadísticas de una métrica
        
        Args:
            metric_name: Nombre de la métrica
            window_seconds: Ventana de tiempo
            
        Returns:
            Diccionario con estadísticas
        """
        points = self.get_metric(metric_name, window_seconds)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": None
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "latest": values[-1] if values else None
        }
    
    def get_all_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Obtiene todas las métricas"""
        with self.lock:
            return {
                name: list(points)
                for name, points in self.metrics.items()
            }
    
    def get_dashboard_data(self) -> Dict:
        """Obtiene datos para dashboard"""
        dashboard = {}
        
        for metric_name in self.metrics.keys():
            stats = self.get_statistics(metric_name)
            recent_points = self.get_metric(metric_name, window_seconds=60)
            
            dashboard[metric_name] = {
                "statistics": stats,
                "recent_points": [
                    {"timestamp": p.timestamp, "value": p.value}
                    for p in recent_points[-20:]  # Últimos 20 puntos
                ]
            }
        
        return dashboard






