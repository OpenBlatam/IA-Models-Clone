"""
Sistema de métricas y dashboards para Robot Movement AI v2.0
Métricas avanzadas con agregaciones y visualizaciones
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class MetricType(str, Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Métrica individual"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: MetricType = MetricType.GAUGE


class MetricsCollector:
    """Recolector de métricas"""
    
    def __init__(self):
        """Inicializar recolector"""
        self.metrics: List[Metric] = []
        self.max_metrics: int = 100000
    
    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """
        Registrar métrica
        
        Args:
            name: Nombre de la métrica
            value: Valor
            labels: Etiquetas adicionales
            metric_type: Tipo de métrica
        """
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.metrics.append(metric)
        
        # Limpiar métricas antiguas si excede límite
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def query(
        self,
        name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Metric]:
        """
        Consultar métricas
        
        Args:
            name: Filtrar por nombre
            labels: Filtrar por labels
            start_time: Fecha de inicio
            end_time: Fecha de fin
            
        Returns:
            Lista de métricas que coinciden
        """
        results = self.metrics
        
        if name:
            results = [m for m in results if m.name == name]
        
        if labels:
            results = [
                m for m in results
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]
        
        if start_time:
            results = [m for m in results if m.timestamp >= start_time]
        
        if end_time:
            results = [m for m in results if m.timestamp <= end_time]
        
        return results
    
    def aggregate(
        self,
        name: str,
        aggregation: str = "sum",
        labels: Optional[Dict[str, str]] = None,
        time_range: Optional[timedelta] = None
    ) -> float:
        """
        Agregar métricas
        
        Args:
            name: Nombre de la métrica
            aggregation: Tipo de agregación (sum, avg, min, max, count)
            labels: Filtrar por labels
            time_range: Rango de tiempo
            
        Returns:
            Valor agregado
        """
        metrics = self.query(name=name, labels=labels)
        
        if time_range:
            cutoff = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        if not metrics:
            return 0.0
        
        values = [m.value for m in metrics]
        
        if aggregation == "sum":
            return sum(values)
        elif aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            raise ValueError(f"Agregación no soportada: {aggregation}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Obtener datos para dashboard
        
        Returns:
            Datos estructurados para dashboard
        """
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Métricas de robots
        robot_metrics = self.query(name="robot", start_time=last_hour)
        robot_count = len(set(m.labels.get('robot_id') for m in robot_metrics if m.labels.get('robot_id')))
        
        # Métricas de movements
        movement_metrics = self.query(name="movement", start_time=last_hour)
        movement_count = self.aggregate("movement", "count", time_range=timedelta(hours=1))
        
        # Métricas de rendimiento
        performance_metrics = self.query(name="performance", start_time=last_hour)
        avg_response_time = self.aggregate("response_time", "avg", time_range=timedelta(hours=1))
        
        # Métricas de errores
        error_metrics = self.query(name="error", start_time=last_hour)
        error_count = self.aggregate("error", "count", time_range=timedelta(hours=1))
        
        return {
            'summary': {
                'robots_active': robot_count,
                'movements_last_hour': movement_count,
                'avg_response_time_ms': avg_response_time,
                'errors_last_hour': error_count
            },
            'time_series': {
                'movements': self._get_time_series("movement", last_hour, timedelta(minutes=5)),
                'errors': self._get_time_series("error", last_hour, timedelta(minutes=5)),
                'response_time': self._get_time_series("response_time", last_hour, timedelta(minutes=5))
            },
            'top_robots': self._get_top_robots(last_hour),
            'error_breakdown': self._get_error_breakdown(last_hour)
        }
    
    def _get_time_series(
        self,
        metric_name: str,
        start_time: datetime,
        interval: timedelta
    ) -> List[Dict[str, Any]]:
        """Obtener serie de tiempo"""
        metrics = self.query(name=metric_name, start_time=start_time)
        
        # Agrupar por intervalo
        buckets = {}
        for metric in metrics:
            bucket_time = metric.timestamp - timedelta(
                seconds=(metric.timestamp - start_time).total_seconds() % interval.total_seconds()
            )
            bucket_key = bucket_time.isoformat()
            
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(metric.value)
        
        # Calcular promedio por bucket
        series = []
        for bucket_time, values in sorted(buckets.items()):
            series.append({
                'timestamp': bucket_time,
                'value': sum(values) / len(values) if values else 0
            })
        
        return series
    
    def _get_top_robots(self, start_time: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener top robots por actividad"""
        metrics = self.query(name="movement", start_time=start_time)
        
        robot_counts = {}
        for metric in metrics:
            robot_id = metric.labels.get('robot_id')
            if robot_id:
                robot_counts[robot_id] = robot_counts.get(robot_id, 0) + 1
        
        top_robots = sorted(
            robot_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {'robot_id': robot_id, 'movement_count': count}
            for robot_id, count in top_robots
        ]
    
    def _get_error_breakdown(self, start_time: datetime) -> Dict[str, int]:
        """Obtener desglose de errores"""
        metrics = self.query(name="error", start_time=start_time)
        
        error_types = {}
        for metric in metrics:
            error_type = metric.labels.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return error_types


# Instancia global
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtener instancia global del recolector de métricas"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector



