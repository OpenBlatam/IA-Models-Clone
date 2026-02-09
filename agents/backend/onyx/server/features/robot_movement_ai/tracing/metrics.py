"""
Metrics Collector - Sistema de métricas mejorado
"""
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics


@dataclass
class MetricData:
    """Datos de una métrica individual."""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricSummary:
    """Resumen estadístico de una métrica."""
    name: str
    count: int
    sum: float
    avg: float
    min: float
    max: float
    median: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    std_dev: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Colector de métricas mejorado con estadísticas."""
    
    def __init__(self, max_metrics_per_name: int = 10000):
        """
        Inicializar colector de métricas.
        
        Args:
            max_metrics_per_name: Número máximo de métricas a mantener por nombre
        """
        self.metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self.max_metrics_per_name = max_metrics_per_name
        self._lock = None
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ):
        """
        Registra una métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            tags: Tags adicionales
            timestamp: Timestamp (opcional, usa tiempo actual si no se proporciona)
        """
        metric_data = MetricData(
            name=name,
            value=value,
            tags=tags or {},
            timestamp=timestamp or time.time()
        )
        
        metric_list = self.metrics[name]
        metric_list.append(metric_data)
        
        if len(metric_list) > self.max_metrics_per_name:
            metric_list.pop(0)
    
    def get_metric(self, name: str) -> List[MetricData]:
        """
        Obtiene métricas por nombre.
        
        Args:
            name: Nombre de la métrica
        
        Returns:
            Lista de métricas
        """
        return self.metrics.get(name, [])
    
    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Obtiene resumen estadístico de una métrica.
        
        Args:
            name: Nombre de la métrica
        
        Returns:
            Resumen estadístico o None si no hay métricas
        """
        metric_list = self.get_metric(name)
        if not metric_list:
            return None
        
        values = [m.value for m in metric_list]
        sorted_values = sorted(values)
        
        count = len(values)
        sum_val = sum(values)
        avg = sum_val / count if count > 0 else 0.0
        min_val = min(values)
        max_val = max(values)
        
        median = statistics.median(values) if count > 0 else None
        p95 = sorted_values[int(count * 0.95)] if count > 1 else None
        p99 = sorted_values[int(count * 0.99)] if count > 1 else None
        std_dev = statistics.stdev(values) if count > 1 else None
        
        tags = metric_list[0].tags if metric_list else {}
        
        return MetricSummary(
            name=name,
            count=count,
            sum=sum_val,
            avg=avg,
            min=min_val,
            max=max_val,
            median=median,
            p95=p95,
            p99=p99,
            std_dev=std_dev,
            tags=tags
        )
    
    def get_all_metrics(self) -> Dict[str, List[MetricData]]:
        """
        Obtiene todas las métricas.
        
        Returns:
            Diccionario con todas las métricas
        """
        return dict(self.metrics)
    
    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """
        Obtiene resúmenes de todas las métricas.
        
        Returns:
            Diccionario con resúmenes
        """
        return {
            name: self.get_summary(name)
            for name in self.metrics.keys()
            if self.get_summary(name) is not None
        }
    
    def clear_metric(self, name: str):
        """
        Limpia métricas de un nombre específico.
        
        Args:
            name: Nombre de la métrica
        """
        if name in self.metrics:
            del self.metrics[name]
    
    def clear_all(self):
        """Limpia todas las métricas."""
        self.metrics.clear()
    
    def get_metrics_in_range(
        self,
        name: str,
        start_time: float,
        end_time: float
    ) -> List[MetricData]:
        """
        Obtiene métricas en un rango de tiempo.
        
        Args:
            name: Nombre de la métrica
            start_time: Timestamp de inicio
            end_time: Timestamp de fin
        
        Returns:
            Lista de métricas en el rango
        """
        metric_list = self.get_metric(name)
        return [
            m for m in metric_list
            if start_time <= m.timestamp <= end_time
        ]


_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Obtener colector global de métricas."""
    return _global_collector

