"""
Analytics Service
=================

Servicio de analytics y métricas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class AnalyticsService:
    """Servicio de analytics."""
    
    def __init__(self):
        """Inicializar servicio de analytics."""
        self.metrics: List[Metric] = []
        self._logger = logger
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> Metric:
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
        
        Returns:
            Métrica registrada
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
        return metric
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        days: int = 30,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """
        Obtener métricas.
        
        Args:
            name: Nombre de métrica específica
            days: Días hacia atrás
            tags: Tags para filtrar
        
        Returns:
            Lista de métricas
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_date
        ]
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if tags:
            for key, value in tags.items():
                metrics = [
                    m for m in metrics
                    if m.tags.get(key) == value
                ]
        
        return sorted(metrics, key=lambda m: m.timestamp, reverse=True)
    
    def get_average(self, name: str, days: int = 30) -> float:
        """
        Obtener promedio de una métrica.
        
        Args:
            name: Nombre de la métrica
            days: Días hacia atrás
        
        Returns:
            Promedio
        """
        metrics = self.get_metrics(name=name, days=days)
        if not metrics:
            return 0.0
        
        return sum(m.value for m in metrics) / len(metrics)
    
    def get_sum(self, name: str, days: int = 30) -> float:
        """
        Obtener suma de una métrica.
        
        Args:
            name: Nombre de la métrica
            days: Días hacia atrás
        
        Returns:
            Suma
        """
        metrics = self.get_metrics(name=name, days=days)
        return sum(m.value for m in metrics)
    
    def get_artist_statistics(self, artist_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Obtener estadísticas de un artista.
        
        Args:
            artist_id: ID del artista
            days: Días hacia atrás
        
        Returns:
            Estadísticas
        """
        artist_metrics = self.get_metrics(
            days=days,
            tags={"artist_id": artist_id}
        )
        
        stats = {
            "artist_id": artist_id,
            "period_days": days,
            "total_metrics": len(artist_metrics),
            "metrics_by_name": {}
        }
        
        # Agrupar por nombre de métrica
        for metric in artist_metrics:
            if metric.name not in stats["metrics_by_name"]:
                stats["metrics_by_name"][metric.name] = {
                    "count": 0,
                    "sum": 0.0,
                    "average": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
            
            m_stats = stats["metrics_by_name"][metric.name]
            m_stats["count"] += 1
            m_stats["sum"] += metric.value
            m_stats["min"] = min(m_stats["min"], metric.value)
            m_stats["max"] = max(m_stats["max"], metric.value)
        
        # Calcular promedios
        for name, m_stats in stats["metrics_by_name"].items():
            if m_stats["count"] > 0:
                m_stats["average"] = m_stats["sum"] / m_stats["count"]
            if m_stats["min"] == float('inf'):
                m_stats["min"] = 0.0
            if m_stats["max"] == float('-inf'):
                m_stats["max"] = 0.0
        
        return stats




