"""
Advanced Metrics - Métricas Avanzadas
=====================================

Sistema avanzado de métricas y análisis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Sistema de métricas avanzadas"""

    def __init__(self):
        """Inicializa el sistema"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.custom_metrics: Dict[str, Dict[str, Any]] = {}
        self.aggregations: Dict[str, List[float]] = defaultdict(list)

    def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Registra una métrica.

        Args:
            metric_name: Nombre de la métrica
            value: Valor
            tags: Tags adicionales
            timestamp: Timestamp (opcional)
        """
        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            "name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": timestamp.isoformat(),
        }

        self.metrics[metric_name].append(metric)
        self.aggregations[metric_name].append(value)

        # Limitar agregaciones
        if len(self.aggregations[metric_name]) > 1000:
            self.aggregations[metric_name] = self.aggregations[metric_name][-1000:]

    def get_metric_stats(
        self,
        metric_name: str,
        time_window_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de una métrica.

        Args:
            metric_name: Nombre de la métrica
            time_window_minutes: Ventana de tiempo (opcional)

        Returns:
            Estadísticas
        """
        if metric_name not in self.metrics:
            return {"error": f"Métrica '{metric_name}' no encontrada"}

        metrics_list = list(self.metrics[metric_name])

        # Filtrar por ventana de tiempo si se especifica
        if time_window_minutes:
            cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
            metrics_list = [
                m for m in metrics_list
                if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]

        if not metrics_list:
            return {"error": "No hay datos en la ventana de tiempo especificada"}

        values = [m["value"] for m in metrics_list]

        return {
            "metric_name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "p25": self._percentile(values, 25),
            "p75": self._percentile(values, 75),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def create_custom_metric(
        self,
        metric_id: str,
        name: str,
        description: str,
        unit: str = "count",
    ):
        """
        Crea una métrica personalizada.

        Args:
            metric_id: ID de la métrica
            name: Nombre
            description: Descripción
            unit: Unidad
        """
        self.custom_metrics[metric_id] = {
            "name": name,
            "description": description,
            "unit": unit,
            "created_at": datetime.now().isoformat(),
        }

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de todas las métricas"""
        summary = {
            "total_metrics": len(self.metrics),
            "total_data_points": sum(len(m) for m in self.metrics.values()),
            "custom_metrics": len(self.custom_metrics),
            "metrics": {},
        }

        for metric_name in self.metrics.keys():
            summary["metrics"][metric_name] = {
                "count": len(self.metrics[metric_name]),
                "latest_value": self.metrics[metric_name][-1]["value"] if self.metrics[metric_name] else None,
            }

        return summary

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcula percentil"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


