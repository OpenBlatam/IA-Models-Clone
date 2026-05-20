"""
Performance Analyzer - Sistema de análisis de performance
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrica de performance"""
    name: str
    value: float
    unit: str
    timestamp: datetime


class PerformanceAnalyzer:
    """Analizador de performance"""

    def __init__(self):
        """Inicializar analizador"""
        self.metrics_history: List[PerformanceMetric] = []

    def measure_operation(
        self,
        operation_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Medir performance de una operación.

        Args:
            operation_name: Nombre de la operación
            operation: Función a medir
            *args: Argumentos posicionales
            **kwargs: Argumentos de palabra clave

        Returns:
            Resultado con métricas
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = operation(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Registrar métrica
        metric = PerformanceMetric(
            name=operation_name,
            value=execution_time,
            unit="seconds",
            timestamp=datetime.utcnow()
        )
        self.metrics_history.append(metric)
        
        return {
            "operation": operation_name,
            "success": success,
            "execution_time": execution_time,
            "memory_used": memory_used,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error,
            "result": result
        }

    def analyze_performance(
        self,
        operation_name: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Analizar performance.

        Args:
            operation_name: Filtrar por nombre de operación
            limit: Límite de métricas

        Returns:
            Análisis de performance
        """
        metrics = self.metrics_history
        
        if operation_name:
            metrics = [m for m in metrics if m.name == operation_name]
        
        metrics = metrics[-limit:]
        
        if not metrics:
            return {"error": "No hay métricas disponibles"}
        
        values = [m.value for m in metrics]
        
        return {
            "operation": operation_name or "all",
            "total_operations": len(metrics),
            "avg_execution_time": sum(values) / len(values),
            "min_execution_time": min(values),
            "max_execution_time": max(values),
            "total_execution_time": sum(values),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in metrics
            ]
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de performance.

        Returns:
            Estadísticas
        """
        if not self.metrics_history:
            return {"error": "No hay métricas disponibles"}
        
        # Agrupar por operación
        by_operation = {}
        for metric in self.metrics_history:
            if metric.name not in by_operation:
                by_operation[metric.name] = []
            by_operation[metric.name].append(metric.value)
        
        stats = {}
        for op_name, values in by_operation.items():
            stats[op_name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values)
            }
        
        return {
            "total_operations": len(self.metrics_history),
            "unique_operations": len(by_operation),
            "by_operation": stats
        }

    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria (simulado)"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def clear_metrics(self):
        """Limpiar métricas"""
        self.metrics_history.clear()
        logger.info("Métricas de performance limpiadas")






