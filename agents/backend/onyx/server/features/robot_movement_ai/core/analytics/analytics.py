"""
Analytics System
================

Sistema de análisis y reportes avanzados.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsMetric:
    """Métrica de análisis."""
    name: str
    value: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Reporte de performance."""
    period_start: str
    period_end: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    metrics: List[AnalyticsMetric] = field(default_factory=list)


class AnalyticsEngine:
    """
    Motor de análisis.
    
    Analiza datos del sistema y genera reportes.
    """
    
    def __init__(self):
        """Inicializar motor de análisis."""
        self.metrics_history: List[AnalyticsMetric] = []
        self.operations_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
    
    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar operación.
        
        Args:
            operation_type: Tipo de operación
            duration: Duración en segundos
            success: Si fue exitosa
            metadata: Metadata adicional
        """
        operation = {
            "type": operation_type,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.operations_history.append(operation)
        
        # Limitar tamaño del historial
        if len(self.operations_history) > self.max_history_size:
            self.operations_history = self.operations_history[-self.max_history_size:]
    
    def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            metadata: Metadata adicional
        """
        metric = AnalyticsMetric(
            name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metric)
        
        # Limitar tamaño del historial
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def generate_performance_report(
        self,
        period_hours: int = 24,
        operation_type: Optional[str] = None
    ) -> PerformanceReport:
        """
        Generar reporte de performance.
        
        Args:
            period_hours: Horas del período
            operation_type: Tipo de operación (opcional)
            
        Returns:
            Reporte de performance
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(hours=period_hours)
        
        # Filtrar operaciones del período
        operations = [
            op for op in self.operations_history
            if datetime.fromisoformat(op["timestamp"]) >= period_start
        ]
        
        if operation_type:
            operations = [op for op in operations if op["type"] == operation_type]
        
        if not operations:
            return PerformanceReport(
                period_start=period_start.isoformat(),
                period_end=period_end.isoformat(),
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_response_time=0.0,
                p50_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                throughput=0.0,
                error_rate=0.0
            )
        
        # Calcular estadísticas
        total_operations = len(operations)
        successful_operations = sum(1 for op in operations if op["success"])
        failed_operations = total_operations - successful_operations
        
        durations = [op["duration"] for op in operations]
        durations_sorted = sorted(durations)
        
        average_response_time = sum(durations) / len(durations)
        p50_response_time = durations_sorted[len(durations_sorted) // 2]
        p95_response_time = durations_sorted[int(len(durations_sorted) * 0.95)]
        p99_response_time = durations_sorted[int(len(durations_sorted) * 0.99)]
        
        # Throughput (operaciones por segundo)
        period_seconds = period_hours * 3600
        throughput = total_operations / period_seconds if period_seconds > 0 else 0.0
        
        # Error rate
        error_rate = failed_operations / total_operations if total_operations > 0 else 0.0
        
        # Métricas del período
        period_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= period_start
        ]
        
        return PerformanceReport(
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            average_response_time=average_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            throughput=throughput,
            error_rate=error_rate,
            metrics=period_metrics
        )
    
    def analyze_trends(
        self,
        metric_name: str,
        period_hours: int = 24,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Analizar tendencias de métrica.
        
        Args:
            metric_name: Nombre de la métrica
            period_hours: Período de análisis
            window_minutes: Ventana de tiempo para promedios
            
        Returns:
            Análisis de tendencias
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(hours=period_hours)
        
        # Filtrar métricas
        metrics = [
            m for m in self.metrics_history
            if m.name == metric_name
            and datetime.fromisoformat(m.timestamp) >= period_start
        ]
        
        if not metrics:
            return {
                "metric_name": metric_name,
                "trend": "no_data",
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "change": 0.0
            }
        
        values = [m.value for m in metrics]
        average = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)
        
        # Calcular tendencia
        first_half = values[:len(values) // 2]
        second_half = values[len(values) // 2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0.0
        second_avg = sum(second_half) / len(second_half) if second_half else 0.0
        
        change = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0.0
        
        if abs(change) < 5:
            trend = "stable"
        elif change > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "metric_name": metric_name,
            "trend": trend,
            "average": average,
            "min": min_value,
            "max": max_value,
            "change_percent": change,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg
        }
    
    def get_operation_statistics(
        self,
        operation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de operaciones.
        
        Args:
            operation_type: Tipo de operación (opcional)
            
        Returns:
            Estadísticas
        """
        operations = self.operations_history
        if operation_type:
            operations = [op for op in operations if op["type"] == operation_type]
        
        if not operations:
            return {
                "total": 0,
                "by_type": {},
                "success_rate": 0.0,
                "average_duration": 0.0
            }
        
        # Estadísticas por tipo
        by_type = defaultdict(lambda: {"count": 0, "success": 0, "total_duration": 0.0})
        
        for op in operations:
            op_type = op["type"]
            by_type[op_type]["count"] += 1
            if op["success"]:
                by_type[op_type]["success"] += 1
            by_type[op_type]["total_duration"] += op["duration"]
        
        # Calcular promedios
        for op_type in by_type:
            stats = by_type[op_type]
            stats["success_rate"] = stats["success"] / stats["count"] if stats["count"] > 0 else 0.0
            stats["average_duration"] = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0.0
        
        total_success = sum(1 for op in operations if op["success"])
        total_duration = sum(op["duration"] for op in operations)
        
        return {
            "total": len(operations),
            "by_type": dict(by_type),
            "success_rate": total_success / len(operations) if operations else 0.0,
            "average_duration": total_duration / len(operations) if operations else 0.0
        }
    
    def export_report(self, filepath: str, report: PerformanceReport) -> None:
        """
        Exportar reporte a archivo.
        
        Args:
            filepath: Ruta del archivo
            report: Reporte a exportar
        """
        try:
            report_dict = {
                "period_start": report.period_start,
                "period_end": report.period_end,
                "total_operations": report.total_operations,
                "successful_operations": report.successful_operations,
                "failed_operations": report.failed_operations,
                "average_response_time": report.average_response_time,
                "p50_response_time": report.p50_response_time,
                "p95_response_time": report.p95_response_time,
                "p99_response_time": report.p99_response_time,
                "throughput": report.throughput,
                "error_rate": report.error_rate,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "metadata": m.metadata
                    }
                    for m in report.metrics
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"Report exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")


# Instancia global
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Obtener instancia global del motor de análisis."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine






