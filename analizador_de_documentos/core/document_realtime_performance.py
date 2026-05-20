"""
Document Realtime Performance - Análisis de Rendimiento en Tiempo Real
=======================================================================

Sistema de monitoreo y análisis de rendimiento en tiempo real.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import time
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento."""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = "seconds"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot de rendimiento."""
    snapshot_time: datetime
    metrics: Dict[str, float]
    alerts: List[str] = field(default_factory=list)
    health_status: str = "healthy"  # 'healthy', 'degraded', 'critical'


class RealtimePerformanceMonitor:
    """Monitor de rendimiento en tiempo real."""
    
    def __init__(self, analyzer, window_size: int = 100):
        """Inicializar monitor."""
        self.analyzer = analyzer
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = {}
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}
        self.active_alerts: List[str] = []
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "seconds",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar métrica."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.window_size)
        
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            metadata=metadata or {}
        )
        
        self.metrics_history[metric_name].append(metric)
        
        # Verificar umbrales
        self._check_thresholds(metric_name, value)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Verificar umbrales de métricas."""
        if metric_name not in self.performance_thresholds:
            return
        
        thresholds = self.performance_thresholds[metric_name]
        
        if "warning" in thresholds and value > thresholds["warning"]:
            alert = f"WARNING: {metric_name} = {value} (umbral: {thresholds['warning']})"
            if alert not in self.active_alerts:
                self.active_alerts.append(alert)
                logger.warning(alert)
        
        if "critical" in thresholds and value > thresholds["critical"]:
            alert = f"CRITICAL: {metric_name} = {value} (umbral: {thresholds['critical']})"
            if alert not in self.active_alerts:
                self.active_alerts.append(alert)
                logger.error(alert)
    
    def set_threshold(
        self,
        metric_name: str,
        warning: Optional[float] = None,
        critical: Optional[float] = None
    ):
        """Establecer umbrales para una métrica."""
        if metric_name not in self.performance_thresholds:
            self.performance_thresholds[metric_name] = {}
        
        if warning is not None:
            self.performance_thresholds[metric_name]["warning"] = warning
        if critical is not None:
            self.performance_thresholds[metric_name]["critical"] = critical
    
    async def get_performance_snapshot(
        self,
        time_window_seconds: int = 60
    ) -> PerformanceSnapshot:
        """
        Obtener snapshot de rendimiento.
        
        Args:
            time_window_seconds: Ventana de tiempo en segundos
        
        Returns:
            PerformanceSnapshot con métricas actuales
        """
        cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
        
        snapshot_metrics = {}
        alerts = []
        
        for metric_name, metrics_deque in self.metrics_history.items():
            # Filtrar métricas recientes
            recent_metrics = [
                m for m in metrics_deque
                if m.timestamp >= cutoff_time
            ]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                snapshot_metrics[metric_name] = {
                    "current": values[-1] if values else 0,
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Determinar estado de salud
        health_status = "healthy"
        critical_count = sum(1 for alert in self.active_alerts if "CRITICAL" in alert)
        warning_count = sum(1 for alert in self.active_alerts if "WARNING" in alert)
        
        if critical_count > 0:
            health_status = "critical"
        elif warning_count > 2:
            health_status = "degraded"
        
        return PerformanceSnapshot(
            snapshot_time=datetime.now(),
            metrics=snapshot_metrics,
            alerts=self.active_alerts.copy(),
            health_status=health_status
        )
    
    async def monitor_analysis_performance(
        self,
        document_id: str,
        analysis_function: callable
    ) -> Any:
        """
        Monitorear rendimiento de análisis.
        
        Args:
            document_id: ID del documento
            analysis_function: Función de análisis a ejecutar
        
        Returns:
            Resultado del análisis
        """
        start_time = time.time()
        
        try:
            result = await analysis_function()
            
            elapsed_time = time.time() - start_time
            
            self.record_metric(
                "analysis_duration",
                elapsed_time,
                unit="seconds",
                metadata={"document_id": document_id}
            )
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            self.record_metric(
                "analysis_error",
                1,
                unit="count",
                metadata={"document_id": document_id, "error": str(e)}
            )
            
            raise
    
    def get_metric_statistics(
        self,
        metric_name: str,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """Obtener estadísticas de una métrica."""
        if metric_name not in self.metrics_history:
            return {}
        
        metrics = list(self.metrics_history[metric_name])
        
        if time_window_seconds:
            cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def clear_alerts(self):
        """Limpiar alertas activas."""
        self.active_alerts.clear()


__all__ = [
    "RealtimePerformanceMonitor",
    "PerformanceMetric",
    "PerformanceSnapshot"
]


