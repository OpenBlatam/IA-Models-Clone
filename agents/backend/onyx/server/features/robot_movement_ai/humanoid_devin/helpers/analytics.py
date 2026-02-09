"""
Analytics System for Humanoid Devin Robot (Optimizado)
=======================================================

Sistema de análisis y métricas avanzadas para el robot humanoide.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Error in analytics system")
class AnalyticsError(Exception):
    """Excepción para errores de analytics."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in analytics system")
        super().__init__(message)
        self.message = message


class AnalyticsSystem:
    """
    Sistema de análisis y métricas avanzadas.
    
    Recopila, analiza y reporta métricas de rendimiento del robot.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        enable_trend_analysis: bool = True
    ):
        """
        Inicializar sistema de analytics.
        
        Args:
            window_size: Tamaño de ventana para análisis
            enable_trend_analysis: Habilitar análisis de tendencias
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        
        self.window_size = window_size
        self.enable_trend_analysis = enable_trend_analysis
        
        # Métricas por tipo de acción
        self.action_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Métricas agregadas
        self.aggregated_metrics: Dict[str, Any] = {}
        
        # Tendencias
        self.trends: Dict[str, Dict[str, float]] = {}
        
        # Comparaciones temporales
        self.historical_data: deque = deque(maxlen=100)
        
        # Estadísticas
        self.total_actions = 0
        self.start_time = datetime.now()
        
        logger.info(
            f"Analytics system initialized: "
            f"window_size={window_size}, trend_analysis={enable_trend_analysis}"
        )
    
    def record_metric(
        self,
        action_type: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar métrica.
        
        Args:
            action_type: Tipo de acción
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            metadata: Metadatos adicionales (opcional)
        """
        if not action_type or not isinstance(action_type, str):
            raise ValueError("action_type must be a non-empty string")
        if not metric_name or not isinstance(metric_name, str):
            raise ValueError("metric_name must be a non-empty string")
        if not np.isfinite(value):
            raise ValueError("value must be a finite number")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "metric_name": metric_name,
            "value": float(value),
            "metadata": metadata or {}
        }
        
        key = f"{action_type}:{metric_name}"
        self.action_metrics[key].append(record)
        self.total_actions += 1
        
        # Actualizar métricas agregadas
        self._update_aggregated_metrics(key, value)
        
        # Actualizar tendencias si está habilitado
        if self.enable_trend_analysis:
            self._update_trends(key, value)
    
    def _update_aggregated_metrics(self, key: str, value: float) -> None:
        """Actualizar métricas agregadas."""
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "mean": 0.0,
                "std": 0.0
            }
        
        metrics = self.aggregated_metrics[key]
        metrics["count"] += 1
        metrics["sum"] += value
        metrics["min"] = min(metrics["min"], value)
        metrics["max"] = max(metrics["max"], value)
        metrics["mean"] = metrics["sum"] / metrics["count"]
        
        # Calcular desviación estándar
        values = [r["value"] for r in self.action_metrics[key]]
        if len(values) > 1:
            metrics["std"] = float(np.std(values))
    
    def _update_trends(self, key: str, value: float) -> None:
        """Actualizar análisis de tendencias."""
        if key not in self.trends:
            self.trends[key] = {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "trend_direction": "stable"
            }
        
        values = [r["value"] for r in self.action_metrics[key]]
        if len(values) < 10:  # Necesitamos suficientes puntos
            return
        
        # Regresión lineal simple
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calcular pendiente e intercepto
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calcular R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Determinar dirección de tendencia
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        self.trends[key] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "trend_direction": direction
        }
    
    def get_metrics(
        self,
        action_type: Optional[str] = None,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener métricas.
        
        Args:
            action_type: Filtrar por tipo de acción (opcional)
            metric_name: Filtrar por nombre de métrica (opcional)
            
        Returns:
            Dict con métricas
        """
        if action_type and metric_name:
            key = f"{action_type}:{metric_name}"
            if key in self.aggregated_metrics:
                return {
                    key: self.aggregated_metrics[key].copy()
                }
            return {}
        
        # Filtrar según criterios
        filtered = {}
        for key, metrics in self.aggregated_metrics.items():
            parts = key.split(":")
            if len(parts) == 2:
                act_type, met_name = parts
                if action_type and act_type != action_type:
                    continue
                if metric_name and met_name != metric_name:
                    continue
                filtered[key] = metrics.copy()
        
        return filtered
    
    def get_trends(
        self,
        action_type: Optional[str] = None,
        metric_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Obtener tendencias.
        
        Args:
            action_type: Filtrar por tipo de acción (opcional)
            metric_name: Filtrar por nombre de métrica (opcional)
            
        Returns:
            Dict con tendencias
        """
        filtered = {}
        for key, trend in self.trends.items():
            parts = key.split(":")
            if len(parts) == 2:
                act_type, met_name = parts
                if action_type and act_type != action_type:
                    continue
                if metric_name and met_name != metric_name:
                    continue
                filtered[key] = trend.copy()
        
        return filtered
    
    def compare_periods(
        self,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime
    ) -> Dict[str, Any]:
        """
        Comparar métricas entre dos períodos.
        
        Args:
            period1_start: Inicio del período 1
            period1_end: Fin del período 1
            period2_start: Inicio del período 2
            period2_end: Fin del período 2
            
        Returns:
            Dict con comparación
        """
        def get_metrics_in_period(start, end):
            metrics = {}
            for key, records in self.action_metrics.items():
                period_values = [
                    r["value"] for r in records
                    if start <= datetime.fromisoformat(r["timestamp"]) <= end
                ]
                if period_values:
                    metrics[key] = {
                        "mean": float(np.mean(period_values)),
                        "std": float(np.std(period_values)),
                        "count": len(period_values)
                    }
            return metrics
        
        period1_metrics = get_metrics_in_period(period1_start, period1_end)
        period2_metrics = get_metrics_in_period(period2_start, period2_end)
        
        comparison = {}
        all_keys = set(period1_metrics.keys()) | set(period2_metrics.keys())
        
        for key in all_keys:
            p1 = period1_metrics.get(key, {})
            p2 = period2_metrics.get(key, {})
            
            p1_mean = p1.get("mean", 0.0)
            p2_mean = p2.get("mean", 0.0)
            
            change = p2_mean - p1_mean
            change_percent = (change / p1_mean * 100) if p1_mean != 0 else 0.0
            
            comparison[key] = {
                "period1": p1,
                "period2": p2,
                "change": float(change),
                "change_percent": float(change_percent),
                "improvement": change < 0 if "time" in key or "energy" in key else change > 0
            }
        
        return comparison
    
    def generate_report(
        self,
        report_type: str = "summary",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generar reporte de analytics.
        
        Args:
            report_type: Tipo de reporte (summary, detailed, trends)
            output_file: Archivo de salida (opcional)
            
        Returns:
            Dict con reporte
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "total_actions": self.total_actions,
            "actions_per_second": self.total_actions / uptime if uptime > 0 else 0.0
        }
        
        if report_type in ["summary", "detailed"]:
            report["aggregated_metrics"] = {
                k: v.copy() for k, v in self.aggregated_metrics.items()
            }
        
        if report_type in ["detailed", "trends"]:
            report["trends"] = {
                k: v.copy() for k, v in self.trends.items()
            }
        
        if report_type == "detailed":
            # Incluir muestras recientes
            report["recent_samples"] = {}
            for key, records in self.action_metrics.items():
                if records:
                    report["recent_samples"][key] = list(records)[-10:]
        
        # Guardar reporte si se especifica archivo
        if output_file:
            try:
                report_file = Path(output_file)
                report_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Analytics report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}", exc_info=True)
                raise AnalyticsError(f"Failed to save report: {str(e)}") from e
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de analytics.
        
        Returns:
            Dict con estadísticas
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_actions": self.total_actions,
            "actions_per_second": self.total_actions / uptime if uptime > 0 else 0.0,
            "unique_metrics": len(self.aggregated_metrics),
            "trends_tracked": len(self.trends),
            "window_size": self.window_size,
            "trend_analysis_enabled": self.enable_trend_analysis
        }
    
    def reset_analytics(self) -> None:
        """Resetear todos los analytics."""
        self.action_metrics.clear()
        self.aggregated_metrics.clear()
        self.trends.clear()
        self.historical_data.clear()
        self.total_actions = 0
        self.start_time = datetime.now()
        logger.info("Analytics reset")

