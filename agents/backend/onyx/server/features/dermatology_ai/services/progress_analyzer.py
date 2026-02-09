"""
Sistema de análisis de progreso
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import numpy as np


@dataclass
class ProgressMetric:
    """Métrica de progreso"""
    metric_name: str
    initial_value: float
    current_value: float
    change: float
    percentage_change: float
    trend: str  # "improving", "declining", "stable"
    timeframe_days: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "initial_value": self.initial_value,
            "current_value": self.current_value,
            "change": self.change,
            "percentage_change": self.percentage_change,
            "trend": self.trend,
            "timeframe_days": self.timeframe_days
        }


class ProgressAnalyzer:
    """Sistema de análisis de progreso"""
    
    def __init__(self):
        """Inicializa el analizador"""
        pass
    
    def analyze_progress(self, historical_data: List[Dict],
                        timeframe_days: int = 30) -> Dict:
        """
        Analiza progreso del usuario
        
        Args:
            historical_data: Datos históricos
            timeframe_days: Período de análisis en días
            
        Returns:
            Diccionario con análisis de progreso
        """
        if len(historical_data) < 2:
            return {"error": "Insufficient data for progress analysis"}
        
        # Filtrar por timeframe
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        filtered_data = [
            d for d in historical_data
            if datetime.fromisoformat(d.get("timestamp", datetime.now().isoformat())) >= cutoff_date
        ]
        
        if len(filtered_data) < 2:
            return {"error": "Insufficient data in timeframe"}
        
        # Analizar métricas principales
        progress_metrics = []
        
        # Overall score
        overall_scores = [
            d.get("quality_scores", {}).get("overall_score", 0)
            for d in filtered_data
        ]
        
        if overall_scores:
            progress_metrics.append(self._create_progress_metric(
                "overall_score",
                overall_scores[0],
                overall_scores[-1],
                timeframe_days
            ))
        
        # Texture score
        texture_scores = [
            d.get("quality_scores", {}).get("texture_score", 0)
            for d in filtered_data
            if d.get("quality_scores", {}).get("texture_score") is not None
        ]
        
        if texture_scores:
            progress_metrics.append(self._create_progress_metric(
                "texture_score",
                texture_scores[0],
                texture_scores[-1],
                timeframe_days
            ))
        
        # Hydration score
        hydration_scores = [
            d.get("quality_scores", {}).get("hydration_score", 0)
            for d in filtered_data
            if d.get("quality_scores", {}).get("hydration_score") is not None
        ]
        
        if hydration_scores:
            progress_metrics.append(self._create_progress_metric(
                "hydration_score",
                hydration_scores[0],
                hydration_scores[-1],
                timeframe_days
            ))
        
        # Resumen general
        summary = self._generate_progress_summary(progress_metrics)
        
        return {
            "timeframe_days": timeframe_days,
            "data_points": len(filtered_data),
            "progress_metrics": [m.to_dict() for m in progress_metrics],
            "summary": summary,
            "recommendations": self._generate_progress_recommendations(progress_metrics)
        }
    
    def _create_progress_metric(self, metric_name: str, initial_value: float,
                                current_value: float, timeframe_days: int) -> ProgressMetric:
        """Crea métrica de progreso"""
        change = current_value - initial_value
        percentage_change = (change / initial_value * 100) if initial_value > 0 else 0
        
        if change > 2:
            trend = "improving"
        elif change < -2:
            trend = "declining"
        else:
            trend = "stable"
        
        return ProgressMetric(
            metric_name=metric_name,
            initial_value=initial_value,
            current_value=current_value,
            change=change,
            percentage_change=percentage_change,
            trend=trend,
            timeframe_days=timeframe_days
        )
    
    def _generate_progress_summary(self, metrics: List[ProgressMetric]) -> Dict:
        """Genera resumen de progreso"""
        if not metrics:
            return {}
        
        improving = sum(1 for m in metrics if m.trend == "improving")
        declining = sum(1 for m in metrics if m.trend == "declining")
        stable = sum(1 for m in metrics if m.trend == "stable")
        
        avg_change = statistics.mean([m.percentage_change for m in metrics])
        
        return {
            "total_metrics": len(metrics),
            "improving_metrics": improving,
            "declining_metrics": declining,
            "stable_metrics": stable,
            "average_change_percentage": avg_change,
            "overall_trend": "improving" if improving > declining else "declining" if declining > improving else "stable"
        }
    
    def _generate_progress_recommendations(self, metrics: List[ProgressMetric]) -> List[str]:
        """Genera recomendaciones basadas en progreso"""
        recommendations = []
        
        for metric in metrics:
            if metric.trend == "declining":
                recommendations.append(
                    f"Tu {metric.metric_name} está empeorando. "
                    f"Considera ajustar tu rutina de cuidado."
                )
            elif metric.trend == "improving":
                recommendations.append(
                    f"¡Excelente! Tu {metric.metric_name} está mejorando. "
                    f"Continúa con tu rutina actual."
                )
        
        return recommendations
    
    def get_progress_timeline(self, historical_data: List[Dict],
                             metric_name: str = "overall_score") -> Dict:
        """
        Obtiene timeline de progreso
        
        Args:
            historical_data: Datos históricos
            metric_name: Nombre de la métrica
            
        Returns:
            Timeline de progreso
        """
        timeline_points = []
        
        for data_point in historical_data:
            timestamp = data_point.get("timestamp", datetime.now().isoformat())
            value = data_point.get("quality_scores", {}).get(metric_name, 0)
            
            timeline_points.append({
                "timestamp": timestamp,
                "value": value
            })
        
        # Ordenar por timestamp
        timeline_points.sort(key=lambda x: x["timestamp"])
        
        return {
            "metric_name": metric_name,
            "timeline": timeline_points,
            "total_points": len(timeline_points),
            "first_value": timeline_points[0]["value"] if timeline_points else None,
            "last_value": timeline_points[-1]["value"] if timeline_points else None
        }






