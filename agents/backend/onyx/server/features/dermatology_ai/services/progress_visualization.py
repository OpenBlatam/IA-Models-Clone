"""
Sistema de análisis de progreso con visualización
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class ProgressDataPoint:
    """Punto de datos de progreso"""
    date: str
    metrics: Dict[str, float]
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "date": self.date,
            "metrics": self.metrics,
            "notes": self.notes
        }


@dataclass
class ProgressVisualization:
    """Visualización de progreso"""
    id: str
    user_id: str
    metric_name: str
    data_points: List[ProgressDataPoint]
    trend_line: List[float]
    average_value: float
    improvement_percentage: float
    visualization_type: str  # "line", "bar", "radar"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "metric_name": self.metric_name,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "trend_line": self.trend_line,
            "average_value": self.average_value,
            "improvement_percentage": self.improvement_percentage,
            "visualization_type": self.visualization_type,
            "created_at": self.created_at
        }


@dataclass
class ComprehensiveProgressReport:
    """Reporte completo de progreso"""
    id: str
    user_id: str
    visualizations: List[ProgressVisualization]
    overall_improvement: float
    best_metric: str
    needs_attention: List[str]
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "visualizations": [v.to_dict() for v in self.visualizations],
            "overall_improvement": self.overall_improvement,
            "best_metric": self.best_metric,
            "needs_attention": self.needs_attention,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class ProgressVisualizationSystem:
    """Sistema de visualización de progreso"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[ComprehensiveProgressReport]] = {}
        self.historical_data: Dict[str, List[ProgressDataPoint]] = {}
    
    def add_data_point(self, user_id: str, date: str, metrics: Dict[str, float], notes: Optional[str] = None):
        """Agrega punto de datos"""
        data_point = ProgressDataPoint(
            date=date,
            metrics=metrics,
            notes=notes
        )
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
        self.historical_data[user_id].sort(key=lambda x: x.date)
    
    def generate_progress_report(self, user_id: str, days: int = 90) -> ComprehensiveProgressReport:
        """Genera reporte completo de progreso"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 2:
            return ComprehensiveProgressReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                visualizations=[],
                overall_improvement=0.0,
                best_metric="",
                needs_attention=[],
                recommendations=["Necesitas más datos para generar visualizaciones"]
            )
        
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_data = [
            dp for dp in data_points
            if datetime.fromisoformat(dp.date).date() >= cutoff
        ]
        
        if len(recent_data) < 2:
            return ComprehensiveProgressReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                visualizations=[],
                overall_improvement=0.0,
                best_metric="",
                needs_attention=[],
                recommendations=["Necesitas más datos recientes"]
            )
        
        # Obtener todas las métricas
        all_metrics = set()
        for dp in recent_data:
            all_metrics.update(dp.metrics.keys())
        
        visualizations = []
        metric_improvements = {}
        
        # Crear visualización para cada métrica
        for metric_name in all_metrics:
            metric_values = []
            dates = []
            
            for dp in recent_data:
                if metric_name in dp.metrics:
                    metric_values.append(dp.metrics[metric_name])
                    dates.append(dp.date)
            
            if len(metric_values) < 2:
                continue
            
            # Calcular tendencia
            trend_line = self._calculate_trend_line(metric_values)
            
            # Calcular promedio
            avg_value = statistics.mean(metric_values)
            
            # Calcular mejora porcentual
            first_value = metric_values[0]
            last_value = metric_values[-1]
            improvement = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0.0
            
            metric_improvements[metric_name] = improvement
            
            # Crear puntos de datos para visualización
            viz_data_points = [
                ProgressDataPoint(date=dates[i], metrics={metric_name: metric_values[i]})
                for i in range(len(dates))
            ]
            
            visualization = ProgressVisualization(
                id=str(uuid.uuid4()),
                user_id=user_id,
                metric_name=metric_name,
                data_points=viz_data_points,
                trend_line=trend_line,
                average_value=avg_value,
                improvement_percentage=improvement,
                visualization_type="line"
            )
            
            visualizations.append(visualization)
        
        # Calcular mejora general
        if metric_improvements:
            overall_improvement = statistics.mean(metric_improvements.values())
            best_metric = max(metric_improvements.items(), key=lambda x: x[1])[0]
        else:
            overall_improvement = 0.0
            best_metric = ""
        
        # Identificar métricas que necesitan atención
        needs_attention = [
            metric for metric, improvement in metric_improvements.items()
            if improvement < -5.0  # Empeoramiento significativo
        ]
        
        # Recomendaciones
        recommendations = []
        
        if overall_improvement > 10:
            recommendations.append("Excelente progreso. Continúa con tu rutina actual")
        elif overall_improvement > 0:
            recommendations.append("Progreso positivo. Mantén la consistencia")
        elif overall_improvement < -5:
            recommendations.append("Progreso negativo. Considera ajustar tu rutina")
        
        if needs_attention:
            recommendations.append(f"Métricas que necesitan atención: {', '.join(needs_attention)}")
        
        if best_metric:
            recommendations.append(f"Mejor mejora en: {best_metric}")
        
        report = ComprehensiveProgressReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            visualizations=visualizations,
            overall_improvement=overall_improvement,
            best_metric=best_metric,
            needs_attention=needs_attention,
            recommendations=recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        self.reports[user_id].append(report)
        
        return report
    
    def _calculate_trend_line(self, values: List[float]) -> List[float]:
        """Calcula línea de tendencia usando regresión lineal simple"""
        if len(values) < 2:
            return values
        
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Pendiente e intercepto
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return values
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Generar línea de tendencia
        trend_line = [slope * i + intercept for i in range(n)]
        return trend_line


