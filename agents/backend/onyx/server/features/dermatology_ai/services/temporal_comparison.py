"""
Sistema de análisis de progreso con comparación temporal
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class TemporalComparison:
    """Comparación temporal"""
    metric_name: str
    baseline_value: float
    current_value: float
    change: float
    change_percentage: float
    trend: str  # "improving", "declining", "stable"
    significance: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change": self.change,
            "change_percentage": self.change_percentage,
            "trend": self.trend,
            "significance": self.significance
        }


@dataclass
class TemporalComparisonReport:
    """Reporte de comparación temporal"""
    id: str
    user_id: str
    baseline_date: str
    current_date: str
    time_period_days: int
    comparisons: List[TemporalComparison]
    overall_trend: str
    key_improvements: List[str]
    areas_of_concern: List[str]
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
            "baseline_date": self.baseline_date,
            "current_date": self.current_date,
            "time_period_days": self.time_period_days,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_trend": self.overall_trend,
            "key_improvements": self.key_improvements,
            "areas_of_concern": self.areas_of_concern,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class TemporalComparison:
    """Sistema de comparación temporal"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[TemporalComparisonReport]] = {}  # user_id -> [reports]
        self.historical_data: Dict[str, List[Dict]] = {}  # user_id -> [data_points]
    
    def add_data_point(self, user_id: str, timestamp: str, metrics: Dict):
        """Agrega punto de datos"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
        self.historical_data[user_id].sort(key=lambda x: x["timestamp"])
    
    def generate_comparison(self, user_id: str, baseline_date: str,
                           current_date: str) -> TemporalComparisonReport:
        """Genera comparación temporal"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 2:
            return TemporalComparisonReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                baseline_date=baseline_date,
                current_date=current_date,
                time_period_days=0,
                comparisons=[],
                overall_trend="insufficient_data",
                key_improvements=[],
                areas_of_concern=["No hay suficientes datos para comparación"],
                recommendations=["Agrega más puntos de datos"]
            )
        
        # Encontrar puntos de datos más cercanos
        baseline_data = self._find_closest_data_point(data_points, baseline_date)
        current_data = self._find_closest_data_point(data_points, current_date)
        
        if not baseline_data or not current_data:
            return TemporalComparisonReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                baseline_date=baseline_date,
                current_date=current_date,
                time_period_days=0,
                comparisons=[],
                overall_trend="data_not_found",
                key_improvements=[],
                areas_of_concern=["Datos no encontrados para las fechas especificadas"],
                recommendations=["Verifica las fechas de los datos"]
            )
        
        baseline_metrics = baseline_data["metrics"]
        current_metrics = current_data["metrics"]
        
        # Calcular período en días
        baseline_dt = datetime.fromisoformat(baseline_date)
        current_dt = datetime.fromisoformat(current_date)
        time_period_days = (current_dt - baseline_dt).days
        
        # Comparar métricas
        comparisons = []
        improvements = []
        concerns = []
        
        for metric_name in set(baseline_metrics.keys()) | set(current_metrics.keys()):
            baseline_value = baseline_metrics.get(metric_name, 0)
            current_value = current_metrics.get(metric_name, 0)
            
            change = current_value - baseline_value
            change_percentage = (change / baseline_value * 100) if baseline_value > 0 else 0
            
            # Determinar tendencia
            if change_percentage > 5:
                trend = "improving"
                improvements.append(f"{metric_name} mejoró {change_percentage:.1f}%")
            elif change_percentage < -5:
                trend = "declining"
                concerns.append(f"{metric_name} empeoró {abs(change_percentage):.1f}%")
            else:
                trend = "stable"
            
            # Determinar significancia
            if abs(change_percentage) > 15:
                significance = "high"
            elif abs(change_percentage) > 5:
                significance = "medium"
            else:
                significance = "low"
            
            comparison = TemporalComparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                change=change,
                change_percentage=change_percentage,
                trend=trend,
                significance=significance
            )
            comparisons.append(comparison)
        
        # Tendencia general
        improving_count = sum(1 for c in comparisons if c.trend == "improving")
        declining_count = sum(1 for c in comparisons if c.trend == "declining")
        
        if improving_count > declining_count:
            overall_trend = "improving"
        elif declining_count > improving_count:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        # Recomendaciones
        recommendations = []
        
        if overall_trend == "improving":
            recommendations.append("¡Excelente progreso! Continúa con tu rutina actual.")
        elif overall_trend == "declining":
            recommendations.append("Revisa tu rutina actual y considera ajustes.")
            recommendations.append("Consulta con un dermatólogo si el declive continúa.")
        else:
            recommendations.append("Tu piel se mantiene estable. Considera intensificar tu rutina.")
        
        if concerns:
            recommendations.append(f"Enfócate en mejorar: {', '.join(concerns[:3])}")
        
        report = TemporalComparisonReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            baseline_date=baseline_date,
            current_date=current_date,
            time_period_days=time_period_days,
            comparisons=comparisons,
            overall_trend=overall_trend,
            key_improvements=improvements,
            areas_of_concern=concerns,
            recommendations=recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _find_closest_data_point(self, data_points: List[Dict], target_date: str) -> Optional[Dict]:
        """Encuentra el punto de datos más cercano a una fecha"""
        if not data_points:
            return None
        
        target_dt = datetime.fromisoformat(target_date)
        closest = min(
            data_points,
            key=lambda x: abs((datetime.fromisoformat(x["timestamp"]) - target_dt).total_seconds())
        )
        return closest
    
    def get_user_reports(self, user_id: str) -> List[TemporalComparisonReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






