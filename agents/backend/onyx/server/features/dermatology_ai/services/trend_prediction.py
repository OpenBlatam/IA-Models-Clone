"""
Sistema de análisis de progreso con predicción de tendencias
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class TrendPrediction:
    """Predicción de tendencia"""
    metric_name: str
    current_value: float
    predicted_value_7d: float
    predicted_value_30d: float
    trend_direction: str  # "improving", "declining", "stable"
    confidence: float  # 0.0 to 1.0
    factors: List[str] = None
    
    def __post_init__(self):
        if self.factors is None:
            self.factors = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "predicted_value_7d": self.predicted_value_7d,
            "predicted_value_30d": self.predicted_value_30d,
            "trend_direction": self.trend_direction,
            "confidence": self.confidence,
            "factors": self.factors
        }


@dataclass
class TrendReport:
    """Reporte de predicción de tendencias"""
    id: str
    user_id: str
    predictions: List[TrendPrediction]
    overall_trend: str
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
            "predictions": [p.to_dict() for p in self.predictions],
            "overall_trend": self.overall_trend,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class TrendPredictionSystem:
    """Sistema de predicción de tendencias"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[TrendReport]] = {}
        self.historical_data: Dict[str, List[Dict]] = {}
    
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
    
    def predict_trends(self, user_id: str, days: int = 60) -> TrendReport:
        """Predice tendencias futuras"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 7:
            return TrendReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            predictions=[],
            overall_trend="insufficient_data",
            recommendations=["Necesitas más datos para predicciones"]
        )
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            d for d in data_points
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        
        if len(recent_data) < 7:
            return TrendReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                predictions=[],
                overall_trend="insufficient_data",
                recommendations=["Necesitas más datos recientes"]
            )
        
        predictions = []
        
        # Analizar cada métrica
        for metric_name in ["overall_score", "hydration_score", "texture_score", "acne_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            timestamps = [d["timestamp"] for d in recent_data]
            
            if len(values) < 7:
                continue
            
            # Calcular tendencia usando regresión lineal simple
            trend_info = self._calculate_trend(values, timestamps, metric_name)
            
            if trend_info:
                predictions.append(trend_info)
        
        # Tendencias generales
        improving_count = sum(1 for p in predictions if p.trend_direction == "improving")
        declining_count = sum(1 for p in predictions if p.trend_direction == "declining")
        
        if improving_count > declining_count:
            overall_trend = "improving"
        elif declining_count > improving_count:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        # Recomendaciones
        recommendations = []
        
        if overall_trend == "improving":
            recommendations.append("Tu piel está mejorando. Continúa con tu rutina actual")
        elif overall_trend == "declining":
            recommendations.append("Tu piel está empeorando. Considera ajustar tu rutina")
        else:
            recommendations.append("Tu piel está estable. Considera optimizar tu rutina")
        
        for pred in predictions:
            if pred.trend_direction == "declining" and pred.confidence > 0.7:
                recommendations.append(f"Monitorea de cerca: {pred.metric_name}")
        
        report = TrendReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            predictions=predictions,
            overall_trend=overall_trend,
            recommendations=recommendations
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _calculate_trend(self, values: List[float], timestamps: List[str], 
                        metric_name: str) -> Optional[TrendPrediction]:
        """Calcula tendencia usando regresión lineal simple"""
        if len(values) < 7:
            return None
        
        # Calcular pendiente (tendencia)
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Pendiente
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Intercepto
        intercept = (sum_y - slope * sum_x) / n
        
        # Valores actuales y predichos
        current_value = values[-1]
        predicted_7d = current_value + slope * 7
        predicted_30d = current_value + slope * 30
        
        # Dirección de tendencia
        if slope > 0.5:
            trend_direction = "improving"
        elif slope < -0.5:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
        
        # Confianza basada en consistencia de datos
        variance = statistics.variance(values) if len(values) > 1 else 0
        confidence = max(0.0, min(1.0, 1.0 - (variance / 100.0)))
        
        # Factores
        factors = []
        if slope > 0:
            factors.append("Tendencia positiva")
        if variance < 10:
            factors.append("Datos consistentes")
        
        return TrendPrediction(
            metric_name=metric_name,
            current_value=current_value,
            predicted_value_7d=predicted_7d,
            predicted_value_30d=predicted_30d,
            trend_direction=trend_direction,
            confidence=confidence,
            factors=factors
        )
    
    def get_user_reports(self, user_id: str) -> List[TrendReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])


