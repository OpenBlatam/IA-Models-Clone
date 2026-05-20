"""
Sistema de análisis de progreso con predicción de resultados futuros
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class FuturePrediction:
    """Predicción futura"""
    metric_name: str
    current_value: float
    predicted_value_30_days: float
    predicted_value_60_days: float
    predicted_value_90_days: float
    confidence: float  # 0-1
    trend: str  # "improving", "declining", "stable"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "predicted_value_30_days": self.predicted_value_30_days,
            "predicted_value_60_days": self.predicted_value_60_days,
            "predicted_value_90_days": self.predicted_value_90_days,
            "confidence": self.confidence,
            "trend": self.trend
        }


@dataclass
class FuturePredictionReport:
    """Reporte de predicción futura"""
    id: str
    user_id: str
    predictions: List[FuturePrediction]
    overall_outlook: str
    key_milestones: List[Dict]
    action_items: List[str]
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
            "overall_outlook": self.overall_outlook,
            "key_milestones": self.key_milestones,
            "action_items": self.action_items,
            "created_at": self.created_at
        }


class FuturePrediction:
    """Sistema de predicción de resultados futuros"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[FuturePredictionReport]] = {}  # user_id -> [reports]
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
    
    def generate_future_prediction(self, user_id: str, days: int = 90) -> FuturePredictionReport:
        """Genera predicción futura"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 3:
            return FuturePredictionReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                predictions=[],
                overall_outlook="insufficient_data",
                key_milestones=[],
                action_items=["Necesitas más datos para predicciones precisas"]
            )
        
        # Filtrar datos recientes
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            d for d in data_points
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        
        if len(recent_data) < 3:
            return FuturePredictionReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                predictions=[],
                overall_outlook="insufficient_data",
                key_milestones=[],
                action_items=["Necesitas más datos recientes"]
            )
        
        predictions = []
        
        # Predecir para cada métrica
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            
            if not values or len(values) < 2:
                continue
            
            current_value = values[-1]
            
            # Calcular tendencia (regresión lineal simple)
            n = len(values)
            x = list(range(n))
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(values)
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Predicciones
            days_30_idx = n + 30
            days_60_idx = n + 60
            days_90_idx = n + 90
            
            predicted_30 = intercept + slope * days_30_idx
            predicted_60 = intercept + slope * days_60_idx
            predicted_90 = intercept + slope * days_90_idx
            
            # Determinar tendencia
            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            # Confianza basada en consistencia
            variance = statistics.variance(values) if len(values) > 1 else 0
            confidence = max(0.5, min(0.95, 1.0 - (variance / 100)))
            
            prediction = FuturePrediction(
                metric_name=metric_name,
                current_value=current_value,
                predicted_value_30_days=predicted_30,
                predicted_value_60_days=predicted_60,
                predicted_value_90_days=predicted_90,
                confidence=confidence,
                trend=trend
            )
            predictions.append(prediction)
        
        # Outlook general
        improving = sum(1 for p in predictions if p.trend == "improving")
        declining = sum(1 for p in predictions if p.trend == "declining")
        
        if improving > declining:
            overall_outlook = "positive"
        elif declining > improving:
            overall_outlook = "cautious"
        else:
            overall_outlook = "stable"
        
        # Milestones clave
        milestones = []
        for prediction in predictions:
            if prediction.trend == "improving":
                milestones.append({
                    "metric": prediction.metric_name,
                    "milestone": f"Mejora esperada en 30 días: {prediction.predicted_value_30_days:.1f}",
                    "days": 30
                })
        
        # Items de acción
        action_items = []
        if overall_outlook == "positive":
            action_items.append("Continúa con tu rutina actual")
        elif overall_outlook == "cautious":
            action_items.append("Revisa y ajusta tu rutina")
            action_items.append("Considera consultar con un dermatólogo")
        else:
            action_items.append("Mantén consistencia en tu rutina")
        
        report = FuturePredictionReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            predictions=predictions,
            overall_outlook=overall_outlook,
            key_milestones=milestones,
            action_items=action_items
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def get_user_reports(self, user_id: str) -> List[FuturePredictionReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






