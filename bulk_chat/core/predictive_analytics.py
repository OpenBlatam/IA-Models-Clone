"""
Predictive Analytics - Análisis Predictivo
==========================================

Sistema de análisis predictivo con machine learning para predecir tendencias y comportamientos futuros.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Tipo de predicción."""
    DEMAND = "demand"
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"
    TREND = "trend"
    ANOMALY = "anomaly"


@dataclass
class Prediction:
    """Predicción."""
    prediction_id: str
    prediction_type: PredictionType
    target: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    horizon_days: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionModel:
    """Modelo de predicción."""
    model_id: str
    model_type: str
    target: str
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    training_data_size: int = 0
    enabled: bool = True


class PredictiveAnalytics:
    """Sistema de análisis predictivo."""
    
    def __init__(self):
        self.predictions: List[Prediction] = []
        self.models: Dict[str, PredictionModel] = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._lock = asyncio.Lock()
    
    def record_data_point(
        self,
        target: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Registrar punto de datos."""
        timestamp = timestamp or datetime.now()
        
        self.historical_data[target].append({
            "value": value,
            "timestamp": timestamp,
        })
        
        # Auto-predecir si hay suficientes datos
        if len(self.historical_data[target]) > 100:
            asyncio.create_task(self._auto_predict(target))
    
    async def _auto_predict(self, target: str):
        """Predicción automática."""
        data = list(self.historical_data[target])
        if len(data) < 50:
            return
        
        # Análisis simple de tendencia
        recent_values = [d["value"] for d in data[-50:]]
        older_values = [d["value"] for d in data[-100:-50]] if len(data) > 50 else recent_values
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        # Calcular tendencia
        trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
        
        # Predicción simple: extrapolar tendencia
        predicted_value = recent_avg * (1 + trend)
        
        # Calcular confianza basada en varianza
        variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - (variance / recent_avg) if recent_avg > 0 else 0.5))
        
        prediction = Prediction(
            prediction_id=f"pred_{target}_{datetime.now().timestamp()}",
            prediction_type=PredictionType.TREND,
            target=target,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=datetime.now(),
            horizon_days=1,
            metadata={
                "trend": trend,
                "recent_avg": recent_avg,
                "variance": variance,
            },
        )
        
        async with self._lock:
            self.predictions.append(prediction)
            if len(self.predictions) > 10000:
                self.predictions.pop(0)
    
    def predict(
        self,
        target: str,
        prediction_type: PredictionType = PredictionType.TREND,
        horizon_days: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Realizar predicción."""
        data = list(self.historical_data.get(target, []))
        if len(data) < 10:
            return None
        
        values = [d["value"] for d in data]
        
        # Predicción simple basada en promedio móvil
        recent_avg = statistics.mean(values[-10:])
        
        # Calcular tendencia
        if len(values) > 20:
            older_avg = statistics.mean(values[-20:-10])
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
        else:
            trend = 0.0
        
        predicted_value = recent_avg * (1 + trend * horizon_days)
        
        # Calcular confianza
        variance = statistics.variance(values[-10:]) if len(values) > 1 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - (variance / recent_avg) if recent_avg > 0 else 0.5))
        
        return {
            "target": target,
            "prediction_type": prediction_type.value,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "current_value": values[-1],
            "trend": trend,
        }
    
    def register_model(
        self,
        model_id: str,
        model_type: str,
        target: str,
        accuracy: float = 0.0,
    ) -> str:
        """Registrar modelo de predicción."""
        model = PredictionModel(
            model_id=model_id,
            model_type=model_type,
            target=target,
            accuracy=accuracy,
        )
        
        async def save_model():
            async with self._lock:
                self.models[model_id] = model
        
        asyncio.create_task(save_model())
        
        logger.info(f"Registered prediction model: {model_id} for {target}")
        return model_id
    
    def get_predictions(
        self,
        target: Optional[str] = None,
        prediction_type: Optional[PredictionType] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener predicciones."""
        predictions = self.predictions
        
        if target:
            predictions = [p for p in predictions if p.target == target]
        
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]
        
        predictions.sort(key=lambda p: p.timestamp, reverse=True)
        
        return [
            {
                "prediction_id": p.prediction_id,
                "prediction_type": p.prediction_type.value,
                "target": p.target,
                "predicted_value": p.predicted_value,
                "confidence": p.confidence,
                "timestamp": p.timestamp.isoformat(),
                "horizon_days": p.horizon_days,
                "metadata": p.metadata,
            }
            for p in predictions[:limit]
        ]
    
    def get_predictive_analytics_summary(self) -> Dict[str, Any]:
        """Obtener resumen del analítico."""
        by_type: Dict[str, int] = defaultdict(int)
        avg_confidence = 0.0
        
        for prediction in self.predictions:
            by_type[prediction.prediction_type.value] += 1
            avg_confidence += prediction.confidence
        
        return {
            "total_predictions": len(self.predictions),
            "predictions_by_type": dict(by_type),
            "total_models": len(self.models),
            "total_data_points": sum(len(data) for data in self.historical_data.values()),
            "avg_confidence": avg_confidence / len(self.predictions) if self.predictions else 0.0,
        }

