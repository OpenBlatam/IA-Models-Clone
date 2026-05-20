"""
Sistema de predicción de tendencias
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import numpy as np


@dataclass
class TrendPrediction:
    """Predicción de tendencia"""
    metric: str
    current_value: float
    predicted_value: float
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float
    timeframe: str  # "7d", "30d", "90d"
    prediction_date: str = None
    
    def __post_init__(self):
        if self.prediction_date is None:
            self.prediction_date = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric": self.metric,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "trend": self.trend,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "prediction_date": self.prediction_date
        }


class TrendPredictor:
    """Sistema de predicción de tendencias"""
    
    def __init__(self):
        """Inicializa el predictor"""
        pass
    
    def predict_trend(self, historical_data: List[Dict], metric: str,
                     timeframe: str = "30d") -> TrendPrediction:
        """
        Predice tendencia basada en datos históricos
        
        Args:
            historical_data: Datos históricos
            metric: Métrica a predecir
            timeframe: Período de predicción
            
        Returns:
            Predicción de tendencia
        """
        # Extraer valores de la métrica
        values = []
        for data_point in historical_data:
            value = self._extract_metric_value(data_point, metric)
            if value is not None:
                values.append(value)
        
        if len(values) < 2:
            # No hay suficientes datos
            return TrendPrediction(
                metric=metric,
                current_value=values[0] if values else 0,
                predicted_value=values[0] if values else 0,
                trend="stable",
                confidence=0.0,
                timeframe=timeframe
            )
        
        current_value = values[-1]
        
        # Calcular tendencia simple (regresión lineal básica)
        x = np.arange(len(values))
        y = np.array(values)
        
        # Regresión lineal simple
        slope = np.polyfit(x, y, 1)[0]
        
        # Predecir valor futuro
        future_x = len(values) + self._timeframe_to_days(timeframe)
        predicted_value = slope * future_x + np.polyfit(x, y, 1)[1]
        
        # Determinar tendencia
        if slope > 0.1:
            trend = "increasing"
        elif slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calcular confianza (basada en variabilidad)
        std_dev = np.std(values)
        mean_value = np.mean(values)
        confidence = max(0, min(1, 1 - (std_dev / mean_value) if mean_value > 0 else 0))
        
        return TrendPrediction(
            metric=metric,
            current_value=current_value,
            predicted_value=float(predicted_value),
            trend=trend,
            confidence=float(confidence),
            timeframe=timeframe
        )
    
    def predict_multiple_metrics(self, historical_data: List[Dict],
                                metrics: List[str],
                                timeframe: str = "30d") -> List[TrendPrediction]:
        """Predice múltiples métricas"""
        predictions = []
        
        for metric in metrics:
            prediction = self.predict_trend(historical_data, metric, timeframe)
            predictions.append(prediction)
        
        return predictions
    
    def _extract_metric_value(self, data_point: Dict, metric: str) -> Optional[float]:
        """Extrae valor de métrica de un punto de datos"""
        # Buscar en diferentes ubicaciones
        if "quality_scores" in data_point:
            scores = data_point["quality_scores"]
            if metric in scores:
                return scores[metric]
            if metric == "overall_score":
                return scores.get("overall_score", 0)
        
        # Buscar directamente
        if metric in data_point:
            value = data_point[metric]
            if isinstance(value, (int, float)):
                return float(value)
        
        return None
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convierte timeframe a días"""
        if timeframe.endswith("d"):
            return int(timeframe[:-1])
        elif timeframe.endswith("w"):
            return int(timeframe[:-1]) * 7
        elif timeframe.endswith("m"):
            return int(timeframe[:-1]) * 30
        return 30

