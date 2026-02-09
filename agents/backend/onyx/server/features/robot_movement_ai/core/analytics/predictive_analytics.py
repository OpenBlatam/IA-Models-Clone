"""
Predictive Analytics System
===========================

Sistema de análisis predictivo para predecir tendencias y comportamientos.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Predicción."""
    prediction_id: str
    metric_name: str
    predicted_value: float
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    horizon: float = 1.0  # Horizonte de predicción en unidades de tiempo
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Forecast:
    """Pronóstico."""
    forecast_id: str
    metric_name: str
    predictions: List[float]
    confidence_intervals: List[tuple]
    timestamps: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveAnalyticsSystem:
    """
    Sistema de análisis predictivo.
    
    Predice valores futuros basándose en datos históricos.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        min_samples: int = 50
    ):
        """
        Inicializar sistema de análisis predictivo.
        
        Args:
            window_size: Tamaño de ventana para datos históricos
            min_samples: Mínimo de muestras para predicción
        """
        self.window_size = window_size
        self.min_samples = min_samples
        
        self.historical_data: Dict[str, deque] = {}
        self.predictions: List[Prediction] = []
        self.forecasts: List[Forecast] = []
        self.max_predictions = 10000
        self.max_forecasts = 1000
    
    def add_data_point(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Agregar punto de datos.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor
            timestamp: Timestamp (opcional)
        """
        if metric_name not in self.historical_data:
            self.historical_data[metric_name] = deque(maxlen=self.window_size)
        
        data_point = {
            "value": value,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        self.historical_data[metric_name].append(data_point)
    
    def predict(
        self,
        metric_name: str,
        horizon: float = 1.0,
        method: str = "linear"
    ) -> Prediction:
        """
        Predecir valor futuro.
        
        Args:
            metric_name: Nombre de la métrica
            horizon: Horizonte de predicción
            method: Método de predicción (linear, exponential, moving_average)
            
        Returns:
            Predicción
        """
        if metric_name not in self.historical_data:
            raise ValueError(f"No data available for metric: {metric_name}")
        
        data = list(self.historical_data[metric_name])
        if len(data) < self.min_samples:
            raise ValueError(f"Insufficient data for prediction: {len(data)} < {self.min_samples}")
        
        values = [d["value"] for d in data]
        
        # Predecir según método
        if method == "linear":
            predicted_value, confidence = self._predict_linear(values, horizon)
        elif method == "exponential":
            predicted_value, confidence = self._predict_exponential(values, horizon)
        elif method == "moving_average":
            predicted_value, confidence = self._predict_moving_average(values, horizon)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        import uuid
        prediction_id = str(uuid.uuid4())
        
        prediction = Prediction(
            prediction_id=prediction_id,
            metric_name=metric_name,
            predicted_value=predicted_value,
            confidence=confidence,
            horizon=horizon,
            metadata={"method": method}
        )
        
        self.predictions.append(prediction)
        if len(self.predictions) > self.max_predictions:
            self.predictions = self.predictions[-self.max_predictions:]
        
        return prediction
    
    def _predict_linear(
        self,
        values: List[float],
        horizon: float
    ) -> tuple:
        """Predecir usando regresión lineal."""
        x = np.arange(len(values))
        y = np.array(values)
        
        # Ajuste lineal
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Predecir
        future_x = len(values) + horizon
        predicted = slope * future_x + intercept
        
        # Calcular confianza basada en R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        confidence = max(0.0, min(1.0, r_squared))
        
        return predicted, confidence
    
    def _predict_exponential(
        self,
        values: List[float],
        horizon: float
    ) -> tuple:
        """Predecir usando suavizado exponencial."""
        if len(values) < 2:
            return values[-1] if values else 0.0, 0.0
        
        # Suavizado exponencial simple
        alpha = 0.3
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        # Extrapolar
        trend = smoothed[-1] - smoothed[-2] if len(smoothed) > 1 else 0.0
        predicted = smoothed[-1] + trend * horizon
        
        # Calcular confianza basada en variabilidad
        variance = np.var(values)
        mean_value = np.mean(values)
        if mean_value != 0:
            cv = np.sqrt(variance) / abs(mean_value)
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5
        
        return predicted, confidence
    
    def _predict_moving_average(
        self,
        values: List[float],
        horizon: float
    ) -> tuple:
        """Predecir usando media móvil."""
        window = min(10, len(values) // 2)
        if window < 1:
            window = 1
        
        # Media móvil
        ma = np.convolve(values, np.ones(window)/window, mode='valid')
        
        # Extrapolar usando tendencia
        if len(ma) > 1:
            trend = ma[-1] - ma[-2]
            predicted = ma[-1] + trend * horizon
        else:
            predicted = ma[-1] if len(ma) > 0 else values[-1]
        
        # Calcular confianza
        recent_values = values[-window:]
        std = np.std(recent_values)
        mean_value = np.mean(recent_values)
        
        if mean_value != 0:
            cv = std / abs(mean_value)
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5
        
        return predicted, confidence
    
    def forecast(
        self,
        metric_name: str,
        steps: int = 10,
        method: str = "linear"
    ) -> Forecast:
        """
        Generar pronóstico.
        
        Args:
            metric_name: Nombre de la métrica
            steps: Número de pasos a predecir
            method: Método de predicción
            
        Returns:
            Pronóstico
        """
        if metric_name not in self.historical_data:
            raise ValueError(f"No data available for metric: {metric_name}")
        
        data = list(self.historical_data[metric_name])
        if len(data) < self.min_samples:
            raise ValueError(f"Insufficient data for forecast: {len(data)} < {self.min_samples}")
        
        values = [d["value"] for d in data]
        predictions = []
        confidence_intervals = []
        timestamps = []
        
        last_timestamp = data[-1]["timestamp"]
        base_time = datetime.fromisoformat(last_timestamp)
        
        for step in range(1, steps + 1):
            if method == "linear":
                pred, conf = self._predict_linear(values, step)
            elif method == "exponential":
                pred, conf = self._predict_exponential(values, step)
            elif method == "moving_average":
                pred, conf = self._predict_moving_average(values, step)
            else:
                raise ValueError(f"Unknown forecast method: {method}")
            
            predictions.append(pred)
            
            # Intervalo de confianza (simplificado)
            std = np.std(values) if len(values) > 1 else 0.0
            lower = pred - 1.96 * std * (1 - conf)
            upper = pred + 1.96 * std * (1 - conf)
            confidence_intervals.append((lower, upper))
            
            # Timestamp
            step_time = base_time.replace(second=base_time.second + step)
            timestamps.append(step_time.isoformat())
        
        import uuid
        forecast_id = str(uuid.uuid4())
        
        forecast = Forecast(
            forecast_id=forecast_id,
            metric_name=metric_name,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            timestamps=timestamps,
            metadata={"method": method, "steps": steps}
        )
        
        self.forecasts.append(forecast)
        if len(self.forecasts) > self.max_forecasts:
            self.forecasts = self.forecasts[-self.max_forecasts:]
        
        return forecast
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema."""
        return {
            "tracked_metrics": len(self.historical_data),
            "total_predictions": len(self.predictions),
            "total_forecasts": len(self.forecasts),
            "data_points": sum(len(d) for d in self.historical_data.values())
        }


# Instancia global
_predictive_analytics_system: Optional[PredictiveAnalyticsSystem] = None


def get_predictive_analytics_system(
    window_size: int = 1000,
    min_samples: int = 50
) -> PredictiveAnalyticsSystem:
    """Obtener instancia global del sistema de análisis predictivo."""
    global _predictive_analytics_system
    if _predictive_analytics_system is None:
        _predictive_analytics_system = PredictiveAnalyticsSystem(
            window_size=window_size,
            min_samples=min_samples
        )
    return _predictive_analytics_system


