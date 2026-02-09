"""
Predictive Analysis - Sistema de análisis predictivo
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class PredictiveAnalyzer:
    """Analizador predictivo"""

    def __init__(self):
        """Inicializar analizador predictivo"""
        self.historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_data_point(
        self,
        metric: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Registrar punto de datos.

        Args:
            metric: Nombre de la métrica
            value: Valor
            timestamp: Timestamp (opcional)
        """
        if not timestamp:
            timestamp = datetime.utcnow()
        
        self.historical_data[metric].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Mantener solo últimos 1000 puntos
        if len(self.historical_data[metric]) > 1000:
            self.historical_data[metric] = self.historical_data[metric][-1000:]

    def predict_trend(
        self,
        metric: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predecir tendencia.

        Args:
            metric: Nombre de la métrica
            days_ahead: Días a predecir

        Returns:
            Predicción de tendencia
        """
        data_points = self.historical_data.get(metric, [])
        
        if len(data_points) < 2:
            return {
                "metric": metric,
                "prediction": "insufficient_data",
                "confidence": 0.0
            }
        
        values = [dp["value"] for dp in data_points]
        
        # Cálculo simple de tendencia (regresión lineal básica)
        n = len(values)
        x = list(range(n))
        
        # Calcular pendiente
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Predicción
        future_x = n + days_ahead
        predicted_value = y_mean + slope * (future_x - x_mean)
        
        # Calcular confianza (basado en variabilidad)
        if len(values) > 1:
            std_dev = statistics.stdev(values)
            confidence = max(0.0, min(1.0, 1.0 - (std_dev / y_mean) if y_mean != 0 else 0.5))
        else:
            confidence = 0.5
        
        trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        
        return {
            "metric": metric,
            "current_value": values[-1],
            "predicted_value": predicted_value,
            "trend": trend,
            "slope": slope,
            "confidence": confidence,
            "days_ahead": days_ahead
        }

    def predict_usage(
        self,
        user_id: Optional[str] = None,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predecir uso futuro.

        Args:
            user_id: ID del usuario (opcional)
            days_ahead: Días a predecir

        Returns:
            Predicción de uso
        """
        metric = f"usage_{user_id}" if user_id else "usage_global"
        return self.predict_trend(metric, days_ahead)

    def detect_anomalies(
        self,
        metric: str,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detectar anomalías.

        Args:
            metric: Nombre de la métrica
            threshold_std: Umbral en desviaciones estándar

        Returns:
            Lista de anomalías detectadas
        """
        data_points = self.historical_data.get(metric, [])
        
        if len(data_points) < 3:
            return []
        
        values = [dp["value"] for dp in data_points]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        anomalies = []
        
        for i, dp in enumerate(data_points):
            z_score = abs((dp["value"] - mean) / std_dev) if std_dev > 0 else 0
            
            if z_score > threshold_std:
                anomalies.append({
                    "timestamp": dp["timestamp"].isoformat() if isinstance(dp["timestamp"], datetime) else dp["timestamp"],
                    "value": dp["value"],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium"
                })
        
        return anomalies

    def forecast_demand(
        self,
        operation_type: str,
        days_ahead: int = 30
    ) -> Dict[str, Any]:
        """
        Pronosticar demanda.

        Args:
            operation_type: Tipo de operación
            days_ahead: Días a pronosticar

        Returns:
            Pronóstico de demanda
        """
        metric = f"demand_{operation_type}"
        prediction = self.predict_trend(metric, days_ahead)
        
        return {
            "operation_type": operation_type,
            "forecast": prediction,
            "recommendations": self._generate_demand_recommendations(prediction)
        }

    def _generate_demand_recommendations(
        self,
        prediction: Dict[str, Any]
    ) -> List[str]:
        """
        Generar recomendaciones basadas en predicción.

        Args:
            prediction: Predicción

        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if prediction.get("trend") == "increasing":
            recommendations.append("La demanda está aumentando. Considera escalar recursos.")
        elif prediction.get("trend") == "decreasing":
            recommendations.append("La demanda está disminuyendo. Puedes optimizar recursos.")
        
        if prediction.get("confidence", 0) < 0.5:
            recommendations.append("Baja confianza en la predicción. Recolecta más datos.")
        
        return recommendations






