"""
Predictive Content Analyzer - Sistema de análisis predictivo de contenido
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ContentPrediction:
    """Predicción de contenido"""
    content_id: str
    predicted_metric: str
    predicted_value: float
    confidence: float
    factors: Dict[str, Any]
    timestamp: datetime


class PredictiveContentAnalyzer:
    """Analizador predictivo de contenido"""

    def __init__(self):
        """Inicializar analizador"""
        self.historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.predictions: List[ContentPrediction] = []

    def record_historical_data(
        self,
        content_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar datos históricos.

        Args:
            content_id: ID del contenido
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            metadata: Metadatos adicionales
        """
        data_point = {
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.historical_data[content_id].append(data_point)
        
        # Limitar tamaño
        if len(self.historical_data[content_id]) > 1000:
            self.historical_data[content_id] = self.historical_data[content_id][-1000:]
        
        logger.debug(f"Dato histórico registrado: {content_id} - {metric_name}")

    def predict_metric(
        self,
        content_id: str,
        metric_name: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Predecir métrica futura.

        Args:
            content_id: ID del contenido
            metric_name: Nombre de la métrica
            days_ahead: Días a predecir

        Returns:
            Predicción
        """
        historical = self.historical_data.get(content_id, [])
        
        # Filtrar por métrica
        metric_data = [
            d for d in historical
            if d.get("metric_name") == metric_name
        ]
        
        if len(metric_data) < 3:
            return {
                "error": "Datos insuficientes para predicción",
                "required": 3,
                "available": len(metric_data)
            }
        
        # Obtener valores recientes
        recent_values = [d["value"] for d in metric_data[-30:]]
        
        # Calcular tendencia simple
        if len(recent_values) >= 2:
            trend = recent_values[-1] - recent_values[0]
            trend_per_day = trend / len(recent_values)
        else:
            trend_per_day = 0
        
        # Calcular promedio reciente
        recent_avg = sum(recent_values) / len(recent_values)
        
        # Predecir valor futuro
        predicted_value = recent_avg + (trend_per_day * days_ahead)
        
        # Calcular confianza basada en cantidad de datos
        confidence = min(1.0, len(recent_values) / 30)
        
        # Factores que influyen
        factors = {
            "historical_data_points": len(recent_values),
            "recent_average": recent_avg,
            "trend_per_day": trend_per_day,
            "volatility": self._calculate_volatility(recent_values)
        }
        
        prediction = ContentPrediction(
            content_id=content_id,
            predicted_metric=metric_name,
            predicted_value=max(0, predicted_value),  # No valores negativos
            confidence=confidence,
            factors=factors,
            timestamp=datetime.utcnow()
        )
        
        self.predictions.append(prediction)
        
        return {
            "content_id": content_id,
            "metric_name": metric_name,
            "days_ahead": days_ahead,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "factors": factors,
            "current_value": recent_values[-1] if recent_values else 0
        }

    def predict_content_performance(
        self,
        content_id: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Predecir performance general del contenido.

        Args:
            content_id: ID del contenido
            metrics: Lista de métricas a predecir

        Returns:
            Predicciones de performance
        """
        predictions = {}
        
        for metric in metrics:
            prediction = self.predict_metric(content_id, metric, days_ahead=7)
            if "error" not in prediction:
                predictions[metric] = prediction
        
        if not predictions:
            return {"error": "No se pudieron generar predicciones"}
        
        # Calcular score de performance general
        avg_confidence = sum(p.get("confidence", 0) for p in predictions.values()) / len(predictions)
        
        return {
            "content_id": content_id,
            "predictions": predictions,
            "overall_confidence": avg_confidence,
            "prediction_date": datetime.utcnow().isoformat()
        }

    def get_prediction_history(
        self,
        content_id: Optional[str] = None,
        metric_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de predicciones.

        Args:
            content_id: ID del contenido (opcional)
            metric_name: Nombre de la métrica (opcional)

        Returns:
            Historial de predicciones
        """
        filtered = self.predictions
        
        if content_id:
            filtered = [p for p in filtered if p.content_id == content_id]
        
        if metric_name:
            filtered = [p for p in filtered if p.predicted_metric == metric_name]
        
        return [
            {
                "content_id": p.content_id,
                "metric_name": p.predicted_metric,
                "predicted_value": p.predicted_value,
                "confidence": p.confidence,
                "factors": p.factors,
                "timestamp": p.timestamp.isoformat()
            }
            for p in filtered
        ]

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calcular volatilidad"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5






