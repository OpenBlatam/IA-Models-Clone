"""
Analizador Predictivo
=====================

Sistema para análisis predictivo y forecasting basado en documentos históricos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from .document_analyzer import DocumentAnalyzer
from .trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Predicción"""
    metric: str
    current_value: float
    predicted_value: float
    confidence: float
    timeframe: str  # "1day", "1week", "1month"
    trend: str  # "increasing", "decreasing", "stable"
    factors: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PredictiveReport:
    """Reporte predictivo"""
    predictions: List[Prediction]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Calcular confidence score promedio
        if self.predictions:
            self.confidence_score = np.mean([p.confidence for p in self.predictions])
        else:
            self.confidence_score = 0.0


class PredictiveAnalyzer:
    """
    Analizador predictivo
    
    Proporciona predicciones basadas en análisis de documentos históricos:
    - Predicción de sentimiento
    - Predicción de temas
    - Predicción de keywords
    - Forecasting de métricas
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador predictivo
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        self.trend_analyzer = TrendAnalyzer(analyzer)
        logger.info("PredictiveAnalyzer inicializado")
    
    async def predict_sentiment(
        self,
        historical_documents: List[Dict[str, Any]],
        timeframe: str = "1week"
    ) -> Prediction:
        """
        Predecir sentimiento futuro
        
        Args:
            historical_documents: Documentos históricos con timestamp
            timeframe: Período de predicción
        
        Returns:
            Prediction de sentimiento
        """
        # Analizar tendencia
        trend = await self.trend_analyzer.analyze_sentiment_trend(
            historical_documents,
            period="day"
        )
        
        # Calcular predicción (regresión lineal simple)
        if len(trend.data_points) < 2:
            return Prediction(
                metric="sentiment",
                current_value=0.0,
                predicted_value=0.0,
                confidence=0.0,
                timeframe=timeframe,
                trend="stable",
                factors=["insufficient_data"]
            )
        
        values = [dp.value for dp in trend.data_points]
        current_value = values[-1]
        
        # Calcular pendiente
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Predicción
        days_ahead = self._timeframe_to_days(timeframe)
        predicted_value = current_value + slope * days_ahead
        
        # Determinar tendencia
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Confianza basada en estabilidad
        variance = np.var(values)
        confidence = max(0.0, min(1.0, 1.0 - variance))
        
        return Prediction(
            metric="sentiment",
            current_value=current_value,
            predicted_value=predicted_value,
            confidence=confidence,
            timeframe=timeframe,
            trend=trend_direction,
            factors=["historical_trend", "linear_regression"]
        )
    
    async def predict_topics(
        self,
        historical_documents: List[Dict[str, Any]],
        timeframe: str = "1week"
    ) -> List[Prediction]:
        """
        Predecir temas futuros
        
        Args:
            historical_documents: Documentos históricos
            timeframe: Período de predicción
        
        Returns:
            Lista de predicciones de temas
        """
        # Analizar tendencia de temas
        topic_trends = await self.trend_analyzer.analyze_topic_trend(
            historical_documents,
            period="day"
        )
        
        predictions = []
        for topic_id, trend in topic_trends.items():
            if len(trend.data_points) < 2:
                continue
            
            values = [dp.value for dp in trend.data_points]
            current_value = values[-1]
            
            # Calcular pendiente
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Predicción
            days_ahead = self._timeframe_to_days(timeframe)
            predicted_value = max(0, current_value + slope * days_ahead)
            
            predictions.append(Prediction(
                metric=f"topic_{topic_id}",
                current_value=current_value,
                predicted_value=predicted_value,
                confidence=trend.trend_strength,
                timeframe=timeframe,
                trend=trend.trend_direction,
                factors=["topic_trend", "frequency_analysis"]
            ))
        
        return predictions
    
    async def generate_predictive_report(
        self,
        historical_documents: List[Dict[str, Any]],
        timeframe: str = "1week"
    ) -> PredictiveReport:
        """
        Generar reporte predictivo completo
        
        Args:
            historical_documents: Documentos históricos
            timeframe: Período de predicción
        
        Returns:
            PredictiveReport con todas las predicciones
        """
        predictions = []
        insights = []
        recommendations = []
        
        # Predicción de sentimiento
        sentiment_pred = await self.predict_sentiment(historical_documents, timeframe)
        predictions.append(sentiment_pred)
        
        if sentiment_pred.trend == "decreasing" and sentiment_pred.confidence > 0.7:
            insights.append("Sentimiento negativo en aumento - se recomienda atención")
            recommendations.append("Revisar estrategia de comunicación")
        
        # Predicción de temas
        topic_predictions = await self.predict_topics(historical_documents, timeframe)
        predictions.extend(topic_predictions)
        
        # Identificar temas emergentes
        emerging_topics = [
            p for p in topic_predictions
            if p.trend == "increasing" and p.predicted_value > p.current_value * 1.2
        ]
        if emerging_topics:
            insights.append(f"{len(emerging_topics)} temas emergentes identificados")
            recommendations.append("Monitorear temas emergentes de cerca")
        
        # Generar insights adicionales
        if sentiment_pred.confidence < 0.5:
            insights.append("Predicciones con baja confianza - necesarios más datos históricos")
            recommendations.append("Recopilar más datos antes de tomar decisiones")
        
        return PredictiveReport(
            predictions=predictions,
            insights=insights,
            recommendations=recommendations,
            confidence_score=0.0  # Se calculará en __post_init__
        )
    
    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convertir timeframe a días"""
        if timeframe == "1day":
            return 1
        elif timeframe == "1week":
            return 7
        elif timeframe == "1month":
            return 30
        elif timeframe == "3months":
            return 90
        else:
            return 7  # Default
















