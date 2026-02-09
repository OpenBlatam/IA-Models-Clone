"""
Análisis Predictivo para Validación Psicológica AI
===================================================
Predicciones y tendencias basadas en datos históricos
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from uuid import UUID
from enum import Enum
import structlog
from collections import defaultdict

from .models import PsychologicalProfile, ValidationReport

logger = structlog.get_logger()


class TrendDirection(str, Enum):
    """Dirección de tendencia"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class Prediction:
    """Representa una predicción"""
    
    def __init__(
        self,
        metric: str,
        current_value: float,
        predicted_value: float,
        confidence: float,
        trend: TrendDirection,
        timeframe_days: int,
        factors: List[str]
    ):
        self.metric = metric
        self.current_value = current_value
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.trend = trend
        self.timeframe_days = timeframe_days
        self.factors = factors
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "metric": self.metric,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "confidence": self.confidence,
            "trend": self.trend.value,
            "timeframe_days": self.timeframe_days,
            "factors": self.factors,
            "created_at": self.created_at.isoformat(),
            "change": self.predicted_value - self.current_value,
            "change_percent": (
                (self.predicted_value - self.current_value) / self.current_value * 100
                if self.current_value != 0 else 0
            )
        }


class PredictiveAnalyzer:
    """Analizador predictivo"""
    
    def __init__(self):
        """Inicializar analizador"""
        logger.info("PredictiveAnalyzer initialized")
    
    def analyze_trends(
        self,
        historical_profiles: List[PsychologicalProfile],
        timeframe_days: int = 30
    ) -> Dict[str, Prediction]:
        """
        Analizar tendencias y generar predicciones
        
        Args:
            historical_profiles: Perfiles históricos ordenados por fecha
            timeframe_days: Período de predicción en días
            
        Returns:
            Diccionario de predicciones por métrica
        """
        if len(historical_profiles) < 2:
            logger.warning("Insufficient data for trend analysis", count=len(historical_profiles))
            return {}
        
        predictions = {}
        
        # Analizar rasgos de personalidad
        trait_predictions = self._predict_personality_traits(historical_profiles, timeframe_days)
        predictions.update(trait_predictions)
        
        # Analizar estado emocional
        emotional_predictions = self._predict_emotional_state(historical_profiles, timeframe_days)
        predictions.update(emotional_predictions)
        
        # Analizar confianza
        confidence_prediction = self._predict_confidence(historical_profiles, timeframe_days)
        if confidence_prediction:
            predictions["confidence_score"] = confidence_prediction
        
        logger.info(
            "Trend analysis completed",
            predictions_count=len(predictions),
            timeframe_days=timeframe_days
        )
        
        return predictions
    
    def _predict_personality_traits(
        self,
        profiles: List[PsychologicalProfile],
        timeframe_days: int
    ) -> Dict[str, Prediction]:
        """Predecir rasgos de personalidad"""
        predictions = {}
        
        # Agrupar por rasgo
        traits_data = defaultdict(list)
        for profile in profiles:
            for trait, value in profile.personality_traits.items():
                traits_data[trait].append(value)
        
        for trait, values in traits_data.items():
            if len(values) < 2:
                continue
            
            current_value = values[-1]
            
            # Calcular tendencia simple (promedio móvil)
            if len(values) >= 3:
                recent_avg = sum(values[-3:]) / 3
                older_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else values[0]
                trend_value = recent_avg - older_avg
            else:
                trend_value = values[-1] - values[0]
            
            # Predecir valor futuro (extrapolación lineal simple)
            predicted_value = current_value + (trend_value / len(values)) * (timeframe_days / 30)
            predicted_value = max(0.0, min(1.0, predicted_value))  # Clamp
            
            # Determinar dirección de tendencia
            if abs(trend_value) < 0.05:
                trend = TrendDirection.STABLE
            elif trend_value > 0.1:
                trend = TrendDirection.INCREASING
            elif trend_value < -0.1:
                trend = TrendDirection.DECREASING
            else:
                trend = TrendDirection.VOLATILE
            
            # Calcular confianza basada en consistencia
            variance = sum((v - sum(values) / len(values))**2 for v in values) / len(values)
            confidence = max(0.3, min(0.9, 1.0 - variance))
            
            factors = []
            if trend == TrendDirection.INCREASING:
                factors.append(f"Tendencia positiva en los últimos {len(values)} análisis")
            elif trend == TrendDirection.DECREASING:
                factors.append(f"Tendencia negativa en los últimos {len(values)} análisis")
            
            predictions[f"personality_{trait}"] = Prediction(
                metric=f"personality_{trait}",
                current_value=current_value,
                predicted_value=predicted_value,
                confidence=confidence,
                trend=trend,
                timeframe_days=timeframe_days,
                factors=factors
            )
        
        return predictions
    
    def _predict_emotional_state(
        self,
        profiles: List[PsychologicalProfile],
        timeframe_days: int
    ) -> Dict[str, Prediction]:
        """Predecir estado emocional"""
        predictions = {}
        
        # Analizar sentimientos
        sentiments = []
        stress_levels = []
        
        for profile in profiles:
            emotional = profile.emotional_state
            sentiment = emotional.get("overall_sentiment", "neutral")
            stress = emotional.get("stress_level", 0.5)
            
            # Convertir sentimiento a número
            sentiment_map = {"positive": 0.7, "neutral": 0.5, "negative": 0.3}
            sentiments.append(sentiment_map.get(sentiment, 0.5))
            stress_levels.append(stress)
        
        if len(sentiments) >= 2:
            # Predecir sentimiento
            current_sentiment = sentiments[-1]
            trend = (sentiments[-1] - sentiments[0]) / len(sentiments)
            predicted_sentiment = current_sentiment + trend * (timeframe_days / 30)
            predicted_sentiment = max(0.0, min(1.0, predicted_sentiment))
            
            trend_dir = TrendDirection.STABLE
            if trend > 0.05:
                trend_dir = TrendDirection.INCREASING
            elif trend < -0.05:
                trend_dir = TrendDirection.DECREASING
            
            predictions["sentiment"] = Prediction(
                metric="sentiment",
                current_value=current_sentiment,
                predicted_value=predicted_sentiment,
                confidence=0.6,
                trend=trend_dir,
                timeframe_days=timeframe_days,
                factors=["Basado en tendencia histórica de sentimientos"]
            )
        
        if len(stress_levels) >= 2:
            # Predecir nivel de estrés
            current_stress = stress_levels[-1]
            trend = (stress_levels[-1] - stress_levels[0]) / len(stress_levels)
            predicted_stress = current_stress + trend * (timeframe_days / 30)
            predicted_stress = max(0.0, min(1.0, predicted_stress))
            
            trend_dir = TrendDirection.STABLE
            if trend > 0.05:
                trend_dir = TrendDirection.INCREASING
            elif trend < -0.05:
                trend_dir = TrendDirection.DECREASING
            
            predictions["stress_level"] = Prediction(
                metric="stress_level",
                current_value=current_stress,
                predicted_value=predicted_stress,
                confidence=0.65,
                trend=trend_dir,
                timeframe_days=timeframe_days,
                factors=["Basado en tendencia histórica de niveles de estrés"]
            )
        
        return predictions
    
    def _predict_confidence(
        self,
        profiles: List[PsychologicalProfile],
        timeframe_days: int
    ) -> Optional[Prediction]:
        """Predecir score de confianza"""
        confidence_scores = [p.confidence_score for p in profiles]
        
        if len(confidence_scores) < 2:
            return None
        
        current_confidence = confidence_scores[-1]
        trend = (confidence_scores[-1] - confidence_scores[0]) / len(confidence_scores)
        predicted_confidence = current_confidence + trend * (timeframe_days / 30)
        predicted_confidence = max(0.0, min(1.0, predicted_confidence))
        
        trend_dir = TrendDirection.STABLE
        if trend > 0.02:
            trend_dir = TrendDirection.INCREASING
        elif trend < -0.02:
            trend_dir = TrendDirection.DECREASING
        
        return Prediction(
            metric="confidence_score",
            current_value=current_confidence,
            predicted_value=predicted_confidence,
            confidence=0.7,
            trend=trend_dir,
            timeframe_days=timeframe_days,
            factors=["Basado en tendencia histórica de confianza"]
        )
    
    def detect_anomalies(
        self,
        current_profile: PsychologicalProfile,
        historical_profiles: List[PsychologicalProfile]
    ) -> List[Dict[str, Any]]:
        """
        Detectar anomalías comparando con histórico
        
        Args:
            current_profile: Perfil actual
            historical_profiles: Perfiles históricos
            
        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []
        
        if not historical_profiles:
            return anomalies
        
        # Calcular promedios históricos
        avg_traits = defaultdict(list)
        avg_stress = []
        avg_sentiment = []
        
        for profile in historical_profiles:
            for trait, value in profile.personality_traits.items():
                avg_traits[trait].append(value)
            avg_stress.append(profile.emotional_state.get("stress_level", 0.5))
            sentiment = profile.emotional_state.get("overall_sentiment", "neutral")
            sentiment_map = {"positive": 0.7, "neutral": 0.5, "negative": 0.3}
            avg_sentiment.append(sentiment_map.get(sentiment, 0.5))
        
        # Detectar anomalías en rasgos
        for trait, values in avg_traits.items():
            if not values:
                continue
            
            avg_value = sum(values) / len(values)
            std_dev = (sum((v - avg_value)**2 for v in values) / len(values))**0.5
            current_value = current_profile.personality_traits.get(trait, 0.5)
            
            # Anomalía si está fuera de 2 desviaciones estándar
            if abs(current_value - avg_value) > 2 * std_dev:
                anomalies.append({
                    "type": "personality_trait",
                    "trait": trait,
                    "current": current_value,
                    "average": avg_value,
                    "deviation": abs(current_value - avg_value),
                    "severity": "high" if abs(current_value - avg_value) > 3 * std_dev else "medium"
                })
        
        # Detectar anomalías en estrés
        if avg_stress:
            avg_stress_value = sum(avg_stress) / len(avg_stress)
            current_stress = current_profile.emotional_state.get("stress_level", 0.5)
            
            if abs(current_stress - avg_stress_value) > 0.3:
                anomalies.append({
                    "type": "stress_level",
                    "current": current_stress,
                    "average": avg_stress_value,
                    "deviation": abs(current_stress - avg_stress_value),
                    "severity": "high" if current_stress > 0.8 else "medium"
                })
        
        return anomalies


# Instancia global del analizador predictivo
predictive_analyzer = PredictiveAnalyzer()

