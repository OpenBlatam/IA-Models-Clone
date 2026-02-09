"""
Machine Learning para Mejoras Continuas
========================================
Sistema de ML para mejorar análisis basado en feedback
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog
from collections import defaultdict

logger = structlog.get_logger()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available, using basic calculations")

from .models import PsychologicalProfile
from .feedback import FeedbackManager, FeedbackRating

logger = structlog.get_logger()


class MLModel:
    """Modelo de ML simplificado para mejoras"""
    
    def __init__(self):
        """Inicializar modelo"""
        self.weights = {
            "openness": 1.0,
            "conscientiousness": 1.0,
            "extraversion": 1.0,
            "agreeableness": 1.0,
            "neuroticism": 1.0
        }
        self.learning_rate = 0.01
        logger.info("MLModel initialized")
    
    def adjust_weights(
        self,
        profile: PsychologicalProfile,
        feedback_rating: FeedbackRating
    ) -> None:
        """
        Ajustar pesos basado en feedback
        
        Args:
            profile: Perfil psicológico
            feedback_rating: Calificación del feedback
        """
        # Mapear rating a valor numérico
        rating_map = {
            FeedbackRating.VERY_POOR: -2,
            FeedbackRating.POOR: -1,
            FeedbackRating.NEUTRAL: 0,
            FeedbackRating.GOOD: 1,
            FeedbackRating.EXCELLENT: 2
        }
        
        rating_value = rating_map.get(feedback_rating, 0)
        
        # Ajustar pesos basado en feedback
        for trait in self.weights:
            trait_value = profile.personality_traits.get(trait, 0.5)
            
            # Si el feedback es positivo y el trait es alto, aumentar peso
            # Si el feedback es negativo, ajustar según corresponda
            adjustment = rating_value * self.learning_rate * trait_value
            self.weights[trait] = max(0.1, min(2.0, self.weights[trait] + adjustment))
        
        logger.info(
            "Weights adjusted",
            rating=feedback_rating.value,
            weights=self.weights.copy()
        )
    
    def apply_weights(
        self,
        traits: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aplicar pesos a rasgos
        
        Args:
            traits: Rasgos de personalidad
            
        Returns:
            Rasgos ajustados
        """
        adjusted = {}
        for trait, value in traits.items():
            weight = self.weights.get(trait, 1.0)
            adjusted_value = value * weight
            adjusted[trait] = max(0.0, min(1.0, adjusted_value))
        
        return adjusted


class MLImprovementEngine:
    """Motor de mejoras con ML"""
    
    def __init__(self, feedback_manager: FeedbackManager):
        """
        Inicializar motor
        
        Args:
            feedback_manager: Gestor de feedback
        """
        self.feedback_manager = feedback_manager
        self.model = MLModel()
        self._training_data: List[Dict[str, Any]] = []
        logger.info("MLImprovementEngine initialized")
    
    def train_from_feedback(
        self,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Entrenar modelo desde feedback
        
        Args:
            limit: Límite de feedbacks a procesar
            
        Returns:
            Estadísticas de entrenamiento
        """
        feedbacks = self.feedback_manager.get_feedback(limit=limit)
        
        if not feedbacks:
            return {
                "trained": False,
                "reason": "No feedback available"
            }
        
        training_count = 0
        for feedback in feedbacks:
            # En producción, obtener perfil asociado
            # Por ahora, simular
            if feedback.rating != FeedbackRating.NEUTRAL:
                training_count += 1
                # self.model.adjust_weights(profile, feedback.rating)
        
        logger.info(
            "Model training completed",
            total_feedbacks=len(feedbacks),
            training_samples=training_count
        )
        
        return {
            "trained": True,
            "total_feedbacks": len(feedbacks),
            "training_samples": training_count,
            "weights": self.model.weights.copy()
        }
    
    def predict_confidence(
        self,
        profile: PsychologicalProfile
    ) -> float:
        """
        Predecir score de confianza usando ML
        
        Args:
            profile: Perfil psicológico
            
        Returns:
            Score de confianza predicho
        """
        # Aplicar pesos ajustados
        adjusted_traits = self.model.apply_weights(profile.personality_traits)
        
        # Calcular confianza basada en consistencia de rasgos ajustados
        trait_values = list(adjusted_traits.values())
        if not trait_values:
            return 0.5
        
        # Confianza basada en varianza (menor varianza = mayor confianza)
        if NUMPY_AVAILABLE:
            variance = np.var(trait_values)
        else:
            # Cálculo básico de varianza
            mean = sum(trait_values) / len(trait_values)
            variance = sum((x - mean) ** 2 for x in trait_values) / len(trait_values)
        confidence = max(0.3, min(0.95, 1.0 - variance))
        
        return confidence
    
    def get_improvement_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de mejora
        
        Returns:
            Métricas
        """
        feedback_stats = self.feedback_manager.get_feedback_stats()
        
        return {
            "model_weights": self.model.weights.copy(),
            "total_feedbacks": feedback_stats.get("total", 0),
            "average_rating": feedback_stats.get("average_rating", 0.0),
            "learning_rate": self.model.learning_rate,
            "training_samples": len(self._training_data)
        }
    
    def suggest_improvements(
        self,
        profile: PsychologicalProfile
    ) -> List[str]:
        """
        Sugerir mejoras basadas en ML
        
        Args:
            profile: Perfil psicológico
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Analizar rasgos con pesos ajustados
        adjusted_traits = self.model.apply_weights(profile.personality_traits)
        
        # Detectar rasgos que necesitan atención
        for trait, value in adjusted_traits.items():
            weight = self.model.weights.get(trait, 1.0)
            
            if weight < 0.8:
                suggestions.append(
                    f"Consider improving analysis for {trait} trait "
                    f"(current weight: {weight:.2f})"
                )
        
        return suggestions


# Instancia global del motor de ML
ml_engine = None  # Se inicializa con feedback_manager

