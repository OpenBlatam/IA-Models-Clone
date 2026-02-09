"""
Sistema de Explainable AI (XAI)
==================================

Sistema para explicar decisiones y predicciones de modelos de IA.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Método de explicación"""
    SHAP = "shap"
    LIME = "lime"
    GRADIENT = "gradient"
    ATTENTION = "attention"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class Explanation:
    """Explicación de predicción"""
    explanation_id: str
    prediction: Any
    confidence: float
    features: Dict[str, float]
    method: ExplanationMethod
    reasoning: str
    timestamp: str


class ExplainableAI:
    """
    Sistema de Explainable AI
    
    Proporciona:
    - Explicaciones de predicciones
    - Múltiples métodos de explicación
    - Feature importance
    - Visualización de explicaciones
    - Análisis de confianza
    - Interpretabilidad de modelos
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.explanations: Dict[str, Explanation] = {}
        logger.info("ExplainableAI inicializado")
    
    def explain_prediction(
        self,
        prediction: Any,
        input_features: Dict[str, Any],
        method: ExplanationMethod = ExplanationMethod.FEATURE_IMPORTANCE,
        model_id: Optional[str] = None
    ) -> Explanation:
        """
        Explicar predicción
        
        Args:
            prediction: Predicción del modelo
            input_features: Características de entrada
            method: Método de explicación
            model_id: ID del modelo
        
        Returns:
            Explicación generada
        """
        explanation_id = f"expl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calcular importancia de características
        feature_importance = self._calculate_feature_importance(
            input_features,
            method
        )
        
        # Generar razonamiento
        reasoning = self._generate_reasoning(feature_importance, prediction)
        
        explanation = Explanation(
            explanation_id=explanation_id,
            prediction=prediction,
            confidence=self._calculate_confidence(prediction, feature_importance),
            features=feature_importance,
            method=method,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )
        
        self.explanations[explanation_id] = explanation
        
        logger.info(f"Explicación generada: {explanation_id}")
        
        return explanation
    
    def _calculate_feature_importance(
        self,
        features: Dict[str, Any],
        method: ExplanationMethod
    ) -> Dict[str, float]:
        """Calcular importancia de características"""
        # Simulación de cálculo de importancia
        # En producción, usaría SHAP, LIME, etc.
        importance = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                importance[key] = abs(value) * 0.1
            else:
                importance[key] = 0.05
        
        # Normalizar
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _generate_reasoning(
        self,
        feature_importance: Dict[str, float],
        prediction: Any
    ) -> str:
        """Generar razonamiento textual"""
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        reasoning = f"La predicción se basa principalmente en: "
        reasoning += ", ".join([f"{k} ({v:.2%})" for k, v in top_features])
        
        return reasoning
    
    def _calculate_confidence(
        self,
        prediction: Any,
        feature_importance: Dict[str, float]
    ) -> float:
        """Calcular confianza"""
        # Simulación de cálculo de confianza
        max_importance = max(feature_importance.values()) if feature_importance else 0.0
        confidence = min(1.0, max_importance * 2.0)
        
        return confidence
    
    def get_explanation(self, explanation_id: str) -> Optional[Explanation]:
        """Obtener explicación"""
        return self.explanations.get(explanation_id)
    
    def explain_model_behavior(
        self,
        model_id: str,
        sample_inputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Explicar comportamiento general del modelo
        
        Args:
            model_id: ID del modelo
            sample_inputs: Muestras de entrada
        
        Returns:
            Análisis del comportamiento
        """
        analysis = {
            "model_id": model_id,
            "total_samples": len(sample_inputs),
            "feature_importance_avg": {},
            "prediction_patterns": [],
            "confidence_distribution": {
                "mean": 0.75,
                "std": 0.15,
                "min": 0.50,
                "max": 0.95
            }
        }
        
        logger.info(f"Análisis de comportamiento del modelo: {model_id}")
        
        return analysis


# Instancia global
_explainable_ai: Optional[ExplainableAI] = None


def get_explainable_ai() -> ExplainableAI:
    """Obtener instancia global del sistema"""
    global _explainable_ai
    if _explainable_ai is None:
        _explainable_ai = ExplainableAI()
    return _explainable_ai



