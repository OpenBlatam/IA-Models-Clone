"""
Sistema de predicción de condiciones de piel
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class ConditionPrediction:
    """Predicción de condición"""
    condition_name: str
    probability: float
    severity: str  # "mild", "moderate", "severe"
    confidence: float
    risk_factors: List[str]
    recommendations: List[str]
    predicted_at: str = None
    
    def __post_init__(self):
        if self.predicted_at is None:
            self.predicted_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "condition_name": self.condition_name,
            "probability": self.probability,
            "severity": self.severity,
            "confidence": self.confidence,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "predicted_at": self.predicted_at
        }


class ConditionPredictor:
    """Sistema de predicción de condiciones"""
    
    def __init__(self):
        """Inicializa el predictor"""
        self.condition_models: Dict[str, Any] = {}
        self.risk_factors_db: Dict[str, List[str]] = {
            "acne": ["exceso_sebo", "poros_obstruidos", "inflamacion"],
            "dryness": ["baja_hidratacion", "barrera_danada", "perdida_agua"],
            "wrinkles": ["edad", "exposicion_solar", "perdida_colageno"],
            "hyperpigmentation": ["exposicion_solar", "inflamacion", "genetica"],
            "sensitivity": ["barrera_danada", "inflamacion", "alergenos"]
        }
    
    def predict_conditions(self, analysis_features: Dict,
                          user_history: Optional[List[Dict]] = None) -> List[ConditionPrediction]:
        """
        Predice condiciones de piel
        
        Args:
            analysis_features: Features del análisis
            user_history: Historial del usuario (opcional)
            
        Returns:
            Lista de predicciones
        """
        predictions = []
        
        # Analizar features para cada condición conocida
        for condition_name, risk_factors in self.risk_factors_db.items():
            probability = self._calculate_condition_probability(
                condition_name, analysis_features, user_history
            )
            
            if probability > 0.3:  # Solo incluir si probabilidad > 30%
                severity = self._determine_severity(probability)
                confidence = self._calculate_confidence(probability, analysis_features)
                
                prediction = ConditionPrediction(
                    condition_name=condition_name,
                    probability=probability,
                    severity=severity,
                    confidence=confidence,
                    risk_factors=risk_factors,
                    recommendations=self._generate_recommendations(condition_name, severity)
                )
                predictions.append(prediction)
        
        # Ordenar por probabilidad
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        return predictions
    
    def _calculate_condition_probability(self, condition_name: str,
                                        features: Dict,
                                        history: Optional[List[Dict]]) -> float:
        """Calcula probabilidad de condición"""
        # Placeholder - implementar con modelo ML real
        base_probability = 0.0
        
        # Factores de riesgo basados en features
        if condition_name == "acne":
            if features.get("sebum_level", 0) > 0.7:
                base_probability += 0.3
            if features.get("pore_density", 0) > 0.6:
                base_probability += 0.2
        
        elif condition_name == "dryness":
            if features.get("hydration_level", 0) < 0.4:
                base_probability += 0.4
            if features.get("barrier_integrity", 0) < 0.5:
                base_probability += 0.3
        
        elif condition_name == "wrinkles":
            if features.get("age_factor", 0) > 0.5:
                base_probability += 0.3
            if features.get("sun_damage", 0) > 0.6:
                base_probability += 0.3
        
        # Ajustar con historial
        if history:
            condition_count = sum(
                1 for h in history
                if condition_name in [c.get("name") for c in h.get("conditions", [])]
            )
            if condition_count > 0:
                base_probability += min(0.3, condition_count * 0.1)
        
        return min(1.0, base_probability)
    
    def _determine_severity(self, probability: float) -> str:
        """Determina severidad"""
        if probability >= 0.7:
            return "severe"
        elif probability >= 0.5:
            return "moderate"
        else:
            return "mild"
    
    def _calculate_confidence(self, probability: float, features: Dict) -> float:
        """Calcula confianza de la predicción"""
        # Confianza basada en calidad de features
        feature_quality = len([v for v in features.values() if v is not None]) / max(len(features), 1)
        confidence = probability * feature_quality
        return float(confidence)
    
    def _generate_recommendations(self, condition_name: str, severity: str) -> List[str]:
        """Genera recomendaciones para condición"""
        recommendations = {
            "acne": {
                "mild": ["Limpieza suave", "Productos no comedogénicos"],
                "moderate": ["Tratamiento con ácido salicílico", "Consulta dermatológica"],
                "severe": ["Consulta dermatológica urgente", "Tratamiento médico"]
            },
            "dryness": {
                "mild": ["Hidratación diaria", "Evitar limpiadores agresivos"],
                "moderate": ["Hidratación intensiva", "Barrera reparadora"],
                "severe": ["Tratamiento profesional", "Consulta dermatológica"]
            },
            "wrinkles": {
                "mild": ["Protección solar", "Antioxidantes"],
                "moderate": ["Retinoides", "Péptidos"],
                "severe": ["Tratamiento profesional", "Procedimientos estéticos"]
            }
        }
        
        return recommendations.get(condition_name, {}).get(severity, ["Consulta profesional"])

