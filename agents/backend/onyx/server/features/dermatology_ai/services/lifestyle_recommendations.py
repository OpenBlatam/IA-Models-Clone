"""
Sistema de recomendaciones basadas en estilo de vida
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LifestyleProfile:
    """Perfil de estilo de vida"""
    user_id: str
    diet: str  # "healthy", "average", "poor"
    exercise_frequency: str  # "daily", "weekly", "rarely", "never"
    sleep_hours: float
    stress_level: str  # "low", "medium", "high"
    sun_exposure: str  # "high", "medium", "low"
    smoking: bool
    alcohol_consumption: str  # "none", "occasional", "regular"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "diet": self.diet,
            "exercise_frequency": self.exercise_frequency,
            "sleep_hours": self.sleep_hours,
            "stress_level": self.stress_level,
            "sun_exposure": self.sun_exposure,
            "smoking": self.smoking,
            "alcohol_consumption": self.alcohol_consumption
        }


@dataclass
class LifestyleRecommendation:
    """Recomendación basada en estilo de vida"""
    lifestyle_factor: str
    impact: str  # "positive", "negative", "neutral"
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "lifestyle_factor": self.lifestyle_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class LifestyleRecommendations:
    """Sistema de recomendaciones basadas en estilo de vida"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, LifestyleProfile] = {}  # user_id -> profile
    
    def create_profile(self, user_id: str, diet: str, exercise_frequency: str,
                      sleep_hours: float, stress_level: str, sun_exposure: str,
                      smoking: bool, alcohol_consumption: str) -> LifestyleProfile:
        """Crea perfil de estilo de vida"""
        profile = LifestyleProfile(
            user_id=user_id,
            diet=diet,
            exercise_frequency=exercise_frequency,
            sleep_hours=sleep_hours,
            stress_level=stress_level,
            sun_exposure=sun_exposure,
            smoking=smoking,
            alcohol_consumption=alcohol_consumption
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_lifestyle_recommendations(self, user_id: str) -> List[LifestyleRecommendation]:
        """Obtiene recomendaciones basadas en estilo de vida"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Análisis de dieta
        if profile.diet == "poor":
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Dieta",
                impact="negative",
                recommendations=[
                    "Mejora tu dieta para mejor salud de la piel",
                    "Aumenta consumo de frutas y verduras",
                    "Reduce alimentos procesados"
                ],
                product_suggestions=[
                    {"category": "supplements", "type": "vitamins", "reason": "Suplementar nutrientes"}
                ],
                priority=2
            ))
        
        # Análisis de ejercicio
        if profile.exercise_frequency in ["rarely", "never"]:
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Ejercicio",
                impact="negative",
                recommendations=[
                    "El ejercicio mejora la circulación y salud de la piel",
                    "Considera ejercicio moderado regular"
                ],
                product_suggestions=[],
                priority=3
            ))
        
        # Análisis de sueño
        if profile.sleep_hours < 6:
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Sueño",
                impact="negative",
                recommendations=[
                    "Sueño insuficiente afecta la regeneración de la piel",
                    "Intenta dormir 7-8 horas diarias",
                    "Usa productos reparadores nocturnos"
                ],
                product_suggestions=[
                    {"category": "night_cream", "reason": "Reparación durante el sueño"}
                ],
                priority=1
            ))
        
        # Análisis de estrés
        if profile.stress_level == "high":
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Estrés",
                impact="negative",
                recommendations=[
                    "El estrés afecta la salud de la piel",
                    "Practica técnicas de relajación",
                    "Considera productos calmantes"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "calming", "reason": "Reducir inflamación por estrés"}
                ],
                priority=2
            ))
        
        # Análisis de exposición solar
        if profile.sun_exposure == "high":
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Exposición Solar",
                impact="negative",
                recommendations=[
                    "Protección solar diaria es esencial",
                    "Usa SPF 50+ si estás mucho al sol",
                    "Reaplica cada 2 horas"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50", "priority": 1, "reason": "Protección máxima"}
                ],
                priority=1
            ))
        
        # Análisis de tabaco
        if profile.smoking:
            recommendations.append(LifestyleRecommendation(
                lifestyle_factor="Tabaco",
                impact="negative",
                recommendations=[
                    "El tabaco acelera el envejecimiento",
                    "Considera dejar de fumar",
                    "Usa productos antioxidantes intensos"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "antioxidant", "reason": "Combatir daño oxidativo"}
                ],
                priority=1
            ))
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations






