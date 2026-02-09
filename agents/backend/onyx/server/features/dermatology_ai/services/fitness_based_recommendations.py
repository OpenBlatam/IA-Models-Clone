"""
Sistema de recomendaciones basadas en actividad física
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FitnessProfile:
    """Perfil de actividad física"""
    user_id: str
    workout_frequency: str  # "daily", "weekly", "occasional", "none"
    workout_type: List[str]  # "cardio", "strength", "yoga", "swimming", "outdoor"
    sweat_level: str  # "high", "medium", "low"
    shower_frequency_post_workout: str  # "always", "usually", "sometimes", "rarely"
    outdoor_activities: bool
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "workout_frequency": self.workout_frequency,
            "workout_type": self.workout_type,
            "sweat_level": self.sweat_level,
            "shower_frequency_post_workout": self.shower_frequency_post_workout,
            "outdoor_activities": self.outdoor_activities
        }


@dataclass
class FitnessRecommendation:
    """Recomendación basada en actividad física"""
    fitness_factor: str
    impact: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "fitness_factor": self.fitness_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class FitnessBasedRecommendations:
    """Sistema de recomendaciones basadas en actividad física"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, FitnessProfile] = {}  # user_id -> profile
    
    def create_fitness_profile(self, user_id: str, workout_frequency: str,
                              workout_type: List[str], sweat_level: str,
                              shower_frequency_post_workout: str,
                              outdoor_activities: bool) -> FitnessProfile:
        """Crea perfil de actividad física"""
        profile = FitnessProfile(
            user_id=user_id,
            workout_frequency=workout_frequency,
            workout_type=workout_type,
            sweat_level=sweat_level,
            shower_frequency_post_workout=shower_frequency_post_workout,
            outdoor_activities=outdoor_activities
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_fitness_recommendations(self, user_id: str) -> List[FitnessRecommendation]:
        """Obtiene recomendaciones basadas en actividad física"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones por frecuencia de ejercicio
        if profile.workout_frequency in ["daily", "weekly"]:
            recommendations.append(FitnessRecommendation(
                fitness_factor="Ejercicio Regular",
                impact="positive",
                recommendations=[
                    "Limpia tu piel inmediatamente después del ejercicio",
                    "Usa productos suaves post-entrenamiento",
                    "Hidrata bien después de la ducha"
                ],
                product_suggestions=[
                    {"category": "cleanser", "type": "gentle", "reason": "Limpieza suave post-entrenamiento"}
                ],
                priority=2
            ))
        
        # Recomendaciones por nivel de sudor
        if profile.sweat_level == "high":
            recommendations.append(FitnessRecommendation(
                fitness_factor="Alta Sudoración",
                impact="medium",
                recommendations=[
                    "Limpia los poros después del ejercicio",
                    "Usa productos con ácido salicílico para prevenir brotes",
                    "Evita dejar el sudor secarse en la piel"
                ],
                product_suggestions=[
                    {"category": "toner", "type": "salicylic_acid", "reason": "Prevenir brotes por sudor"}
                ],
                priority=1
            ))
        
        # Recomendaciones por actividades al aire libre
        if profile.outdoor_activities:
            recommendations.append(FitnessRecommendation(
                fitness_factor="Actividades al Aire Libre",
                impact="high",
                recommendations=[
                    "Protección solar es esencial",
                    "Reaplica SPF cada 2 horas durante ejercicio",
                    "Usa productos con antioxidantes para combatir daño ambiental"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "sport", "priority": 1, "reason": "Resistente al sudor"},
                    {"category": "serum", "type": "antioxidant", "reason": "Protección ambiental"}
                ],
                priority=1
            ))
        
        # Recomendaciones por tipo de ejercicio
        if "swimming" in profile.workout_type:
            recommendations.append(FitnessRecommendation(
                fitness_factor="Natación",
                impact="medium",
                recommendations=[
                    "El cloro puede secar la piel",
                    "Dúchate inmediatamente después de nadar",
                    "Hidratación intensa necesaria"
                ],
                product_suggestions=[
                    {"category": "moisturizer", "type": "barrier_repair", "reason": "Reparar daño del cloro"}
                ],
                priority=2
            ))
        
        if "outdoor" in profile.workout_type or profile.outdoor_activities:
            recommendations.append(FitnessRecommendation(
                fitness_factor="Ejercicio al Aire Libre",
                impact="high",
                recommendations=[
                    "Protección solar obligatoria",
                    "Evita horas pico de sol (10am-4pm)",
                    "Usa gorra o visera"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50", "priority": 1}
                ],
                priority=1
            ))
        
        # Recomendaciones por frecuencia de ducha
        if profile.shower_frequency_post_workout in ["rarely", "sometimes"]:
            recommendations.append(FitnessRecommendation(
                fitness_factor="Limpieza Post-Entrenamiento",
                impact="high",
                recommendations=[
                    "Es importante limpiar la piel después del ejercicio",
                    "Al menos usa toallitas limpiadoras si no puedes ducharte",
                    "El sudor y bacterias pueden causar brotes"
                ],
                product_suggestions=[
                    {"category": "wipes", "type": "cleansing", "reason": "Limpieza rápida post-entrenamiento"}
                ],
                priority=1
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations






