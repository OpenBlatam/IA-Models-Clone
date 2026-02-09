"""
Sistema de recomendaciones basadas en tipo de trabajo
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class OccupationProfile:
    """Perfil de ocupación"""
    user_id: str
    occupation_type: str  # "office", "outdoor", "healthcare", "hospitality", "retail", "construction", "education"
    work_hours: str  # "day", "night", "shift"
    exposure_factors: List[str] = None  # "sun", "pollution", "chemicals", "stress", "screen_time"
    
    def __post_init__(self):
        if self.exposure_factors is None:
            self.exposure_factors = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "occupation_type": self.occupation_type,
            "work_hours": self.work_hours,
            "exposure_factors": self.exposure_factors
        }


@dataclass
class OccupationRecommendation:
    """Recomendación basada en ocupación"""
    occupation_factor: str
    impact: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "occupation_factor": self.occupation_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class OccupationRecommendations:
    """Sistema de recomendaciones basadas en ocupación"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, OccupationProfile] = {}
    
    def create_occupation_profile(self, user_id: str, occupation_type: str,
                                 work_hours: str, exposure_factors: Optional[List[str]] = None) -> OccupationProfile:
        """Crea perfil de ocupación"""
        profile = OccupationProfile(
            user_id=user_id,
            occupation_type=occupation_type,
            work_hours=work_hours,
            exposure_factors=exposure_factors or []
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_occupation_recommendations(self, user_id: str) -> List[OccupationRecommendation]:
        """Obtiene recomendaciones basadas en ocupación"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones para trabajo al aire libre
        if profile.occupation_type == "outdoor" or "sun" in profile.exposure_factors:
            recommendations.append(OccupationRecommendation(
                occupation_factor="Exposición Solar",
                impact="high",
                recommendations=[
                    "Protección solar extrema es crítica",
                    "Reaplica SPF cada 2 horas",
                    "Usa sombrero y protección física",
                    "Productos con antioxidantes para combatir daño solar"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50", "priority": 1},
                    {"category": "serum", "type": "antioxidant", "reason": "Protección adicional"}
                ],
                priority=1
            ))
        
        # Recomendaciones para trabajo de oficina
        if profile.occupation_type == "office" or "screen_time" in profile.exposure_factors:
            recommendations.append(OccupationRecommendation(
                occupation_factor="Tiempo frente a Pantalla",
                impact="medium",
                recommendations=[
                    "Luz azul puede afectar la piel",
                    "Hidratación regular importante",
                    "Considera productos con protección de luz azul",
                    "Descansos regulares de la pantalla"
                ],
                product_suggestions=[
                    {"category": "moisturizer", "type": "hydrating", "reason": "Hidratación durante el día"}
                ],
                priority=2
            ))
        
        # Recomendaciones para trabajo nocturno
        if profile.work_hours == "night":
            recommendations.append(OccupationRecommendation(
                occupation_factor="Trabajo Nocturno",
                impact="medium",
                recommendations=[
                    "Rutina de cuidado puede ser diferente",
                    "Protección solar aún necesaria (luz artificial)",
                    "Prioriza descanso y sueño de calidad",
                    "Productos reparadores nocturnos importantes"
                ],
                product_suggestions=[
                    {"category": "moisturizer", "type": "night_repair", "reason": "Reparación durante el sueño"}
                ],
                priority=2
            ))
        
        # Recomendaciones para exposición a químicos
        if "chemicals" in profile.exposure_factors:
            recommendations.append(OccupationRecommendation(
                occupation_factor="Exposición a Químicos",
                impact="high",
                recommendations=[
                    "Limpieza profunda al final del día",
                    "Productos de barrera protectora",
                    "Evita productos que puedan aumentar sensibilidad",
                    "Considera guantes y protección física"
                ],
                product_suggestions=[
                    {"category": "cleanser", "type": "deep_cleansing", "priority": 1},
                    {"category": "moisturizer", "type": "barrier_repair", "priority": 1}
                ],
                priority=1
            ))
        
        # Recomendaciones para contaminación
        if "pollution" in profile.exposure_factors:
            recommendations.append(OccupationRecommendation(
                occupation_factor="Contaminación",
                impact="high",
                recommendations=[
                    "Doble limpieza al final del día",
                    "Productos con antioxidantes",
                    "Protección de barrera",
                    "Exfoliación regular para remover partículas"
                ],
                product_suggestions=[
                    {"category": "cleanser", "type": "double_cleansing", "priority": 1},
                    {"category": "serum", "type": "antioxidant", "reason": "Combatir daño de contaminación"}
                ],
                priority=1
            ))
        
        # Recomendaciones para estrés laboral
        if "stress" in profile.exposure_factors:
            recommendations.append(OccupationRecommendation(
                occupation_factor="Estrés Laboral",
                impact="medium",
                recommendations=[
                    "El estrés afecta la piel",
                    "Productos calmantes pueden ayudar",
                    "Técnicas de manejo de estrés",
                    "Rutina de cuidado como autocuidado"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "calming", "reason": "Reducir inflamación por estrés"}
                ],
                priority=2
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations


