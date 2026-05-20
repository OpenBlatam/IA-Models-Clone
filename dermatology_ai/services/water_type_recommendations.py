"""
Sistema de recomendaciones basadas en tipo de agua
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WaterProfile:
    """Perfil de tipo de agua"""
    user_id: str
    water_type: str  # "hard", "soft", "chlorinated", "well", "filtered", "distilled"
    ph_level: Optional[float] = None
    mineral_content: str  # "high", "medium", "low"
    chlorine_present: bool = False
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "water_type": self.water_type,
            "ph_level": self.ph_level,
            "mineral_content": self.mineral_content,
            "chlorine_present": self.chlorine_present
        }


@dataclass
class WaterRecommendation:
    """Recomendación basada en tipo de agua"""
    water_factor: str
    impact: str  # "positive", "negative", "neutral"
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "water_factor": self.water_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class WaterTypeRecommendations:
    """Sistema de recomendaciones basadas en tipo de agua"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, WaterProfile] = {}  # user_id -> profile
    
    def create_water_profile(self, user_id: str, water_type: str,
                            ph_level: Optional[float] = None,
                            mineral_content: str = "medium",
                            chlorine_present: bool = False) -> WaterProfile:
        """Crea perfil de tipo de agua"""
        profile = WaterProfile(
            user_id=user_id,
            water_type=water_type,
            ph_level=ph_level,
            mineral_content=mineral_content,
            chlorine_present=chlorine_present
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_water_recommendations(self, user_id: str) -> List[WaterRecommendation]:
        """Obtiene recomendaciones basadas en tipo de agua"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones por tipo de agua dura
        if profile.water_type == "hard":
            recommendations.append(WaterRecommendation(
                water_factor="Agua Dura",
                impact="negative",
                recommendations=[
                    "El agua dura puede dejar residuos en la piel",
                    "Usa un limpiador que elimine minerales",
                    "Considera un filtro de agua para la ducha",
                    "Hidrata bien después de lavarte la cara"
                ],
                product_suggestions=[
                    {"category": "cleanser", "type": "chelating", "reason": "Elimina minerales del agua dura"}
                ],
                priority=2
            ))
        
        # Recomendaciones por cloro
        if profile.chlorine_present or profile.water_type == "chlorinated":
            recommendations.append(WaterRecommendation(
                water_factor="Cloro en el Agua",
                impact="negative",
                recommendations=[
                    "El cloro puede secar e irritar la piel",
                    "Usa productos con ingredientes calmantes",
                    "Considera un filtro de ducha que elimine cloro",
                    "Hidratación intensa después de la ducha"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "calming", "reason": "Calmar irritación por cloro"},
                    {"category": "moisturizer", "type": "barrier_repair", "reason": "Reparar barrera dañada"}
                ],
                priority=1
            ))
        
        # Recomendaciones por pH
        if profile.ph_level:
            if profile.ph_level > 8.0:
                recommendations.append(WaterRecommendation(
                    water_factor="pH Alto",
                    impact="negative",
                    recommendations=[
                        "Agua alcalina puede alterar el pH natural de la piel",
                        "Usa tóners con pH balanceado",
                        "Considera agua filtrada o destilada para limpieza"
                    ],
                    product_suggestions=[
                        {"category": "toner", "type": "ph_balanced", "reason": "Balancear pH de la piel"}
                    ],
                    priority=2
                ))
            elif profile.ph_level < 6.0:
                recommendations.append(WaterRecommendation(
                    water_factor="pH Bajo",
                    impact="neutral",
                    recommendations=[
                        "Agua ácida puede ser beneficiosa para algunos tipos de piel",
                        "Monitorea la sensibilidad de tu piel"
                    ],
                    product_suggestions=[],
                    priority=3
                ))
        
        # Recomendaciones por contenido mineral
        if profile.mineral_content == "high":
            recommendations.append(WaterRecommendation(
                water_factor="Alto Contenido Mineral",
                impact="neutral",
                recommendations=[
                    "Minerales pueden acumularse en la piel",
                    "Exfoliación regular puede ayudar",
                    "Limpieza profunda semanal recomendada"
                ],
                product_suggestions=[
                    {"category": "exfoliant", "type": "gentle", "reason": "Remover acumulación de minerales"}
                ],
                priority=3
            ))
        
        # Recomendaciones para agua filtrada/destilada
        if profile.water_type in ["filtered", "distilled"]:
            recommendations.append(WaterRecommendation(
                water_factor="Agua Filtrada/Destilada",
                impact="positive",
                recommendations=[
                    "Excelente para la piel - sin minerales ni cloro",
                    "Mantén tu rutina actual",
                    "No necesitas productos especiales para compensar"
                ],
                product_suggestions=[],
                priority=3
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations






