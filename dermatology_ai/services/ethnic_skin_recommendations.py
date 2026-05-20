"""
Sistema de recomendaciones basadas en tipo de piel étnica
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EthnicSkinProfile:
    """Perfil de piel étnica"""
    user_id: str
    skin_ethnicity: str  # "caucasian", "african", "asian", "hispanic", "middle_eastern", "mixed"
    skin_tone: str  # "light", "medium", "dark", "very_dark"
    specific_concerns: List[str] = None
    
    def __post_init__(self):
        if self.specific_concerns is None:
            self.specific_concerns = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "skin_ethnicity": self.skin_ethnicity,
            "skin_tone": self.skin_tone,
            "specific_concerns": self.specific_concerns
        }


@dataclass
class EthnicSkinRecommendation:
    """Recomendación basada en piel étnica"""
    ethnicity_factor: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    special_considerations: List[str] = None
    priority: int = 2
    
    def __post_init__(self):
        if self.special_considerations is None:
            self.special_considerations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "ethnicity_factor": self.ethnicity_factor,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "special_considerations": self.special_considerations,
            "priority": self.priority
        }


class EthnicSkinRecommendations:
    """Sistema de recomendaciones basadas en tipo de piel étnica"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, EthnicSkinProfile] = {}
    
    def create_ethnic_profile(self, user_id: str, skin_ethnicity: str,
                             skin_tone: str, specific_concerns: Optional[List[str]] = None) -> EthnicSkinProfile:
        """Crea perfil de piel étnica"""
        profile = EthnicSkinProfile(
            user_id=user_id,
            skin_ethnicity=skin_ethnicity,
            skin_tone=skin_tone,
            specific_concerns=specific_concerns or []
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_ethnic_recommendations(self, user_id: str) -> List[EthnicSkinRecommendation]:
        """Obtiene recomendaciones basadas en piel étnica"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones para piel africana
        if profile.skin_ethnicity == "african":
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Piel Africana",
                recommendations=[
                    "Mayor riesgo de hiperpigmentación post-inflamatoria",
                    "Protección solar es esencial (aunque la piel sea más oscura)",
                    "Evita productos que puedan causar irritación",
                    "Considera productos con niacinamida para uniformizar tono"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "broad_spectrum", "priority": 1},
                    {"category": "serum", "type": "niacinamide", "reason": "Uniformizar tono"}
                ],
                special_considerations=[
                    "La hiperpigmentación es común - sé paciente con tratamientos",
                    "Evita exfoliación agresiva que pueda causar más pigmentación"
                ],
                priority=1
            ))
        
        # Recomendaciones para piel asiática
        if profile.skin_ethnicity == "asian":
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Piel Asiática",
                recommendations=[
                    "Tendencia a hiperpigmentación",
                    "Protección solar diaria crítica",
                    "Productos suaves y no irritantes",
                    "Considera productos con arbutin o ácido kójico"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50", "priority": 1},
                    {"category": "serum", "type": "brightening", "reason": "Uniformizar tono"}
                ],
                special_considerations=[
                    "Evita productos con fragancias fuertes",
                    "Rutina de doble limpieza puede ser beneficiosa"
                ],
                priority=1
            ))
        
        # Recomendaciones para piel hispana/latina
        if profile.skin_ethnicity == "hispanic":
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Piel Hispana/Latina",
                recommendations=[
                    "Rango amplio de tonos - personaliza según tu tono específico",
                    "Protección solar importante para todos los tonos",
                    "Considera productos para uniformizar si hay hiperpigmentación"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "broad_spectrum", "priority": 1}
                ],
                special_considerations=[
                    "Las necesidades varían según el tono específico"
                ],
                priority=2
            ))
        
        # Recomendaciones para piel caucásica
        if profile.skin_ethnicity == "caucasian":
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Piel Caucásica",
                recommendations=[
                    "Mayor riesgo de daño solar y cáncer de piel",
                    "Protección solar diaria obligatoria",
                    "Productos anti-envejecimiento pueden ser beneficiosos temprano",
                    "Monitorea cambios en lunares y manchas"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "spf_50", "priority": 1},
                    {"category": "serum", "type": "antioxidant", "reason": "Protección adicional"}
                ],
                special_considerations=[
                    "Exámenes regulares de la piel recomendados",
                    "Evita exposición solar prolongada sin protección"
                ],
                priority=1
            ))
        
        # Recomendaciones para piel de Medio Oriente
        if profile.skin_ethnicity == "middle_eastern":
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Piel de Medio Oriente",
                recommendations=[
                    "Puede tener tendencia a oleosidad",
                    "Protección solar importante",
                    "Considera productos para controlar brillo",
                    "Productos para uniformizar tono si es necesario"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "oil_free", "priority": 1},
                    {"category": "serum", "type": "niacinamide", "reason": "Control de oleosidad y tono"}
                ],
                priority=2
            ))
        
        # Recomendaciones según tono
        if profile.skin_tone in ["dark", "very_dark"]:
            recommendations.append(EthnicSkinRecommendation(
                ethnicity_factor="Tono Oscuro",
                recommendations=[
                    "Aunque la piel sea más oscura, el protector solar es esencial",
                    "Busca protectores solares que no dejen residuo blanco",
                    "Considera protectores con óxido de zinc o dióxido de titanio micronizado"
                ],
                product_suggestions=[
                    {"category": "sunscreen", "type": "tinted_or_clear", "reason": "Sin residuo blanco"}
                ],
                priority=1
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations


