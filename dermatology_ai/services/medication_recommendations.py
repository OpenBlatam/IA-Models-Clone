"""
Sistema de recomendaciones basadas en medicamentos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MedicationProfile:
    """Perfil de medicamentos"""
    user_id: str
    medications: List[Dict]  # [{"name": str, "type": str, "frequency": str}]
    supplements: List[str] = None
    skin_related_medications: List[str] = None
    
    def __post_init__(self):
        if self.supplements is None:
            self.supplements = []
        if self.skin_related_medications is None:
            self.skin_related_medications = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "medications": self.medications,
            "supplements": self.supplements,
            "skin_related_medications": self.skin_related_medications
        }


@dataclass
class MedicationRecommendation:
    """Recomendación basada en medicamentos"""
    medication_factor: str
    impact: str  # "positive", "negative", "neutral", "interaction"
    recommendations: List[str]
    product_suggestions: List[Dict]
    warnings: List[str] = None
    priority: int = 2
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "medication_factor": self.medication_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "warnings": self.warnings,
            "priority": self.priority
        }


class MedicationRecommendations:
    """Sistema de recomendaciones basadas en medicamentos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, MedicationProfile] = {}  # user_id -> profile
    
    def create_medication_profile(self, user_id: str, medications: List[Dict],
                                 supplements: Optional[List[str]] = None,
                                 skin_related_medications: Optional[List[str]] = None) -> MedicationProfile:
        """Crea perfil de medicamentos"""
        profile = MedicationProfile(
            user_id=user_id,
            medications=medications,
            supplements=supplements or [],
            skin_related_medications=skin_related_medications or []
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_medication_recommendations(self, user_id: str) -> List[MedicationRecommendation]:
        """Obtiene recomendaciones basadas en medicamentos"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Verificar medicamentos que afectan la piel
        skin_affecting_meds = [
            "accutane", "isotretinoin", "retinoids", "antibiotics", "hormones",
            "birth_control", "steroids", "antihistamines"
        ]
        
        for med in profile.medications:
            med_name_lower = med.get("name", "").lower()
            med_type = med.get("type", "").lower()
            
            # Retinoides orales
            if "retinoid" in med_name_lower or "accutane" in med_name_lower or "isotretinoin" in med_name_lower:
                recommendations.append(MedicationRecommendation(
                    medication_factor="Retinoide Oral",
                    impact="negative",
                    recommendations=[
                        "Evita productos tópicos con retinol mientras tomas retinoides orales",
                        "Protección solar extrema es crítica (SPF 50+)",
                        "Hidratación intensa necesaria",
                        "Evita exfoliación agresiva"
                    ],
                    product_suggestions=[
                        {"category": "sunscreen", "type": "spf_50", "priority": 1, "reason": "Protección máxima necesaria"},
                        {"category": "moisturizer", "type": "barrier_repair", "priority": 1, "reason": "Reparar barrera dañada"}
                    ],
                    warnings=["No uses retinol tópico", "Consulta con dermatólogo antes de usar otros activos"],
                    priority=1
                ))
            
            # Antibióticos
            if "antibiotic" in med_type or "antibiotic" in med_name_lower:
                recommendations.append(MedicationRecommendation(
                    medication_factor="Antibióticos",
                    impact="neutral",
                    recommendations=[
                        "Los antibióticos pueden afectar el microbioma de la piel",
                        "Considera probióticos tópicos",
                        "Mantén rutina suave mientras tomas antibióticos"
                    ],
                    product_suggestions=[
                        {"category": "serum", "type": "probiotic", "reason": "Restaurar microbioma"}
                    ],
                    priority=2
                ))
            
            # Hormonas / Anticonceptivos
            if "hormone" in med_type or "birth_control" in med_name_lower or "contraceptive" in med_name_lower:
                recommendations.append(MedicationRecommendation(
                    medication_factor="Hormonas",
                    impact="neutral",
                    recommendations=[
                        "Las hormonas pueden causar cambios en la piel",
                        "Monitorea cambios en acné o pigmentación",
                        "Ajusta rutina según cambios observados"
                    ],
                    product_suggestions=[],
                    priority=3
                ))
            
            # Esteroides
            if "steroid" in med_name_lower or "corticosteroid" in med_name_lower:
                recommendations.append(MedicationRecommendation(
                    medication_factor="Esteroides",
                    impact="negative",
                    recommendations=[
                        "Los esteroides pueden hacer la piel más delgada y sensible",
                        "Evita productos agresivos",
                        "Hidratación y protección solar esenciales",
                        "Consulta con médico sobre uso tópico de productos"
                    ],
                    product_suggestions=[
                        {"category": "moisturizer", "type": "gentle", "priority": 1},
                        {"category": "sunscreen", "type": "mineral", "priority": 1}
                    ],
                    warnings=["Evita retinol y AHA/BHA mientras usas esteroides"],
                    priority=1
                ))
        
        # Verificar interacciones con productos comunes
        if any("retinoid" in m.get("name", "").lower() for m in profile.medications):
            recommendations.append(MedicationRecommendation(
                medication_factor="Interacción con Retinoides",
                impact="interaction",
                recommendations=[
                    "No combines retinoides orales con tópicos",
                    "Evita otros activos agresivos",
                    "Protección solar obligatoria"
                ],
                product_suggestions=[],
                warnings=["Interacción potencial con retinol tópico", "Consulta con dermatólogo"],
                priority=1
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations






