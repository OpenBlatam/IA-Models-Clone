"""
Sistema de recomendaciones basadas en genética
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GeneticProfile:
    """Perfil genético"""
    user_id: str
    skin_type_genetic: str  # Basado en genética
    aging_tendency: str  # "slow", "normal", "fast"
    collagen_production: str  # "high", "normal", "low"
    melanin_production: str  # "high", "normal", "low"
    sensitivity_tendency: str  # "low", "medium", "high"
    hydration_capacity: str  # "high", "normal", "low"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "skin_type_genetic": self.skin_type_genetic,
            "aging_tendency": self.aging_tendency,
            "collagen_production": self.collagen_production,
            "melanin_production": self.melanin_production,
            "sensitivity_tendency": self.sensitivity_tendency,
            "hydration_capacity": self.hydration_capacity
        }


@dataclass
class GeneticRecommendation:
    """Recomendación basada en genética"""
    genetic_factor: str
    impact: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "genetic_factor": self.genetic_factor,
            "impact": self.impact,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "priority": self.priority
        }


class GeneticRecommendations:
    """Sistema de recomendaciones basadas en genética"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.profiles: Dict[str, GeneticProfile] = {}  # user_id -> profile
    
    def create_genetic_profile(self, user_id: str, skin_type_genetic: str,
                              aging_tendency: str, collagen_production: str,
                              melanin_production: str, sensitivity_tendency: str,
                              hydration_capacity: str) -> GeneticProfile:
        """Crea perfil genético"""
        profile = GeneticProfile(
            user_id=user_id,
            skin_type_genetic=skin_type_genetic,
            aging_tendency=aging_tendency,
            collagen_production=collagen_production,
            melanin_production=melanin_production,
            sensitivity_tendency=sensitivity_tendency,
            hydration_capacity=hydration_capacity
        )
        
        self.profiles[user_id] = profile
        return profile
    
    def get_genetic_recommendations(self, user_id: str) -> List[GeneticRecommendation]:
        """Obtiene recomendaciones basadas en genética"""
        profile = self.profiles.get(user_id)
        
        if not profile:
            return []
        
        recommendations = []
        
        # Recomendaciones por tendencia de envejecimiento
        if profile.aging_tendency == "fast":
            recommendations.append(GeneticRecommendation(
                genetic_factor="Envejecimiento Acelerado",
                impact="high",
                recommendations=[
                    "Inicia productos anti-envejecimiento temprano",
                    "Usa retinol desde los 20s",
                    "Protección solar diaria es crítica"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "retinol", "priority": 1},
                    {"category": "sunscreen", "type": "spf_50", "priority": 1}
                ],
                priority=1
            ))
        
        # Recomendaciones por producción de colágeno
        if profile.collagen_production == "low":
            recommendations.append(GeneticRecommendation(
                genetic_factor="Producción Baja de Colágeno",
                impact="high",
                recommendations=[
                    "Usa productos con péptidos",
                    "Considera tratamientos con factores de crecimiento",
                    "Evita productos que degraden colágeno"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "peptides", "priority": 1},
                    {"category": "serum", "type": "growth_factors", "priority": 2}
                ],
                priority=1
            ))
        
        # Recomendaciones por producción de melanina
        if profile.melanin_production == "high":
            recommendations.append(GeneticRecommendation(
                genetic_factor="Alta Producción de Melanina",
                impact="medium",
                recommendations=[
                    "Mayor riesgo de hiperpigmentación",
                    "Protección solar intensa es esencial",
                    "Usa productos con ingredientes para uniformizar tono"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "vitamin_c", "priority": 1},
                    {"category": "treatment", "type": "brightening", "priority": 2}
                ],
                priority=2
            ))
        
        # Recomendaciones por sensibilidad
        if profile.sensitivity_tendency == "high":
            recommendations.append(GeneticRecommendation(
                genetic_factor="Alta Sensibilidad",
                impact="high",
                recommendations=[
                    "Evita productos agresivos",
                    "Usa productos hipoalergénicos",
                    "Haz pruebas de parche siempre",
                    "Evita fragancias y alcohol"
                ],
                product_suggestions=[
                    {"category": "cleanser", "type": "gentle", "priority": 1},
                    {"category": "moisturizer", "type": "barrier_repair", "priority": 1}
                ],
                priority=1
            ))
        
        # Recomendaciones por capacidad de hidratación
        if profile.hydration_capacity == "low":
            recommendations.append(GeneticRecommendation(
                genetic_factor="Baja Capacidad de Hidratación",
                impact="medium",
                recommendations=[
                    "Hidratación intensa necesaria",
                    "Usa productos con múltiples humectantes",
                    "Evita productos astringentes"
                ],
                product_suggestions=[
                    {"category": "serum", "type": "hyaluronic_acid", "priority": 1},
                    {"category": "moisturizer", "type": "rich", "priority": 1}
                ],
                priority=2
            ))
        
        recommendations.sort(key=lambda x: x.priority)
        return recommendations






