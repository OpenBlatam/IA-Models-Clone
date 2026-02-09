"""
Sistema de recomendaciones estacionales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Season(str, Enum):
    """Estación del año"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


@dataclass
class SeasonalRecommendation:
    """Recomendación estacional"""
    season: Season
    location: str
    recommendations: List[str]
    product_suggestions: List[Dict]
    routine_changes: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "season": self.season.value,
            "location": self.location,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "routine_changes": self.routine_changes,
            "warnings": self.warnings
        }


class SeasonalRecommendationsSystem:
    """Sistema de recomendaciones estacionales"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def get_seasonal_recommendations(self, season: Season, location: str,
                                   user_skin_type: str) -> SeasonalRecommendation:
        """Obtiene recomendaciones estacionales"""
        recommendations = []
        product_suggestions = []
        routine_changes = []
        warnings = []
        
        if season == Season.WINTER:
            recommendations.append("Invierno: Clima seco y frío")
            recommendations.append("Hidratación intensa es esencial")
            recommendations.append("Protege tu piel del viento frío")
            
            product_suggestions = [
                {"category": "moisturizer", "type": "rich_cream", "reason": "Hidratación profunda"},
                {"category": "cleanser", "type": "gentle", "reason": "No resecar la piel"},
                {"category": "serum", "type": "hyaluronic_acid", "reason": "Retención de humedad"}
            ]
            
            routine_changes = [
                "Usa cremas más ricas y nutritivas",
                "Evita duchas muy calientes",
                "Considera usar humidificador"
            ]
            
            if user_skin_type == "dry":
                warnings.append("Piel seca: Necesitas hidratación extra en invierno")
        
        elif season == Season.SUMMER:
            recommendations.append("Verano: Mayor exposición UV y calor")
            recommendations.append("Protección solar diaria es crítica")
            recommendations.append("Hidratación ligera y refrescante")
            
            product_suggestions = [
                {"category": "sunscreen", "type": "spf_50", "reason": "Protección máxima"},
                {"category": "moisturizer", "type": "light_gel", "reason": "No pesado en calor"},
                {"category": "serum", "type": "vitamin_c", "reason": "Protección antioxidante"}
            ]
            
            routine_changes = [
                "Aplica SPF cada 2 horas si estás al sol",
                "Usa productos oil-free",
                "Limpia más frecuentemente por el sudor"
            ]
            
            warnings.append("Índice UV alto: Protección solar esencial")
        
        elif season == Season.SPRING:
            recommendations.append("Primavera: Transición de invierno a verano")
            recommendations.append("Ajusta tu rutina gradualmente")
            recommendations.append("Prepara tu piel para mayor exposición solar")
            
            product_suggestions = [
                {"category": "moisturizer", "type": "medium", "reason": "Transición de crema pesada"},
                {"category": "exfoliant", "type": "gentle", "reason": "Renovación celular"},
                {"category": "sunscreen", "type": "spf_30", "reason": "Protección temprana"}
            ]
            
            routine_changes = [
                "Reduce gradualmente cremas pesadas",
                "Introduce protección solar",
                "Exfolia suavemente para renovar piel"
            ]
        
        elif season == Season.FALL:
            recommendations.append("Otoño: Preparación para invierno")
            recommendations.append("Fortalece la barrera cutánea")
            recommendations.append("Repara daños del verano")
            
            product_suggestions = [
                {"category": "moisturizer", "type": "barrier_repair", "reason": "Fortalece barrera"},
                {"category": "serum", "type": "niacinamide", "reason": "Reparación"},
                {"category": "treatment", "type": "retinol", "reason": "Renovación celular"}
            ]
            
            routine_changes = [
                "Aumenta hidratación gradualmente",
                "Repara daños del sol del verano",
                "Prepara para clima más seco"
            ]
        
        return SeasonalRecommendation(
            season=season,
            location=location,
            recommendations=recommendations,
            product_suggestions=product_suggestions,
            routine_changes=routine_changes,
            warnings=warnings
        )
    
    def get_current_season(self, location: str) -> Season:
        """Determina estación actual basada en ubicación"""
        # Lógica simplificada - en producción usaría datos de ubicación reales
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return Season.WINTER
        elif month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL






