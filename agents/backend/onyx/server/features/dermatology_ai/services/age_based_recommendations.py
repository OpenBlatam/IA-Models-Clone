"""
Sistema de recomendaciones basadas en edad
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgeBasedRecommendation:
    """Recomendación basada en edad"""
    age_range: str  # "20s", "30s", "40s", "50s+"
    recommendations: List[str]
    product_suggestions: List[Dict]
    routine_focus: List[str]
    concerns: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "age_range": self.age_range,
            "recommendations": self.recommendations,
            "product_suggestions": self.product_suggestions,
            "routine_focus": self.routine_focus,
            "concerns": self.concerns
        }


class AgeBasedRecommendations:
    """Sistema de recomendaciones basadas en edad"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def get_age_recommendations(self, age: int, skin_type: str) -> AgeBasedRecommendation:
        """Obtiene recomendaciones basadas en edad"""
        # Determinar rango de edad
        if age < 25:
            age_range = "20s"
        elif age < 35:
            age_range = "30s"
        elif age < 45:
            age_range = "40s"
        else:
            age_range = "50s+"
        
        recommendations = []
        product_suggestions = []
        routine_focus = []
        concerns = []
        
        if age_range == "20s":
            recommendations.append("Enfócate en prevención y protección solar")
            recommendations.append("Establece una rutina básica consistente")
            
            product_suggestions = [
                {"category": "sunscreen", "priority": 1, "reason": "Protección temprana es clave"},
                {"category": "cleanser", "priority": 1, "reason": "Limpieza suave diaria"},
                {"category": "moisturizer", "priority": 2, "reason": "Hidratación ligera"}
            ]
            
            routine_focus = ["Prevención", "Protección UV", "Hidratación básica"]
            concerns = ["Acné", "Piel grasa", "Poros visibles"]
        
        elif age_range == "30s":
            recommendations.append("Introduce antioxidantes y exfoliantes suaves")
            recommendations.append("Comienza con productos anti-envejecimiento preventivos")
            
            product_suggestions = [
                {"category": "serum", "type": "vitamin_c", "priority": 1, "reason": "Antioxidante y protección"},
                {"category": "exfoliant", "type": "gentle_aha", "priority": 2, "reason": "Renovación celular"},
                {"category": "eye_cream", "priority": 2, "reason": "Prevención de líneas finas"}
            ]
            
            routine_focus = ["Prevención avanzada", "Antioxidantes", "Exfoliación suave"]
            concerns = ["Primeras líneas finas", "Pérdida de brillo", "Textura irregular"]
        
        elif age_range == "40s":
            recommendations.append("Intensifica productos anti-envejecimiento")
            recommendations.append("Considera retinol y péptidos")
            
            product_suggestions = [
                {"category": "serum", "type": "retinol", "priority": 1, "reason": "Renovación celular profunda"},
                {"category": "serum", "type": "peptides", "priority": 1, "reason": "Firmeza y elasticidad"},
                {"category": "moisturizer", "type": "rich", "priority": 1, "reason": "Hidratación intensa"}
            ]
            
            routine_focus = ["Anti-envejecimiento", "Firmeza", "Hidratación profunda"]
            concerns = ["Arrugas", "Pérdida de firmeza", "Sequedad"]
        
        else:  # 50s+
            recommendations.append("Enfócate en hidratación intensa y reparación")
            recommendations.append("Productos ricos en nutrientes y humectantes")
            
            product_suggestions = [
                {"category": "moisturizer", "type": "barrier_repair", "priority": 1, "reason": "Reparación de barrera"},
                {"category": "serum", "type": "growth_factors", "priority": 1, "reason": "Regeneración celular"},
                {"category": "treatment", "type": "intensive_night", "priority": 1, "reason": "Reparación nocturna"}
            ]
            
            routine_focus = ["Reparación", "Hidratación intensa", "Nutrición"]
            concerns = ["Arrugas profundas", "Pérdida de volumen", "Piel delgada"]
        
        return AgeBasedRecommendation(
            age_range=age_range,
            recommendations=recommendations,
            product_suggestions=product_suggestions,
            routine_focus=routine_focus,
            concerns=concerns
        )






