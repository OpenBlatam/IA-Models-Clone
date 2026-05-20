"""
Sistema de análisis de edad aparente
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class AgeAnalysis:
    """Análisis de edad aparente"""
    id: str
    user_id: str
    image_url: str
    chronological_age: int
    apparent_age: float
    age_difference: float
    factors: Dict
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.age_difference = self.apparent_age - self.chronological_age
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_url": self.image_url,
            "chronological_age": self.chronological_age,
            "apparent_age": self.apparent_age,
            "age_difference": self.age_difference,
            "factors": self.factors,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class AgeAnalysisSystem:
    """Sistema de análisis de edad aparente"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.analyses: Dict[str, List[AgeAnalysis]] = {}  # user_id -> [analyses]
    
    def analyze_age(self, user_id: str, image_url: str, chronological_age: int,
                   skin_analysis: Dict) -> AgeAnalysis:
        """Analiza edad aparente"""
        # Simulación de análisis - en producción usaría modelos de ML
        
        scores = skin_analysis.get("quality_scores", {})
        overall_score = scores.get("overall_score", 0)
        texture_score = scores.get("texture_score", 0)
        hydration_score = scores.get("hydration_score", 0)
        
        # Calcular edad aparente basada en scores
        # Score más alto = apariencia más joven
        age_reduction_factor = (overall_score / 100) * 5  # Máximo 5 años de reducción
        apparent_age = max(chronological_age - age_reduction_factor, chronological_age - 10)
        
        # Factores que afectan la edad aparente
        factors = {
            "texture_impact": "Bueno" if texture_score > 70 else "Necesita mejora",
            "hydration_impact": "Óptimo" if hydration_score > 60 else "Bajo",
            "overall_condition": "Excelente" if overall_score > 80 else "Buena" if overall_score > 60 else "Regular"
        }
        
        # Recomendaciones
        recommendations = []
        
        if apparent_age > chronological_age:
            recommendations.append("Tu piel aparenta más edad. Considera productos anti-envejecimiento.")
        elif apparent_age < chronological_age - 3:
            recommendations.append("¡Excelente! Tu piel aparenta menos edad. Mantén tu rutina.")
        else:
            recommendations.append("Tu edad aparente coincide con tu edad cronológica.")
        
        if texture_score < 60:
            recommendations.append("Mejora la textura para reducir edad aparente.")
        
        if hydration_score < 50:
            recommendations.append("Aumenta la hidratación para una apariencia más joven.")
        
        analysis = AgeAnalysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            chronological_age=chronological_age,
            apparent_age=apparent_age,
            factors=factors,
            recommendations=recommendations
        )
        
        if user_id not in self.analyses:
            self.analyses[user_id] = []
        
        self.analyses[user_id].append(analysis)
        return analysis
    
    def get_user_analyses(self, user_id: str) -> List[AgeAnalysis]:
        """Obtiene análisis del usuario"""
        return self.analyses.get(user_id, [])






