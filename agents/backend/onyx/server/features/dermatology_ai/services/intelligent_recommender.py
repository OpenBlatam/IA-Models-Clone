"""
Sistema de recomendaciones inteligentes
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class Recommendation:
    """Recomendación inteligente"""
    id: str
    type: str  # "product", "routine", "tip", etc.
    title: str
    description: str
    confidence: float
    reasoning: str
    priority: int
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "created_at": self.created_at
        }


class IntelligentRecommender:
    """Sistema de recomendaciones inteligentes"""
    
    def __init__(self):
        """Inicializa el recomendador"""
        self.user_preferences: Dict[str, Dict] = {}
        self.recommendation_history: Dict[str, List[str]] = {}  # user_id -> [recommendation_ids]
        self.feedback_scores: Dict[str, float] = {}  # recommendation_id -> score
    
    def generate_intelligent_recommendations(self, user_id: str,
                                           analysis_result: Dict,
                                           user_history: Optional[List[Dict]] = None) -> List[Recommendation]:
        """
        Genera recomendaciones inteligentes
        
        Args:
            user_id: ID del usuario
            analysis_result: Resultado del análisis
            user_history: Historial del usuario (opcional)
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar resultados
        overall_score = analysis_result.get("quality_scores", {}).get("overall_score", 0)
        conditions = analysis_result.get("conditions", [])
        skin_type = analysis_result.get("skin_type", "normal")
        
        # Recomendación basada en score
        if overall_score < 50:
            recommendations.append(self._create_recommendation(
                type="routine",
                title="Rutina Intensiva Recomendada",
                description="Tu piel necesita atención especial. Te recomendamos una rutina intensiva.",
                confidence=0.9,
                reasoning=f"Score bajo ({overall_score:.1f}) indica necesidad de cuidado intensivo",
                priority=1
            ))
        
        # Recomendaciones basadas en condiciones
        for condition in conditions:
            condition_name = condition.get("name", "")
            severity = condition.get("severity", "mild")
            
            if condition_name == "acne":
                recommendations.append(self._create_recommendation(
                    type="product",
                    title="Tratamiento para Acné",
                    description="Productos específicos para tratar el acné",
                    confidence=0.85,
                    reasoning=f"Condición detectada: {condition_name} ({severity})",
                    priority=2
                ))
        
        # Recomendaciones personalizadas basadas en historial
        if user_history:
            recommendations.extend(self._personalize_from_history(user_id, user_history, analysis_result))
        
        # Ordenar por prioridad y confianza
        recommendations.sort(key=lambda r: (r.priority, -r.confidence))
        
        return recommendations
    
    def _create_recommendation(self, type: str, title: str, description: str,
                              confidence: float, reasoning: str,
                              priority: int) -> Recommendation:
        """Crea una recomendación"""
        import uuid
        return Recommendation(
            id=str(uuid.uuid4()),
            type=type,
            title=title,
            description=description,
            confidence=confidence,
            reasoning=reasoning,
            priority=priority
        )
    
    def _personalize_from_history(self, user_id: str, history: List[Dict],
                                 current_analysis: Dict) -> List[Recommendation]:
        """Personaliza recomendaciones basadas en historial"""
        recommendations = []
        
        # Analizar tendencias
        scores = [h.get("quality_scores", {}).get("overall_score", 0) for h in history]
        
        if len(scores) >= 2:
            trend = scores[-1] - scores[0]
            
            if trend < -5:
                recommendations.append(self._create_recommendation(
                    type="tip",
                    title="Mejora tu Rutina",
                    description="Tu piel ha empeorado. Revisa tu rutina actual.",
                    confidence=0.8,
                    reasoning=f"Tendencia negativa detectada: {trend:.1f} puntos",
                    priority=1
                ))
            elif trend > 5:
                recommendations.append(self._create_recommendation(
                    type="tip",
                    title="¡Sigue Así!",
                    description="Tu piel está mejorando. Mantén tu rutina actual.",
                    confidence=0.9,
                    reasoning=f"Tendencia positiva: +{trend:.1f} puntos",
                    priority=3
                ))
        
        return recommendations
    
    def record_feedback(self, recommendation_id: str, user_id: str, rating: float):
        """
        Registra feedback de una recomendación
        
        Args:
            recommendation_id: ID de la recomendación
            user_id: ID del usuario
            rating: Rating (0-5)
        """
        self.feedback_scores[recommendation_id] = rating
        
        if user_id not in self.recommendation_history:
            self.recommendation_history[user_id] = []
        self.recommendation_history[user_id].append(recommendation_id)
    
    def get_recommendation_stats(self) -> Dict:
        """Obtiene estadísticas de recomendaciones"""
        if not self.feedback_scores:
            return {
                "total_recommendations": 0,
                "average_rating": 0,
                "total_feedback": 0
            }
        
        ratings = list(self.feedback_scores.values())
        
        return {
            "total_recommendations": len(self.recommendation_history),
            "average_rating": statistics.mean(ratings),
            "total_feedback": len(ratings),
            "rating_distribution": {
                "5_stars": sum(1 for r in ratings if r >= 4.5),
                "4_stars": sum(1 for r in ratings if 3.5 <= r < 4.5),
                "3_stars": sum(1 for r in ratings if 2.5 <= r < 3.5),
                "2_stars": sum(1 for r in ratings if 1.5 <= r < 2.5),
                "1_star": sum(1 for r in ratings if r < 1.5)
            }
        }






