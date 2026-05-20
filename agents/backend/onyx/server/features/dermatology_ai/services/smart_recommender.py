"""
Sistema de recomendaciones inteligente mejorado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class SmartRecommendation:
    """Recomendación inteligente mejorada"""
    id: str
    category: str  # "product", "routine", "lifestyle", "treatment"
    title: str
    description: str
    priority: int  # 1-5, 1 = más importante
    confidence: float
    reasoning: str
    expected_impact: str  # "high", "medium", "low"
    time_to_see_results: str  # "1 week", "2 weeks", etc.
    cost_estimate: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_impact": self.expected_impact,
            "time_to_see_results": self.time_to_see_results,
            "cost_estimate": self.cost_estimate,
            "created_at": self.created_at
        }


class SmartRecommender:
    """Sistema de recomendaciones inteligente mejorado"""
    
    def __init__(self):
        """Inicializa el recomendador"""
        self.recommendation_templates: Dict[str, Dict] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Inicializa plantillas de recomendaciones"""
        self.recommendation_templates = {
            "low_hydration": {
                "category": "product",
                "title": "Hidratante Intensivo",
                "description": "Tu piel necesita más hidratación. Recomendamos un hidratante con ácido hialurónico.",
                "priority": 1,
                "expected_impact": "high",
                "time_to_see_results": "1-2 weeks"
            },
            "high_sebum": {
                "category": "routine",
                "title": "Rutina para Piel Grasa",
                "description": "Ajusta tu rutina para controlar el exceso de sebo.",
                "priority": 2,
                "expected_impact": "medium",
                "time_to_see_results": "2-3 weeks"
            },
            "sun_damage": {
                "category": "lifestyle",
                "title": "Protección Solar Diaria",
                "description": "Protege tu piel del sol con SPF 30+ diariamente.",
                "priority": 1,
                "expected_impact": "high",
                "time_to_see_results": "Ongoing"
            }
        }
    
    def generate_smart_recommendations(self, user_id: str,
                                      analysis_result: Dict,
                                      user_profile: Optional[Dict] = None,
                                      progress_data: Optional[Dict] = None) -> List[SmartRecommendation]:
        """
        Genera recomendaciones inteligentes
        
        Args:
            user_id: ID del usuario
            analysis_result: Resultado del análisis
            user_profile: Perfil del usuario (opcional)
            progress_data: Datos de progreso (opcional)
            
        Returns:
            Lista de recomendaciones inteligentes
        """
        recommendations = []
        
        # Analizar scores
        scores = analysis_result.get("quality_scores", {})
        overall_score = scores.get("overall_score", 0)
        hydration = scores.get("hydration_score", 0)
        texture = scores.get("texture_score", 0)
        sebum = scores.get("sebum_level", 0)
        
        # Recomendación 1: Hidratación baja
        if hydration < 50:
            rec = self._create_recommendation(
                template_key="low_hydration",
                confidence=0.9,
                reasoning=f"Hidratación baja detectada: {hydration:.1f}%"
            )
            recommendations.append(rec)
        
        # Recomendación 2: Exceso de sebo
        if sebum > 0.7:
            rec = self._create_recommendation(
                template_key="high_sebum",
                confidence=0.85,
                reasoning=f"Exceso de sebo detectado: {sebum:.2f}"
            )
            recommendations.append(rec)
        
        # Recomendación 3: Score general bajo
        if overall_score < 60:
            rec = self._create_recommendation(
                category="routine",
                title="Rutina Completa de Cuidado",
                description="Tu piel necesita atención completa. Te recomendamos una rutina estructurada.",
                priority=1,
                confidence=0.8,
                reasoning=f"Score general bajo: {overall_score:.1f}",
                expected_impact="high",
                time_to_see_results="3-4 weeks"
            )
            recommendations.append(rec)
        
        # Recomendaciones basadas en progreso
        if progress_data:
            progress_recs = self._generate_progress_based_recommendations(progress_data)
            recommendations.extend(progress_recs)
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda r: r.priority)
        
        return recommendations
    
    def _create_recommendation(self, template_key: Optional[str] = None,
                             category: Optional[str] = None,
                             title: Optional[str] = None,
                             description: Optional[str] = None,
                             priority: int = 3,
                             confidence: float = 0.7,
                             reasoning: str = "",
                             expected_impact: str = "medium",
                             time_to_see_results: str = "2-3 weeks") -> SmartRecommendation:
        """Crea una recomendación"""
        import uuid
        
        if template_key and template_key in self.recommendation_templates:
            template = self.recommendation_templates[template_key]
            return SmartRecommendation(
                id=str(uuid.uuid4()),
                category=template["category"],
                title=template["title"],
                description=template["description"],
                priority=template.get("priority", priority),
                confidence=confidence,
                reasoning=reasoning,
                expected_impact=template.get("expected_impact", expected_impact),
                time_to_see_results=template.get("time_to_see_results", time_to_see_results)
            )
        else:
            return SmartRecommendation(
                id=str(uuid.uuid4()),
                category=category or "general",
                title=title or "Recomendación",
                description=description or "",
                priority=priority,
                confidence=confidence,
                reasoning=reasoning,
                expected_impact=expected_impact,
                time_to_see_results=time_to_see_results
            )
    
    def _generate_progress_based_recommendations(self, progress_data: Dict) -> List[SmartRecommendation]:
        """Genera recomendaciones basadas en progreso"""
        recommendations = []
        
        summary = progress_data.get("summary", {})
        overall_trend = summary.get("overall_trend", "stable")
        
        if overall_trend == "declining":
            rec = self._create_recommendation(
                category="routine",
                title="Revisa tu Rutina",
                description="Tu piel está empeorando. Considera revisar y ajustar tu rutina actual.",
                priority=1,
                confidence=0.75,
                reasoning="Tendencia negativa detectada en el progreso",
                expected_impact="high",
                time_to_see_results="2-3 weeks"
            )
            recommendations.append(rec)
        
        return recommendations






