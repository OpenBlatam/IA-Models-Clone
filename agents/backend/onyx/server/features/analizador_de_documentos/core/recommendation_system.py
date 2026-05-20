"""
Sistema de Recomendaciones Inteligentes
========================================

Sistema para generar recomendaciones personalizadas basadas en análisis.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Tipos de recomendación"""
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    ACTION = "action"


@dataclass
class Recommendation:
    """Recomendación"""
    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    priority: int  # 1-10, mayor = más importante
    confidence: float  # 0.0-1.0
    data: Dict[str, Any]
    created_at: str


class RecommendationSystem:
    """
    Sistema de recomendaciones
    
    Proporciona:
    - Recomendaciones personalizadas
    - Scoring de recomendaciones
    - Historial de recomendaciones
    - Filtrado por tipo y prioridad
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.recommendations: List[Recommendation] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        logger.info("RecommendationSystem inicializado")
    
    def generate_recommendation(
        self,
        analysis_result: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Generar recomendaciones basadas en análisis
        
        Args:
            analysis_result: Resultado de análisis
            user_id: ID de usuario (opcional)
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar resultado y generar recomendaciones
        if "sentiment" in analysis_result:
            sentiment = analysis_result["sentiment"]
            negative_score = sentiment.get("negative", 0)
            
            if negative_score > 0.7:
                recommendations.append(Recommendation(
                    recommendation_id=f"rec_{datetime.now().timestamp()}",
                    type=RecommendationType.ACTION,
                    title="Contenido altamente negativo detectado",
                    description="Se detectó un alto nivel de sentimiento negativo. Considera revisar el contenido.",
                    priority=9,
                    confidence=0.85,
                    data={"sentiment_score": negative_score},
                    created_at=datetime.now().isoformat()
                ))
        
        if "keywords" in analysis_result:
            keywords = analysis_result.get("keywords", [])
            if len(keywords) < 5:
                recommendations.append(Recommendation(
                    recommendation_id=f"rec_{datetime.now().timestamp()}",
                    type=RecommendationType.OPTIMIZATION,
                    title="Pocas palabras clave identificadas",
                    description="El documento tiene pocas palabras clave. Considera agregar más contenido relevante.",
                    priority=5,
                    confidence=0.7,
                    data={"keyword_count": len(keywords)},
                    created_at=datetime.now().isoformat()
                ))
        
        if "classification" in analysis_result:
            classification = analysis_result["classification"]
            max_confidence = max(classification.values()) if classification else 0
            
            if max_confidence < 0.5:
                recommendations.append(Recommendation(
                    recommendation_id=f"rec_{datetime.now().timestamp()}",
                    type=RecommendationType.ANALYSIS,
                    title="Clasificación con baja confianza",
                    description="La clasificación tiene baja confianza. Considera usar análisis adicionales.",
                    priority=7,
                    confidence=0.8,
                    data={"max_confidence": max_confidence},
                    created_at=datetime.now().isoformat()
                ))
        
        # Guardar recomendaciones
        self.recommendations.extend(recommendations)
        
        # Mantener solo últimos 10000 recomendaciones
        if len(self.recommendations) > 10000:
            self.recommendations = self.recommendations[-10000:]
        
        return recommendations
    
    def get_recommendations(
        self,
        user_id: Optional[str] = None,
        recommendation_type: Optional[RecommendationType] = None,
        min_priority: int = 1,
        limit: int = 10
    ) -> List[Recommendation]:
        """
        Obtener recomendaciones
        
        Args:
            user_id: ID de usuario
            recommendation_type: Tipo de recomendación
            min_priority: Prioridad mínima
            limit: Límite de resultados
        
        Returns:
            Lista de recomendaciones
        """
        filtered = self.recommendations
        
        if recommendation_type:
            filtered = [r for r in filtered if r.type == recommendation_type]
        
        filtered = [r for r in filtered if r.priority >= min_priority]
        
        # Ordenar por prioridad y confianza
        filtered.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return filtered[:limit]
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de recomendaciones"""
        if not self.recommendations:
            return {"total": 0}
        
        by_type = {}
        for rec in self.recommendations:
            rec_type = rec.type.value
            by_type[rec_type] = by_type.get(rec_type, 0) + 1
        
        avg_priority = sum(r.priority for r in self.recommendations) / len(self.recommendations)
        avg_confidence = sum(r.confidence for r in self.recommendations) / len(self.recommendations)
        
        return {
            "total": len(self.recommendations),
            "by_type": by_type,
            "avg_priority": avg_priority,
            "avg_confidence": avg_confidence
        }


# Instancia global
_recommendation_system: Optional[RecommendationSystem] = None


def get_recommendation_system() -> RecommendationSystem:
    """Obtener instancia global del sistema"""
    global _recommendation_system
    if _recommendation_system is None:
        _recommendation_system = RecommendationSystem()
    return _recommendation_system
















