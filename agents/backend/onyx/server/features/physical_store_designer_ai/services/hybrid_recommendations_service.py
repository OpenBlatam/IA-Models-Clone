"""
Hybrid Recommendations Service - Sistema de recomendaciones híbridas
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..services.collaborative_filtering_service import CollaborativeFilteringService
from ..services.ml_recommendations_service import MLRecommendationsService

logger = logging.getLogger(__name__)


class HybridRecommendationsService:
    """Servicio para recomendaciones híbridas"""
    
    def __init__(
        self,
        collaborative_service: Optional[CollaborativeFilteringService] = None,
        ml_service: Optional[MLRecommendationsService] = None
    ):
        self.collaborative_service = collaborative_service or CollaborativeFilteringService()
        self.ml_service = ml_service or MLRecommendationsService()
        self.hybrid_recommendations: Dict[str, List[Dict[str, Any]]] = {}
    
    async def generate_hybrid_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generar recomendaciones híbridas"""
        
        # Obtener recomendaciones colaborativas
        collaborative_recs = self.collaborative_service.recommend_items(user_id, limit=10)
        
        # Obtener recomendaciones ML
        ml_recs = await self.ml_service.generate_personalized_recommendations(
            user_id, [], context
        )
        
        # Combinar y rankear
        hybrid_recs = self._combine_recommendations(collaborative_recs, ml_recs)
        
        result = {
            "user_id": user_id,
            "recommendations": hybrid_recs,
            "collaborative_count": len(collaborative_recs),
            "ml_count": len(ml_recs.get("style_recommendations", [])),
            "hybrid_count": len(hybrid_recs),
            "generated_at": datetime.now().isoformat()
        }
        
        self.hybrid_recommendations[user_id] = hybrid_recs
        
        return result
    
    def _combine_recommendations(
        self,
        collaborative: List[Dict[str, Any]],
        ml: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combinar recomendaciones"""
        
        combined = []
        
        # Agregar recomendaciones colaborativas con peso
        for rec in collaborative:
            combined.append({
                **rec,
                "source": "collaborative",
                "weight": 0.6
            })
        
        # Agregar recomendaciones ML con peso
        ml_styles = ml.get("style_recommendations", [])
        for style_rec in ml_styles:
            combined.append({
                "item_id": f"style_{style_rec.get('style', 'unknown')}",
                "score": style_rec.get("confidence", 0.5) * 0.4,
                "source": "ml",
                "weight": 0.4,
                "reason": style_rec.get("reason", "")
            })
        
        # Ordenar por score combinado
        combined.sort(key=lambda x: x.get("score", 0) * x.get("weight", 1), reverse=True)
        
        return combined[:20]  # Top 20
    
    async def explain_recommendation(
        self,
        user_id: str,
        item_id: str
    ) -> Dict[str, Any]:
        """Explicar recomendación"""
        
        user_recs = self.hybrid_recommendations.get(user_id, [])
        recommendation = next((r for r in user_recs if r.get("item_id") == item_id), None)
        
        if not recommendation:
            return {
                "item_id": item_id,
                "message": "Recomendación no encontrada"
            }
        
        explanation = {
            "item_id": item_id,
            "source": recommendation.get("source"),
            "score": recommendation.get("score"),
            "weight": recommendation.get("weight"),
            "reasons": [
                recommendation.get("reason", "Basado en preferencias similares")
            ]
        }
        
        # Agregar razones específicas según fuente
        if recommendation.get("source") == "collaborative":
            explanation["reasons"].append("Usuarios similares también eligieron esto")
        elif recommendation.get("source") == "ml":
            explanation["reasons"].append("Basado en tu historial y perfil")
        
        return explanation




