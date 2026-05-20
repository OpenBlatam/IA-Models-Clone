"""
Personalized Recommendations - Sistema de recomendaciones personalizadas
=========================================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class PersonalizedRecommendations:
    """Sistema de recomendaciones personalizadas"""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.user_interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.item_similarities: Dict[str, Dict[str, float]] = {}
    
    def update_user_profile(self, user_id: str, preferences: Dict[str, Any]):
        """Actualiza perfil de usuario"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat()
            }
        
        self.user_profiles[user_id].update(preferences)
        self.user_profiles[user_id]["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Perfil actualizado para usuario: {user_id}")
    
    def record_interaction(self, user_id: str, item_id: str, item_type: str,
                         interaction_type: str, rating: Optional[float] = None):
        """Registra interacción de usuario"""
        interaction = {
            "item_id": item_id,
            "item_type": item_type,
            "interaction_type": interaction_type,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        self.user_interactions[user_id].append(interaction)
        
        # Mantener solo últimas 1000 interacciones
        if len(self.user_interactions[user_id]) > 1000:
            self.user_interactions[user_id] = self.user_interactions[user_id][-1000:]
    
    def get_recommendations(self, user_id: str, limit: int = 10,
                          item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones personalizadas"""
        user_interactions = self.user_interactions.get(user_id, [])
        user_profile = self.user_profiles.get(user_id, {})
        
        if not user_interactions:
            # Recomendaciones basadas en popularidad
            return self._get_popular_items(limit, item_type)
        
        # Recomendaciones basadas en colaboración (simplificado)
        recommendations = []
        
        # Analizar interacciones del usuario
        liked_items = [
            i["item_id"] for i in user_interactions
            if i.get("rating", 0) >= 4.0 or i["interaction_type"] == "like"
        ]
        
        # Encontrar items similares
        for item_id in liked_items[:5]:  # Top 5 items que le gustaron
            similar = self._find_similar_items(item_id, limit=2)
            recommendations.extend(similar)
        
        # Agregar recomendaciones basadas en perfil
        profile_based = self._get_profile_based_recommendations(user_profile, limit=3)
        recommendations.extend(profile_based)
        
        # Remover duplicados y items ya interactuados
        seen_items = {i["item_id"] for i in user_interactions}
        recommendations = [
            r for r in recommendations
            if r["item_id"] not in seen_items
        ]
        
        # Ordenar por score y limitar
        recommendations.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return recommendations[:limit]
    
    def _get_popular_items(self, limit: int, item_type: Optional[str]) -> List[Dict[str, Any]]:
        """Obtiene items populares"""
        # Simulado - en producción vendría de base de datos
        return [
            {
                "item_id": f"popular_{i}",
                "item_type": item_type or "prototype",
                "score": 0.8 - (i * 0.1),
                "reason": "popular"
            }
            for i in range(limit)
        ]
    
    def _find_similar_items(self, item_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Encuentra items similares"""
        # Simulado - en producción usaría algoritmos de similitud
        return [
            {
                "item_id": f"similar_{item_id}_{i}",
                "item_type": "prototype",
                "score": 0.7 - (i * 0.1),
                "reason": f"similar_to_{item_id}"
            }
            for i in range(limit)
        ]
    
    def _get_profile_based_recommendations(self, profile: Dict[str, Any],
                                          limit: int = 5) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones basadas en perfil"""
        recommendations = []
        
        preferred_types = profile.get("preferred_product_types", [])
        preferred_budget = profile.get("preferred_budget_range", "medium")
        
        for ptype in preferred_types[:2]:
            recommendations.append({
                "item_id": f"profile_{ptype}",
                "item_type": "prototype",
                "score": 0.75,
                "reason": f"matches_preference_{ptype}"
            })
        
        return recommendations[:limit]
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene perfil de usuario"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        interactions_count = len(self.user_interactions.get(user_id, []))
        
        return {
            **profile,
            "interactions_count": interactions_count,
            "recommendations_available": interactions_count > 0
        }




