"""
Motor de Recomendaciones Inteligentes

Proporciona:
- Recomendaciones basadas en contenido
- Recomendaciones colaborativas
- Recomendaciones híbridas
- Personalización por usuario
- Trending y popular
"""

import logging
import math
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Motor de recomendaciones"""
    
    def __init__(self):
        self.user_interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.item_features: Dict[str, Dict[str, Any]] = {}
        self.item_popularity: Counter = Counter()
        logger.info("RecommendationEngine initialized")
    
    def record_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str = "view",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra una interacción usuario-item
        
        Args:
            user_id: ID del usuario
            item_id: ID del item
            interaction_type: Tipo de interacción (view, like, play, share)
            metadata: Metadatos adicionales
        """
        interaction = {
            "item_id": item_id,
            "type": interaction_type,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.user_interactions[user_id].append(interaction)
        self.item_popularity[item_id] += 1
    
    def set_item_features(self, item_id: str, features: Dict[str, Any]):
        """Establece características de un item"""
        self.item_features[item_id] = features
    
    def get_content_based_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Recomendaciones basadas en contenido
        
        Args:
            user_id: ID del usuario
            limit: Número de recomendaciones
            min_similarity: Similitud mínima
        
        Returns:
            Lista de items recomendados con scores
        """
        # Obtener items que el usuario ha interactuado
        user_items = {
            interaction["item_id"]
            for interaction in self.user_interactions.get(user_id, [])
        }
        
        if not user_items:
            return []
        
        # Obtener características de los items del usuario
        user_features = self._aggregate_user_features(user_id, user_items)
        
        # Calcular similitud con otros items
        recommendations = []
        for item_id, item_features in self.item_features.items():
            if item_id in user_items:
                continue  # Ya interactuó con este item
            
            similarity = self._calculate_cosine_similarity(
                user_features,
                item_features
            )
            
            if similarity >= min_similarity:
                recommendations.append({
                    "item_id": item_id,
                    "score": similarity,
                    "type": "content_based"
                })
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]
    
    def get_collaborative_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        min_users: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Recomendaciones colaborativas
        
        Args:
            user_id: ID del usuario
            limit: Número de recomendaciones
            min_users: Mínimo de usuarios que deben haber interactuado
        
        Returns:
            Lista de items recomendados
        """
        # Obtener items del usuario
        user_items = {
            interaction["item_id"]
            for interaction in self.user_interactions.get(user_id, [])
        }
        
        if not user_items:
            return []
        
        # Encontrar usuarios similares
        similar_users = self._find_similar_users(user_id, user_items)
        
        # Obtener items de usuarios similares
        recommended_items = defaultdict(float)
        
        for similar_user_id, similarity in similar_users.items():
            similar_user_items = {
                interaction["item_id"]
                for interaction in self.user_interactions.get(similar_user_id, [])
            }
            
            # Items que el usuario similar tiene pero el usuario actual no
            new_items = similar_user_items - user_items
            
            for item_id in new_items:
                recommended_items[item_id] += similarity
        
        # Filtrar items con suficiente soporte
        recommendations = []
        for item_id, score in recommended_items.items():
            # Contar cuántos usuarios similares tienen este item
            user_count = sum(
                1 for similar_user_id in similar_users.keys()
                if item_id in {
                    i["item_id"]
                    for i in self.user_interactions.get(similar_user_id, [])
                }
            )
            
            if user_count >= min_users:
                recommendations.append({
                    "item_id": item_id,
                    "score": score / user_count,  # Normalizar
                    "type": "collaborative",
                    "user_support": user_count
                })
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]
    
    def get_hybrid_recommendations(
        self,
        user_id: str,
        limit: int = 10,
        content_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recomendaciones híbridas (contenido + colaborativo)
        
        Args:
            user_id: ID del usuario
            limit: Número de recomendaciones
            content_weight: Peso de recomendaciones basadas en contenido (0-1)
        
        Returns:
            Lista de items recomendados
        """
        # Obtener ambos tipos de recomendaciones
        content_recs = {
            rec["item_id"]: rec["score"]
            for rec in self.get_content_based_recommendations(user_id, limit=limit * 2)
        }
        
        collab_recs = {
            rec["item_id"]: rec["score"]
            for rec in self.get_collaborative_recommendations(user_id, limit=limit * 2)
        }
        
        # Combinar scores
        combined = defaultdict(float)
        all_items = set(content_recs.keys()) | set(collab_recs.keys())
        
        for item_id in all_items:
            content_score = content_recs.get(item_id, 0)
            collab_score = collab_recs.get(item_id, 0)
            
            # Normalizar scores
            if content_score > 0:
                content_score = (content_score - 0.3) / 0.7  # Normalizar a 0-1
            if collab_score > 0:
                collab_score = min(collab_score, 1.0)
            
            combined[item_id] = (
                content_weight * content_score +
                (1 - content_weight) * collab_score
            )
        
        recommendations = [
            {
                "item_id": item_id,
                "score": score,
                "type": "hybrid",
                "content_score": content_recs.get(item_id, 0),
                "collab_score": collab_recs.get(item_id, 0)
            }
            for item_id, score in combined.items()
            if score > 0
        ]
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:limit]
    
    def get_trending(
        self,
        limit: int = 10,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Obtiene items trending
        
        Args:
            limit: Número de items
            hours: Ventana de tiempo en horas
        
        Returns:
            Lista de items trending
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Contar interacciones recientes
        recent_interactions = Counter()
        for user_id, interactions in self.user_interactions.items():
            for interaction in interactions:
                if interaction["timestamp"] >= cutoff_time:
                    recent_interactions[interaction["item_id"]] += 1
        
        trending = [
            {
                "item_id": item_id,
                "score": count,
                "type": "trending"
            }
            for item_id, count in recent_interactions.most_common(limit)
        ]
        
        return trending
    
    def get_popular(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtiene items populares
        
        Args:
            limit: Número de items
        
        Returns:
            Lista de items populares
        """
        popular = [
            {
                "item_id": item_id,
                "score": count,
                "type": "popular"
            }
            for item_id, count in self.item_popularity.most_common(limit)
        ]
        
        return popular
    
    def _aggregate_user_features(
        self,
        user_id: str,
        user_items: set
    ) -> Dict[str, float]:
        """Agrega características de los items del usuario"""
        aggregated = defaultdict(float)
        count = 0
        
        for item_id in user_items:
            if item_id in self.item_features:
                features = self.item_features[item_id]
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        aggregated[key] += value
                count += 1
        
        # Promediar
        if count > 0:
            for key in aggregated:
                aggregated[key] /= count
        
        return dict(aggregated)
    
    def _calculate_cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """Calcula similitud coseno entre dos vectores"""
        # Obtener todas las claves
        keys = set(vec1.keys()) | set(vec2.keys())
        
        if not keys:
            return 0.0
        
        # Calcular producto punto y magnitudes
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
        magnitude1 = math.sqrt(sum(vec1.get(k, 0) ** 2 for k in keys))
        magnitude2 = math.sqrt(sum(vec2.get(k, 0) ** 2 for k in keys))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _find_similar_users(
        self,
        user_id: str,
        user_items: set
    ) -> Dict[str, float]:
        """Encuentra usuarios similares"""
        similar_users = {}
        
        for other_user_id, interactions in self.user_interactions.items():
            if other_user_id == user_id:
                continue
            
            other_items = {i["item_id"] for i in interactions}
            
            # Calcular similitud Jaccard
            intersection = len(user_items & other_items)
            union = len(user_items | other_items)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0:
                    similar_users[other_user_id] = similarity
        
        return similar_users


# Instancia global
_recommendation_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Obtiene la instancia global del motor de recomendaciones"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine

