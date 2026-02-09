"""
Collaborative Filtering Service - Sistema de recomendaciones colaborativas
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborativeFilteringService:
    """Servicio para recomendaciones colaborativas"""
    
    def __init__(self):
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.item_ratings: Dict[str, Dict[str, float]] = {}
        self.similarities: Dict[str, Dict[str, float]] = {}
    
    def record_preference(
        self,
        user_id: str,
        item_id: str,
        rating: float,  # 1-5
        item_type: str = "design"
    ) -> Dict[str, Any]:
        """Registrar preferencia de usuario"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id][item_id] = {
            "rating": rating,
            "type": item_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if item_id not in self.item_ratings:
            self.item_ratings[item_id] = {}
        
        self.item_ratings[item_id][user_id] = rating
        
        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "recorded_at": datetime.now().isoformat()
        }
    
    def find_similar_users(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Encontrar usuarios similares"""
        
        user_prefs = self.user_preferences.get(user_id, {})
        
        if not user_prefs:
            return []
        
        similarities = []
        
        for other_user_id, other_prefs in self.user_preferences.items():
            if other_user_id == user_id:
                continue
            
            similarity = self._calculate_user_similarity(user_prefs, other_prefs)
            
            if similarity > 0:
                similarities.append({
                    "user_id": other_user_id,
                    "similarity": similarity
                })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:limit]
    
    def _calculate_user_similarity(
        self,
        user1_prefs: Dict[str, Any],
        user2_prefs: Dict[str, Any]
    ) -> float:
        """Calcular similitud entre usuarios (cosine similarity simplificado)"""
        
        common_items = set(user1_prefs.keys()) & set(user2_prefs.keys())
        
        if not common_items:
            return 0.0
        
        # Calcular similitud basada en ratings comunes
        ratings1 = [user1_prefs[item]["rating"] for item in common_items]
        ratings2 = [user2_prefs[item]["rating"] for item in common_items]
        
        # Correlación simple
        mean1 = sum(ratings1) / len(ratings1)
        mean2 = sum(ratings2) / len(ratings2)
        
        numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))
        denominator = (
            (sum((r1 - mean1) ** 2 for r1 in ratings1) * sum((r2 - mean2) ** 2 for r2 in ratings2)) ** 0.5
        )
        
        if denominator == 0:
            return 0.0
        
        similarity = numerator / denominator
        return max(0.0, min(1.0, similarity))
    
    def recommend_items(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Recomendar items usando collaborative filtering"""
        
        # Encontrar usuarios similares
        similar_users = self.find_similar_users(user_id, limit=20)
        
        if not similar_users:
            return []
        
        # Items que usuarios similares han calificado pero el usuario no
        user_prefs = self.user_preferences.get(user_id, {})
        user_items = set(user_prefs.keys())
        
        recommendations = defaultdict(float)
        
        for similar_user in similar_users:
            similarity = similar_user["similarity"]
            similar_user_id = similar_user["user_id"]
            similar_prefs = self.user_preferences.get(similar_user_id, {})
            
            for item_id, item_data in similar_prefs.items():
                if item_id not in user_items:
                    recommendations[item_id] += similarity * item_data["rating"]
        
        # Ordenar por score
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                "item_id": item_id,
                "score": round(score, 2),
                "reason": "Recommended by similar users"
            }
            for item_id, score in sorted_recommendations[:limit]
        ]
    
    def get_item_similarity(
        self,
        item_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Obtener items similares"""
        
        item_ratings = self.item_ratings.get(item_id, {})
        
        if not item_ratings:
            return []
        
        similarities = []
        
        for other_item_id, other_ratings in self.item_ratings.items():
            if other_item_id == item_id:
                continue
            
            similarity = self._calculate_item_similarity(item_ratings, other_ratings)
            
            if similarity > 0:
                similarities.append({
                    "item_id": other_item_id,
                    "similarity": similarity
                })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:limit]
    
    def _calculate_item_similarity(
        self,
        item1_ratings: Dict[str, float],
        item2_ratings: Dict[str, float]
    ) -> float:
        """Calcular similitud entre items"""
        
        common_users = set(item1_ratings.keys()) & set(item2_ratings.keys())
        
        if not common_users:
            return 0.0
        
        ratings1 = [item1_ratings[user] for user in common_users]
        ratings2 = [item2_ratings[user] for user in common_users]
        
        mean1 = sum(ratings1) / len(ratings1)
        mean2 = sum(ratings2) / len(ratings2)
        
        numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))
        denominator = (
            (sum((r1 - mean1) ** 2 for r1 in ratings1) * sum((r2 - mean2) ** 2 for r2 in ratings2)) ** 0.5
        )
        
        if denominator == 0:
            return 0.0
        
        return max(0.0, min(1.0, numerator / denominator))




