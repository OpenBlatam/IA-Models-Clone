"""
Advanced Recommendation System
Enhanced recommendation algorithms with multiple strategies
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedRecommender:
    """
    Advanced recommendation system with multiple strategies
    """
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.item_features: Dict[str, np.ndarray] = {}
        self.interaction_matrix: Optional[np.ndarray] = None
    
    def collaborative_filtering(
        self,
        user_id: str,
        item_ids: List[str],
        k: int = 10,
        similarity_metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """Collaborative filtering recommendations"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarities
        similarities = []
        for item_id in item_ids:
            if item_id in self.item_features:
                similarity = self._calculate_similarity(
                    user_profile.get("preferences", np.array([])),
                    self.item_features[item_id],
                    similarity_metric
                )
                similarities.append({
                    "item_id": item_id,
                    "similarity": similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:k]
    
    def content_based_filtering(
        self,
        item_id: str,
        item_ids: List[str],
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Content-based filtering"""
        if item_id not in self.item_features:
            return []
        
        target_features = self.item_features[item_id]
        
        # Calculate similarities
        similarities = []
        for other_id in item_ids:
            if other_id != item_id and other_id in self.item_features:
                similarity = np.dot(
                    target_features,
                    self.item_features[other_id]
                ) / (
                    np.linalg.norm(target_features) *
                    np.linalg.norm(self.item_features[other_id])
                )
                similarities.append({
                    "item_id": other_id,
                    "similarity": float(similarity)
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:k]
    
    def hybrid_recommendation(
        self,
        user_id: Optional[str],
        item_id: str,
        item_ids: List[str],
        k: int = 10,
        collaborative_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Hybrid recommendation combining collaborative and content-based"""
        results = {}
        
        # Collaborative filtering
        if user_id:
            cf_results = self.collaborative_filtering(user_id, item_ids, k=k*2)
            for result in cf_results:
                item_id_key = result["item_id"]
                if item_id_key not in results:
                    results[item_id_key] = {"item_id": item_id_key, "score": 0.0}
                results[item_id_key]["score"] += result["similarity"] * collaborative_weight
        
        # Content-based filtering
        cb_results = self.content_based_filtering(item_id, item_ids, k=k*2)
        for result in cb_results:
            item_id_key = result["item_id"]
            if item_id_key not in results:
                results[item_id_key] = {"item_id": item_id_key, "score": 0.0}
            results[item_id_key]["score"] += result["similarity"] * (1 - collaborative_weight)
        
        # Sort by combined score
        final_results = list(results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:k]
    
    def _calculate_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Calculate similarity between vectors"""
        if metric == "cosine":
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        elif metric == "euclidean":
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def update_user_profile(
        self,
        user_id: str,
        preferences: np.ndarray,
        interactions: Optional[Dict[str, float]] = None
    ):
        """Update user profile"""
        self.user_profiles[user_id] = {
            "preferences": preferences,
            "interactions": interactions or {}
        }
    
    def update_item_features(
        self,
        item_id: str,
        features: np.ndarray
    ):
        """Update item features"""
        self.item_features[item_id] = features

