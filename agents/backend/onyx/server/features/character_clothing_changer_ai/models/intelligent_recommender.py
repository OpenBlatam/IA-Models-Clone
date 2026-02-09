"""
Intelligent Recommender for Flux2 Clothing Changer
====================================================

Intelligent recommendations based on user behavior and preferences.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Recommendation result."""
    item: str
    score: float
    reason: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IntelligentRecommender:
    """Intelligent recommendation system."""
    
    def __init__(
        self,
        enable_collaborative: bool = True,
        enable_content_based: bool = True,
    ):
        """
        Initialize intelligent recommender.
        
        Args:
            enable_collaborative: Enable collaborative filtering
            enable_content_based: Enable content-based filtering
        """
        self.enable_collaborative = enable_collaborative
        self.enable_content_based = enable_content_based
        
        # User preferences
        self.user_preferences: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Item features
        self.item_features: Dict[str, Dict[str, Any]] = {}
        
        # User-item interactions
        self.user_item_interactions: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Popular items
        self.item_popularity: Counter = Counter()
        
        # Similarity cache
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
    
    def record_interaction(
        self,
        user_id: str,
        item: str,
        rating: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record user-item interaction.
        
        Args:
            user_id: User identifier
            item: Item identifier
            rating: Rating (0.0 to 1.0)
            metadata: Optional metadata
        """
        self.user_item_interactions[user_id][item] = rating
        self.item_popularity[item] += 1
        
        # Update user preferences
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    if key not in self.user_preferences[user_id]:
                        self.user_preferences[user_id][key] = []
                    self.user_preferences[user_id][key].append(value)
    
    def set_item_features(
        self,
        item: str,
        features: Dict[str, Any],
    ) -> None:
        """
        Set item features.
        
        Args:
            item: Item identifier
            features: Item features
        """
        self.item_features[item] = features
    
    def recommend(
        self,
        user_id: str,
        num_recommendations: int = 10,
        item_type: Optional[str] = None,
    ) -> List[Recommendation]:
        """
        Generate recommendations for user.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations
            item_type: Optional item type filter
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Collaborative filtering
        if self.enable_collaborative:
            collab_recs = self._collaborative_filtering(user_id, item_type)
            recommendations.extend(collab_recs)
        
        # Content-based filtering
        if self.enable_content_based:
            content_recs = self._content_based_filtering(user_id, item_type)
            recommendations.extend(content_recs)
        
        # Popular items fallback
        if not recommendations:
            popular_recs = self._popular_items_filtering(item_type)
            recommendations.extend(popular_recs)
        
        # Aggregate and sort
        aggregated = self._aggregate_recommendations(recommendations)
        
        # Sort by score
        aggregated.sort(key=lambda r: r.score, reverse=True)
        
        return aggregated[:num_recommendations]
    
    def _collaborative_filtering(
        self,
        user_id: str,
        item_type: Optional[str],
    ) -> List[Recommendation]:
        """Collaborative filtering recommendations."""
        if user_id not in self.user_item_interactions:
            return []
        
        user_items = set(self.user_item_interactions[user_id].keys())
        
        # Find similar users
        similar_users = self._find_similar_users(user_id)
        
        recommendations = []
        item_scores = defaultdict(float)
        
        for similar_user, similarity in similar_users:
            for item, rating in self.user_item_interactions[similar_user].items():
                if item not in user_items and (not item_type or item.startswith(item_type)):
                    item_scores[item] += similarity * rating
        
        for item, score in item_scores.items():
            recommendations.append(Recommendation(
                item=item,
                score=score,
                reason="Similar users also liked this",
            ))
        
        return recommendations
    
    def _content_based_filtering(
        self,
        user_id: str,
        item_type: Optional[str],
    ) -> List[Recommendation]:
        """Content-based filtering recommendations."""
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            return []
        
        user_prefs = self.user_preferences[user_id]
        recommendations = []
        
        for item, features in self.item_features.items():
            if item_type and not item.startswith(item_type):
                continue
            
            # Calculate similarity
            similarity = self._calculate_feature_similarity(user_prefs, features)
            
            if similarity > 0.0:
                recommendations.append(Recommendation(
                    item=item,
                    score=similarity,
                    reason="Matches your preferences",
                ))
        
        return recommendations
    
    def _popular_items_filtering(
        self,
        item_type: Optional[str],
    ) -> List[Recommendation]:
        """Popular items recommendations."""
        items = [
            item for item, count in self.item_popularity.most_common(20)
            if not item_type or item.startswith(item_type)
        ]
        
        return [
            Recommendation(
                item=item,
                score=self.item_popularity[item] / max(self.item_popularity.values()) if self.item_popularity else 0.0,
                reason="Popular choice",
            )
            for item in items
        ]
    
    def _find_similar_users(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar users."""
        if user_id not in self.user_item_interactions:
            return []
        
        user_items = self.user_item_interactions[user_id]
        similarities = []
        
        for other_user, other_items in self.user_item_interactions.items():
            if other_user == user_id:
                continue
            
            similarity = self._calculate_user_similarity(user_items, other_items)
            if similarity > 0.0:
                similarities.append((other_user, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_user_similarity(
        self,
        user1_items: Dict[str, float],
        user2_items: Dict[str, float],
    ) -> float:
        """Calculate similarity between users."""
        common_items = set(user1_items.keys()) & set(user2_items.keys())
        
        if not common_items:
            return 0.0
        
        # Cosine similarity
        vec1 = [user1_items[item] for item in common_items]
        vec2 = [user2_items[item] for item in common_items]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_feature_similarity(
        self,
        user_prefs: Dict[str, List[float]],
        item_features: Dict[str, Any],
    ) -> float:
        """Calculate similarity between user preferences and item features."""
        similarity = 0.0
        matches = 0
        
        for key, pref_values in user_prefs.items():
            if key in item_features:
                item_value = item_features[key]
                if isinstance(item_value, (int, float)) and pref_values:
                    avg_pref = sum(pref_values) / len(pref_values)
                    # Simple similarity (can be enhanced)
                    similarity += 1.0 - abs(avg_pref - item_value) / max(abs(avg_pref), abs(item_value), 1.0)
                    matches += 1
        
        return similarity / matches if matches > 0 else 0.0
    
    def _aggregate_recommendations(
        self,
        recommendations: List[Recommendation],
    ) -> List[Recommendation]:
        """Aggregate duplicate recommendations."""
        aggregated = {}
        
        for rec in recommendations:
            if rec.item in aggregated:
                # Combine scores
                aggregated[rec.item].score = (aggregated[rec.item].score + rec.score) / 2
                aggregated[rec.item].reason += f", {rec.reason}"
            else:
                aggregated[rec.item] = rec
        
        return list(aggregated.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recommender statistics."""
        return {
            "total_users": len(self.user_item_interactions),
            "total_items": len(self.item_features),
            "total_interactions": sum(len(items) for items in self.user_item_interactions.values()),
            "popular_items": dict(self.item_popularity.most_common(10)),
        }


