"""
Recommendation Service Adapter

Adapter that wraps recommendation services to implement IRecommendationService interface.
"""

import logging
from typing import Dict, List, Any, Optional

from ...domain.interfaces.recommendations import IRecommendationService

logger = logging.getLogger(__name__)


class RecommendationServiceAdapter(IRecommendationService):
    """
    Adapter that wraps existing recommendation services to implement IRecommendationService.
    """
    
    def __init__(self, intelligent_recommender, contextual_recommender=None):
        """
        Initialize adapter with recommendation services.
        
        Args:
            intelligent_recommender: Instance of IntelligentRecommender
            contextual_recommender: Optional instance of ContextualRecommender
        """
        self.intelligent_recommender = intelligent_recommender
        self.contextual_recommender = contextual_recommender
    
    async def get_similar_tracks(
        self,
        track_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get tracks similar to a given track"""
        try:
            # Use intelligent_recommender's similarity method if available
            # Otherwise fall back to Spotify recommendations
            recommendations = self.intelligent_recommender.get_similar_tracks(
                track_id, limit
            ) if hasattr(self.intelligent_recommender, 'get_similar_tracks') else []
            return recommendations
        except Exception as e:
            logger.warning(f"Failed to get similar tracks: {e}")
            return []
    
    async def get_mood_based_recommendations(
        self,
        track_id: str,
        mood: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on mood"""
        try:
            if self.contextual_recommender:
                recommendations = self.contextual_recommender.get_mood_recommendations(
                    track_id, mood, limit
                ) if hasattr(self.contextual_recommender, 'get_mood_recommendations') else []
                return recommendations
            return []
        except Exception as e:
            logger.warning(f"Failed to get mood-based recommendations: {e}")
            return []
    
    async def get_genre_based_recommendations(
        self,
        track_id: str,
        genre: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on genre"""
        try:
            recommendations = self.intelligent_recommender.get_genre_recommendations(
                track_id, genre, limit
            ) if hasattr(self.intelligent_recommender, 'get_genre_recommendations') else []
            return recommendations
        except Exception as e:
            logger.warning(f"Failed to get genre-based recommendations: {e}")
            return []
    
    async def get_contextual_recommendations(
        self,
        context: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on context"""
        try:
            if self.contextual_recommender:
                recommendations = self.contextual_recommender.get_contextual_recommendations(
                    context, limit
                ) if hasattr(self.contextual_recommender, 'get_contextual_recommendations') else []
                return recommendations
            return []
        except Exception as e:
            logger.warning(f"Failed to get contextual recommendations: {e}")
            return []
    
    async def generate_playlist(
        self,
        criteria: Dict[str, Any],
        length: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate a playlist based on criteria"""
        try:
            # Use intelligent_recommender to generate playlist
            playlist = self.intelligent_recommender.generate_playlist(
                criteria, length
            ) if hasattr(self.intelligent_recommender, 'generate_playlist') else []
            return playlist
        except Exception as e:
            logger.warning(f"Failed to generate playlist: {e}")
            return []




