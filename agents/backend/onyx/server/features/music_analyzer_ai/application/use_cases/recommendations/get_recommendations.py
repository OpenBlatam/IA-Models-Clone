"""
Use Case: Get Recommendations

Orchestrates the generation of track recommendations.
"""

from typing import List, Optional
import logging

from ...dto.recommendations import RecommendationDTO
from ...exceptions import TrackNotFoundException, RecommendationException
from ....domain.interfaces.repositories import ITrackRepository
from ....domain.interfaces.recommendations import IRecommendationService
from ...utils.dto_converters import convert_dict_list_to_recommendation_dtos

logger = logging.getLogger(__name__)


class GetRecommendationsUseCase:
    """
    Use case for getting track recommendations.
    
    This use case:
    1. Validates the seed track exists
    2. Gets recommendations based on method
    3. Returns formatted recommendations
    """
    
    def __init__(
        self,
        track_repository: ITrackRepository,
        recommendation_service: IRecommendationService
    ):
        self.track_repository = track_repository
        self.recommendation_service = recommendation_service
    
    async def execute(
        self,
        track_id: str,
        limit: int = 20,
        method: str = "similarity",
        mood: Optional[str] = None,
        genre: Optional[str] = None
    ) -> List[RecommendationDTO]:
        """
        Execute recommendation generation.
        
        Args:
            track_id: ID of the seed track
            limit: Maximum number of recommendations
            method: Recommendation method (similarity, mood, genre, contextual)
            mood: Target mood (for mood-based recommendations)
            genre: Target genre (for genre-based recommendations)
        
        Returns:
            List of RecommendationDTOs
        
        Raises:
            TrackNotFoundException: If seed track doesn't exist
            RecommendationException: If recommendation generation fails
        """
        logger.info(f"Getting recommendations for track: {track_id}, method: {method}")
        
        try:
            # 1. Validate track exists
            track = await self.track_repository.get_by_id(track_id)
            if not track:
                raise TrackNotFoundException(f"Track {track_id} not found")
            
            # 2. Get recommendations based on method
            if method == "similarity":
                recommendations_data = await self.recommendation_service.get_similar_tracks(
                    track_id, limit
                )
            elif method == "mood":
                recommendations_data = await self.recommendation_service.get_mood_based_recommendations(
                    track_id, mood, limit
                )
            elif method == "genre":
                recommendations_data = await self.recommendation_service.get_genre_based_recommendations(
                    track_id, genre, limit
                )
            else:
                raise RecommendationException(f"Unknown recommendation method: {method}")
            
            # 3. Convert to DTOs using helper
            recommendations = convert_dict_list_to_recommendation_dtos(recommendations_data)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except TrackNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Recommendation generation failed for track {track_id}: {e}")
            raise RecommendationException(f"Failed to generate recommendations: {str(e)}") from e

