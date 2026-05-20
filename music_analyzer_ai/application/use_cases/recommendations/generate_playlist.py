"""
Use Case: Generate Playlist

Orchestrates the generation of a playlist based on criteria.
"""

import logging
from typing import Dict, Any

from ...dto.recommendations import PlaylistDTO, RecommendationDTO
from ...exceptions import RecommendationException
from ....domain.interfaces.recommendations import IRecommendationService
from ...utils.validation_helpers import validate_numeric_range
from ...utils.dto_converters import convert_dict_list_to_recommendation_dtos

logger = logging.getLogger(__name__)


class GeneratePlaylistUseCase:
    """
    Use case for generating a playlist.
    
    This use case:
    1. Validates criteria
    2. Generates playlist based on criteria
    3. Returns formatted playlist
    """
    
    def __init__(self, recommendation_service: IRecommendationService):
        self.recommendation_service = recommendation_service
    
    async def execute(
        self,
        criteria: Dict[str, Any],
        length: int = 20
    ) -> PlaylistDTO:
        """
        Execute playlist generation.
        
        Args:
            criteria: Dictionary with genres, moods, energy_range, tempo_range, etc.
            length: Desired playlist length
        
        Returns:
            PlaylistDTO with generated tracks
        
        Raises:
            RecommendationException: If playlist generation fails
        """
        logger.info(f"Generating playlist with criteria: {criteria}, length: {length}")
        
        # Validate length using helper
        length = validate_numeric_range(length, 1, 100, "Playlist length", RecommendationException)
        
        try:
            # Generate playlist
            tracks_data = await self.recommendation_service.generate_playlist(criteria, length)
            
            # Convert to DTOs using helper
            tracks = convert_dict_list_to_recommendation_dtos(tracks_data)
            
            # Build playlist DTO
            playlist = PlaylistDTO(
                tracks=tracks,
                total_tracks=len(tracks),
                criteria=criteria
            )
            
            logger.info(f"Generated playlist with {len(tracks)} tracks")
            return playlist
            
        except Exception as e:
            logger.error(f"Playlist generation failed: {e}")
            raise RecommendationException(f"Failed to generate playlist: {str(e)}") from e

