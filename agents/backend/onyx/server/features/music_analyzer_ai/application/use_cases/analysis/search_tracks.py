"""
Use Case: Search Tracks

Orchestrates the search for music tracks.
"""

from typing import List, Dict, Any
import logging

from ...dto.analysis import TrackAnalysisDTO
from ...exceptions import UseCaseException
from ....domain.interfaces.repositories import ITrackRepository
from ...utils.validation_helpers import validate_string_not_empty, validate_numeric_range
from ...utils.dto_converters import convert_dict_list_to_track_analysis_dtos

logger = logging.getLogger(__name__)


class SearchTracksUseCase:
    """
    Use case for searching music tracks.
    
    This use case:
    1. Validates search query
    2. Searches for tracks
    3. Returns formatted results
    """
    
    def __init__(self, track_repository: ITrackRepository):
        self.track_repository = track_repository
    
    async def execute(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Execute track search.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Pagination offset
        
        Returns:
            Dictionary with search results and metadata
        
        Raises:
            UseCaseException: If search fails
        """
        logger.info(f"Searching tracks with query: {query}")
        
        # Validate parameters using helpers
        query = validate_string_not_empty(query, "Search query")
        limit = validate_numeric_range(limit, 1, 50, "Limit")
        
        try:
            # Search tracks
            search_results = await self.track_repository.search(query, limit, offset)
            
            # Extract tracks from results
            tracks_data = search_results.get("tracks", {}).get("items", [])
            
            # Convert to DTOs using helper
            tracks = convert_dict_list_to_track_analysis_dtos(tracks_data)
            
            # Build response
            total = search_results.get("tracks", {}).get("total", len(tracks))
            
            result = {
                "success": True,
                "query": query,
                "results": [track.__dict__ for track in tracks],
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
            logger.info(f"Search completed: {len(tracks)} tracks found")
            return result
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise UseCaseException(f"Search failed: {str(e)}") from e

