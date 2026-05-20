"""
Use Case: Analyze Track

Orchestrates the analysis of a music track.
"""

from typing import Dict, Any, Optional
import logging

from ...dto.analysis import AnalysisResultDTO, TrackAnalysisDTO
from ...exceptions import TrackNotFoundException, AnalysisException
from ....domain.interfaces.repositories import ITrackRepository
from ....domain.interfaces.analysis import IAnalysisService
from ....domain.interfaces.coaching import ICoachingService
from ....domain.interfaces.spotify import ISpotifyService
from ...utils.data_extractors import (
    extract_track_name,
    extract_artists,
    extract_album_name,
)

logger = logging.getLogger(__name__)


class AnalyzeTrackUseCase:
    """
    Use case for analyzing a music track.
    
    This use case:
    1. Validates the track exists (by ID or name)
    2. Retrieves track data from Spotify
    3. Performs musical analysis
    4. Optionally generates coaching recommendations
    """
    
    def __init__(
        self,
        spotify_service: ISpotifyService,
        track_repository: ITrackRepository,
        analysis_service: IAnalysisService,
        coaching_service: ICoachingService = None
    ):
        self.spotify_service = spotify_service
        self.track_repository = track_repository
        self.analysis_service = analysis_service
        self.coaching_service = coaching_service
    
    async def _find_track_by_name(self, track_name: str) -> Optional[str]:
        """
        Find track ID by name using search.
        
        Args:
            track_name: Track name to search for
        
        Returns:
            Track ID if found, None otherwise
        """
        try:
            search_results = await self.track_repository.search(track_name, limit=1)
            tracks = search_results.get("tracks", {}).get("items", [])
            
            if tracks:
                track_id = tracks[0].get("id")
                logger.info(f"Found track ID {track_id} for name '{track_name}'")
                return track_id
            
            logger.warning(f"No track found for name '{track_name}'")
            return None
        except Exception as e:
            logger.error(f"Error searching for track '{track_name}': {e}")
            return None
    
    async def execute(
        self,
        track_id: Optional[str] = None,
        track_name: Optional[str] = None,
        include_coaching: bool = False
    ) -> AnalysisResultDTO:
        """
        Execute track analysis.
        
        Args:
            track_id: Spotify track ID to analyze (optional if track_name provided)
            track_name: Track name to search for (optional if track_id provided)
            include_coaching: Whether to include coaching recommendations
        
        Returns:
            AnalysisResultDTO with complete analysis
        
        Raises:
            TrackNotFoundException: If track doesn't exist
            AnalysisException: If analysis fails
        """
        # 1. Resolve track_id from track_name if needed
        if not track_id and track_name:
            logger.info(f"Searching for track by name: {track_name}")
            track_id = await self._find_track_by_name(track_name)
            if not track_id:
                raise TrackNotFoundException(f"Track '{track_name}' not found")
        
        if not track_id:
            raise TrackNotFoundException("Either track_id or track_name must be provided")
        
        logger.info(f"Starting analysis for track: {track_id}")
        
        try:
            # 2. Validate track exists
            track_data = await self.track_repository.get_by_id(track_id)
            if not track_data:
                # Try to get from Spotify directly
                track_data = await self.spotify_service.get_track(track_id)
                if not track_data:
                    raise TrackNotFoundException(f"Track {track_id} not found")
            
            # 3. Get complete Spotify data
            try:
                spotify_full_data = await self.spotify_service.get_track_full_analysis(track_id)
            except Exception as e:
                logger.warning(f"Could not get full analysis, building from parts: {e}")
                # Build from parts if full analysis fails
                audio_features = await self.track_repository.get_audio_features(track_id)
                audio_analysis = await self.track_repository.get_audio_analysis(track_id)
                spotify_full_data = {
                    "track_info": track_data,
                    "audio_features": audio_features or {},
                    "audio_analysis": audio_analysis or {}
                }
            
            # 4. Perform analysis
            try:
                analysis_result = await self.analysis_service.analyze_track(spotify_full_data)
            except Exception as e:
                logger.error(f"Analysis failed for track {track_id}: {e}")
                raise AnalysisException(f"Failed to analyze track: {str(e)}") from e
            
            # 5. Extract track basic info using helpers
            track_info = spotify_full_data.get("track_info", track_data)
            track_name = extract_track_name(track_info)
            artists = extract_artists(track_info)
            album = extract_album_name(track_info)
            duration_ms = track_info.get("duration_ms")
            duration_seconds = duration_ms / 1000.0 if duration_ms else None
            
            # 6. Generate coaching if requested
            coaching_data = None
            if include_coaching and self.coaching_service:
                try:
                    coaching_data = await self.coaching_service.generate_coaching_analysis(analysis_result)
                except Exception as e:
                    logger.warning(f"Coaching generation failed for track {track_id}: {e}")
                    # Don't fail the whole request if coaching fails
            
            # 7. Build result DTO
            result = AnalysisResultDTO(
                track_id=track_id,
                track_name=track_name,
                artists=artists,
                album=album,
                duration_seconds=duration_seconds,
                analysis=analysis_result,
                coaching=coaching_data
            )
            
            logger.info(f"Analysis completed successfully for track: {track_id}")
            return result
            
        except TrackNotFoundException:
            raise
        except AnalysisException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error analyzing track {track_id}: {e}")
            raise AnalysisException(f"Unexpected error during analysis: {str(e)}") from e

