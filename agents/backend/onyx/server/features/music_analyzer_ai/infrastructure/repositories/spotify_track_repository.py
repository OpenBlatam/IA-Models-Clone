"""
Spotify Track Repository

Implementation of ITrackRepository using SpotifyService.
This adapter wraps the existing SpotifyService to implement the domain interface.
Uses asyncio to run synchronous methods in executor for true async behavior.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ...domain.interfaces.repositories import ITrackRepository
from ...domain.interfaces.cache import ICacheService

logger = logging.getLogger(__name__)


class SpotifyTrackRepository(ITrackRepository):
    """
    Repository implementation for tracks using Spotify API.
    
    This is an adapter that wraps the existing SpotifyService
    to implement the ITrackRepository interface.
    """
    
    def __init__(
        self,
        spotify_service,
        cache_service: Optional[ICacheService] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize repository with Spotify service.
        
        Args:
            spotify_service: Instance of SpotifyService (or any service with compatible methods)
            cache_service: Optional cache service for caching results
            executor: Optional thread pool executor for running sync code
        """
        self.spotify_service = spotify_service
        self.cache_service = cache_service
        self.executor = executor or ThreadPoolExecutor(max_workers=5)
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
    
    async def get_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a track by its ID.
        
        Uses cache if available to reduce API calls.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Track data dictionary or None if not found
        """
        # Try cache first
        if self.cache_service:
            try:
                cache_key = f"track:{track_id}"
                cached = await self.cache_service.get("spotify", cache_key)
                if cached:
                    logger.debug(f"Cache hit for track {track_id}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed for track {track_id}: {e}")
        
        try:
            track = await self._run_sync(self.spotify_service.get_track, track_id)
            
            # Cache the result
            if track and self.cache_service:
                try:
                    cache_key = f"track:{track_id}"
                    await self.cache_service.set("spotify", cache_key, track, ttl=3600)  # 1 hour
                    logger.debug(f"Cached track {track_id}")
                except Exception as e:
                    logger.warning(f"Cache write failed for track {track_id}: {e}")
            
            return track
        except Exception as e:
            logger.warning(f"Failed to get track {track_id}: {e}")
            return None
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search for tracks.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Pagination offset
        
        Returns:
            Dictionary with search results in Spotify API format
        """
        try:
            # SpotifyService.search_track returns a list, we need to format it
            tracks = await self._run_sync(self.spotify_service.search_track, query, limit)
            
            # Format to match expected structure
            return {
                "tracks": {
                    "items": tracks,
                    "total": len(tracks),
                    "limit": limit,
                    "offset": offset
                }
            }
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return {
                "tracks": {
                    "items": [],
                    "total": 0,
                    "limit": limit,
                    "offset": offset
                }
            }
    
    async def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get audio features for a track.
        
        Uses cache if available to reduce API calls.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Audio features dictionary or None if not found
        """
        # Try cache first
        if self.cache_service:
            try:
                cache_key = f"audio_features:{track_id}"
                cached = await self.cache_service.get("spotify", cache_key)
                if cached:
                    logger.debug(f"Cache hit for audio features {track_id}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed for audio features {track_id}: {e}")
        
        try:
            features = await self._run_sync(self.spotify_service.get_track_audio_features, track_id)
            
            # Cache the result
            if features and self.cache_service:
                try:
                    cache_key = f"audio_features:{track_id}"
                    await self.cache_service.set("spotify", cache_key, features, ttl=86400)  # 24 hours
                    logger.debug(f"Cached audio features {track_id}")
                except Exception as e:
                    logger.warning(f"Cache write failed for audio features {track_id}: {e}")
            
            return features
        except Exception as e:
            logger.warning(f"Failed to get audio features for track {track_id}: {e}")
            return None
    
    async def get_audio_analysis(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed audio analysis for a track.
        
        Uses cache if available to reduce API calls.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Audio analysis dictionary or None if not found
        """
        # Try cache first
        if self.cache_service:
            try:
                cache_key = f"audio_analysis:{track_id}"
                cached = await self.cache_service.get("spotify", cache_key)
                if cached:
                    logger.debug(f"Cache hit for audio analysis {track_id}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read failed for audio analysis {track_id}: {e}")
        
        try:
            analysis = await self._run_sync(self.spotify_service.get_track_audio_analysis, track_id)
            
            # Cache the result
            if analysis and self.cache_service:
                try:
                    cache_key = f"audio_analysis:{track_id}"
                    await self.cache_service.set("spotify", cache_key, analysis, ttl=86400)  # 24 hours
                    logger.debug(f"Cached audio analysis {track_id}")
                except Exception as e:
                    logger.warning(f"Cache write failed for audio analysis {track_id}: {e}")
            
            return analysis
        except Exception as e:
            logger.warning(f"Failed to get audio analysis for track {track_id}: {e}")
            return None
    
    async def get_recommendations(
        self,
        seed_tracks: List[str],
        limit: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get track recommendations based on seed tracks.
        
        Args:
            seed_tracks: List of track IDs to use as seeds
            limit: Maximum number of recommendations
            **kwargs: Additional parameters (target_energy, target_tempo, etc.)
        
        Returns:
            List of recommended tracks
        """
        try:
            # SpotifyService.get_recommendations takes a single track_id
            # For multiple seeds, we'll use the first one
            if seed_tracks:
                recommendations = await self._run_sync(
                    self.spotify_service.get_recommendations,
                    seed_tracks[0],
                    limit
                )
                return recommendations
            return []
        except Exception as e:
            logger.warning(f"Failed to get recommendations: {e}")
            return []

