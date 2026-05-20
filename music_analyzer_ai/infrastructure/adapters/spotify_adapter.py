"""
Spotify Service Adapter

Adapter that wraps SpotifyService to implement ISpotifyService interface.
Uses asyncio to run synchronous methods in executor for true async behavior.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ...domain.interfaces.spotify import ISpotifyService

logger = logging.getLogger(__name__)


class SpotifyServiceAdapter(ISpotifyService):
    """
    Adapter that wraps existing SpotifyService to implement ISpotifyService.
    
    This allows the existing SpotifyService to be used with the new architecture
    without modifying the original service.
    """
    
    def __init__(self, spotify_service, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize adapter with SpotifyService instance.
        
        Args:
            spotify_service: Instance of SpotifyService
            executor: Optional thread pool executor for running sync code
        """
        self.spotify_service = spotify_service
        self.executor = executor or ThreadPoolExecutor(max_workers=5)
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
    
    async def search_track(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search for tracks on Spotify"""
        try:
            tracks = await self._run_sync(self.spotify_service.search_track, query, limit)
            return {
                "tracks": {
                    "items": tracks,
                    "total": len(tracks),
                    "limit": limit,
                    "offset": offset
                }
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"tracks": {"items": [], "total": 0, "limit": limit, "offset": offset}}
    
    async def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track information by ID"""
        try:
            return await self._run_sync(self.spotify_service.get_track, track_id)
        except Exception as e:
            logger.warning(f"Failed to get track {track_id}: {e}")
            return None
    
    async def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get audio features for a track"""
        try:
            return await self._run_sync(self.spotify_service.get_track_audio_features, track_id)
        except Exception as e:
            logger.warning(f"Failed to get audio features for {track_id}: {e}")
            return None
    
    async def get_audio_analysis(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed audio analysis for a track"""
        try:
            return await self._run_sync(self.spotify_service.get_track_audio_analysis, track_id)
        except Exception as e:
            logger.warning(f"Failed to get audio analysis for {track_id}: {e}")
            return None
    
    async def get_track_full_analysis(self, track_id: str) -> Dict[str, Any]:
        """Get complete analysis data for a track"""
        try:
            return await self._run_sync(self.spotify_service.get_track_full_analysis, track_id)
        except Exception as e:
            logger.error(f"Failed to get full analysis for {track_id}: {e}")
            raise
    
    async def get_recommendations(
        self,
        seed_tracks: List[str],
        limit: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get track recommendations from Spotify"""
        try:
            if seed_tracks:
                return await self._run_sync(
                    self.spotify_service.get_recommendations,
                    seed_tracks[0],
                    limit
                )
            return []
        except Exception as e:
            logger.warning(f"Failed to get recommendations: {e}")
            return []
    
    async def get_artist(self, artist_id: str) -> Optional[Dict[str, Any]]:
        """Get artist information by ID"""
        try:
            return await self._run_sync(self.spotify_service.get_artist, artist_id)
        except Exception as e:
            logger.warning(f"Failed to get artist {artist_id}: {e}")
            return None
    
    async def get_artist_tracks(
        self,
        artist_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get tracks by an artist"""
        try:
            # Note: SpotifyService might not have this method directly
            # This is a placeholder for future implementation
            logger.warning("get_artist_tracks not fully implemented")
            return []
        except Exception as e:
            logger.warning(f"Failed to get artist tracks: {e}")
            return []

