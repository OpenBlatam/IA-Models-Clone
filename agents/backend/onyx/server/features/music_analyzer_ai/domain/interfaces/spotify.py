"""
Spotify Service Interface

Defines contract for Spotify API integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class ISpotifyService(ABC):
    """Interface for Spotify API service"""
    
    @abstractmethod
    async def search_track(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search for tracks on Spotify.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Pagination offset
        
        Returns:
            Search results dictionary
        """
        pass
    
    @abstractmethod
    async def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get track information by ID.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Track information dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get audio features for a track.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Audio features dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def get_audio_analysis(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed audio analysis for a track.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Audio analysis dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def get_track_full_analysis(self, track_id: str) -> Dict[str, Any]:
        """
        Get complete analysis data for a track (track info + features + analysis).
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Dictionary with track_info, audio_features, and audio_analysis
        """
        pass
    
    @abstractmethod
    async def get_recommendations(
        self,
        seed_tracks: List[str],
        limit: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get track recommendations from Spotify.
        
        Args:
            seed_tracks: List of track IDs to use as seeds
            limit: Maximum number of recommendations
            **kwargs: Additional parameters (target_energy, target_tempo, etc.)
        
        Returns:
            List of recommended tracks
        """
        pass
    
    @abstractmethod
    async def get_artist(self, artist_id: str) -> Optional[Dict[str, Any]]:
        """
        Get artist information by ID.
        
        Args:
            artist_id: Spotify artist ID
        
        Returns:
            Artist information dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def get_artist_tracks(
        self,
        artist_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get tracks by an artist.
        
        Args:
            artist_id: Spotify artist ID
            limit: Maximum number of tracks
        
        Returns:
            List of track dictionaries
        """
        pass




