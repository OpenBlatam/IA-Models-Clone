"""
Repository Interfaces

Defines contracts for data access operations.
Implementations should be in the infrastructure layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class ITrackRepository(ABC):
    """Interface for track data repository"""
    
    @abstractmethod
    async def get_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a track by its ID.
        
        Args:
            track_id: Spotify track ID
        
        Returns:
            Track data dictionary or None if not found
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Search for tracks.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            offset: Pagination offset
        
        Returns:
            Dictionary with search results
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
        pass


class IUserRepository(ABC):
    """Interface for user data repository"""
    
    @abstractmethod
    async def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user"""
        pass
    
    @abstractmethod
    async def update(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user data"""
        pass
    
    @abstractmethod
    async def get_favorites(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's favorite tracks"""
        pass
    
    @abstractmethod
    async def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's analysis history"""
        pass


class IPlaylistRepository(ABC):
    """Interface for playlist data repository"""
    
    @abstractmethod
    async def get_by_id(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Get playlist by ID"""
        pass
    
    @abstractmethod
    async def create(self, playlist_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new playlist"""
        pass
    
    @abstractmethod
    async def add_track(self, playlist_id: str, track_id: str) -> bool:
        """Add track to playlist"""
        pass
    
    @abstractmethod
    async def remove_track(self, playlist_id: str, track_id: str) -> bool:
        """Remove track from playlist"""
        pass
    
    @abstractmethod
    async def get_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Get all tracks in a playlist"""
        pass




