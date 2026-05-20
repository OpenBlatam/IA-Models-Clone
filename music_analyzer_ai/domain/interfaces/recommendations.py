"""
Recommendation Service Interfaces

Defines contracts for recommendation services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class IRecommendationService(ABC):
    """Interface for music recommendation service"""
    
    @abstractmethod
    async def get_similar_tracks(
        self,
        track_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get tracks similar to a given track.
        
        Args:
            track_id: ID of the reference track
            limit: Maximum number of recommendations
        
        Returns:
            List of similar tracks with similarity scores
        """
        pass
    
    @abstractmethod
    async def get_mood_based_recommendations(
        self,
        track_id: str,
        mood: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on mood.
        
        Args:
            track_id: ID of the reference track
            mood: Target mood (happy, sad, energetic, calm, etc.)
            limit: Maximum number of recommendations
        
        Returns:
            List of mood-based recommendations
        """
        pass
    
    @abstractmethod
    async def get_genre_based_recommendations(
        self,
        track_id: str,
        genre: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on genre.
        
        Args:
            track_id: ID of the reference track
            genre: Target genre
            limit: Maximum number of recommendations
        
        Returns:
            List of genre-based recommendations
        """
        pass
    
    @abstractmethod
    async def get_contextual_recommendations(
        self,
        context: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on context (time of day, activity, etc.).
        
        Args:
            context: Context dictionary with time_of_day, activity, mood, etc.
            limit: Maximum number of recommendations
        
        Returns:
            List of contextual recommendations
        """
        pass
    
    @abstractmethod
    async def generate_playlist(
        self,
        criteria: Dict[str, Any],
        length: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate a playlist based on criteria.
        
        Args:
            criteria: Dictionary with genres, moods, energy_range, tempo_range, etc.
            length: Desired playlist length
        
        Returns:
            List of tracks for the playlist
        """
        pass




