"""
Coaching Service Interfaces

Defines contracts for music coaching services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class ICoachingService(ABC):
    """Interface for music coaching service"""
    
    @abstractmethod
    async def generate_coaching_analysis(
        self,
        music_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete coaching analysis for a track.
        
        Args:
            music_analysis: Complete music analysis dictionary
        
        Returns:
            Coaching analysis with overview, learning path, exercises, etc.
        """
        pass
    
    @abstractmethod
    async def generate_learning_path(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate a step-by-step learning path.
        
        Args:
            music_analysis: Music analysis dictionary
        
        Returns:
            List of learning steps with descriptions and durations
        """
        pass
    
    @abstractmethod
    async def generate_practice_exercises(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate practice exercises based on track analysis.
        
        Args:
            music_analysis: Music analysis dictionary
        
        Returns:
            List of practice exercises with descriptions and difficulty
        """
        pass
    
    @abstractmethod
    async def get_performance_tips(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Get performance tips for a track.
        
        Args:
            music_analysis: Music analysis dictionary
        
        Returns:
            List of performance tips
        """
        pass
    
    @abstractmethod
    async def assess_difficulty(
        self,
        music_analysis: Dict[str, Any]
    ) -> str:
        """
        Assess the difficulty level of a track.
        
        Args:
            music_analysis: Music analysis dictionary
        
        Returns:
            Difficulty level (Beginner, Intermediate, Advanced)
        """
        pass




