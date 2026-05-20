"""
Coaching Service Adapter

Adapter that wraps MusicCoach to implement ICoachingService interface.
Uses asyncio to run synchronous methods in executor for true async behavior.
"""

import asyncio
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from ...domain.interfaces.coaching import ICoachingService

logger = logging.getLogger(__name__)


class CoachingServiceAdapter(ICoachingService):
    """
    Adapter that wraps existing MusicCoach to implement ICoachingService.
    """
    
    def __init__(self, music_coach, executor: ThreadPoolExecutor = None):
        """
        Initialize adapter with MusicCoach instance.
        
        Args:
            music_coach: Instance of MusicCoach
            executor: Optional thread pool executor for running sync code
        """
        self.music_coach = music_coach
        self.executor = executor or ThreadPoolExecutor(max_workers=3)
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
    
    async def generate_coaching_analysis(
        self,
        music_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate complete coaching analysis for a track"""
        try:
            coaching = await self._run_sync(
                self.music_coach.generate_coaching_analysis,
                music_analysis
            )
            return coaching
        except Exception as e:
            logger.error(f"Coaching generation failed: {e}")
            raise
    
    async def generate_learning_path(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate a step-by-step learning path"""
        try:
            coaching = await self.generate_coaching_analysis(music_analysis)
            return coaching.get("learning_path", [])
        except Exception as e:
            logger.warning(f"Failed to generate learning path: {e}")
            return []
    
    async def generate_practice_exercises(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate practice exercises based on track analysis"""
        try:
            coaching = await self.generate_coaching_analysis(music_analysis)
            return coaching.get("practice_exercises", [])
        except Exception as e:
            logger.warning(f"Failed to generate exercises: {e}")
            return []
    
    async def get_performance_tips(
        self,
        music_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get performance tips for a track"""
        try:
            coaching = await self.generate_coaching_analysis(music_analysis)
            return coaching.get("performance_tips", [])
        except Exception as e:
            logger.warning(f"Failed to get performance tips: {e}")
            return []
    
    async def assess_difficulty(
        self,
        music_analysis: Dict[str, Any]
    ) -> str:
        """Assess the difficulty level of a track"""
        try:
            coaching = await self.generate_coaching_analysis(music_analysis)
            overview = coaching.get("overview", {})
            return overview.get("difficulty_level", "Intermediate")
        except Exception as e:
            logger.warning(f"Failed to assess difficulty: {e}")
            return "Intermediate"

