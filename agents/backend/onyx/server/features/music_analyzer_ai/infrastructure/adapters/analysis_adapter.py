"""
Analysis Service Adapter

Adapter that wraps MusicAnalyzer to implement IAnalysisService interface.
Uses asyncio to run synchronous methods in executor for true async behavior.
"""

import asyncio
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from ...domain.interfaces.analysis import IAnalysisService

logger = logging.getLogger(__name__)


class AnalysisServiceAdapter(IAnalysisService):
    """
    Adapter that wraps existing MusicAnalyzer to implement IAnalysisService.
    """
    
    def __init__(self, music_analyzer, executor: ThreadPoolExecutor = None):
        """
        Initialize adapter with MusicAnalyzer instance.
        
        Args:
            music_analyzer: Instance of MusicAnalyzer
            executor: Optional thread pool executor for running sync code
        """
        self.music_analyzer = music_analyzer
        self.executor = executor or ThreadPoolExecutor(max_workers=3)
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
    
    async def analyze_track(self, spotify_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a track with complete Spotify data.
        
        Args:
            spotify_data: Dictionary containing track_info, audio_features, and audio_analysis
        
        Returns:
            Complete analysis dictionary
        """
        try:
            analysis = await self._run_sync(self.music_analyzer.analyze_track, spotify_data)
            return analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    async def analyze_tracks_batch(
        self,
        tracks_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple tracks in batch.
        
        Args:
            tracks_data: List of track data dictionaries
        
        Returns:
            List of analysis dictionaries
        """
        results = []
        for track_data in tracks_data:
            try:
                analysis = await self.analyze_track(track_data)
                results.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze track in batch: {e}")
                results.append({})  # Empty result on error
        return results

